"""Tests for the Decision agent (P4). No Ollama, no data files: hand-made diagnoses and incidents.

Each test pins one rule the rest of the team relies on: the 4 scenarios of the deck, the table, the
guardrails that stop a wrong automatic action, and the hand-off to the Action agent (P5).
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from agents.decision import policy
from agents.decision.agent import DecisionAgent
from agents.decision.config import DecisionConfig
from agents.decision.inbox import ActionInbox, InboxError
from agents.decision.log import DecisionLog
from agents.decision.node import make_decision_node, route_after_decision
from common.schemas import (Action, ActionDecision, ActionStatus, Alert, Category, Decision, Diagnosis,
                            Evidence, Hypothesis, Incident, Route, Severity)

NOON = datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc)
NIGHT = datetime(2026, 1, 20, 2, 0, tzinfo=timezone.utc)
C = Category


# ------------------------------------------------------------------ helpers
def diag(category=C.FLAKY_TEST, confidence=0.9, incident_id="INC-1", **kw) -> Diagnosis:
    action = kw.pop("suggested_action", {C.FLAKY_TEST: "rerun", C.BAD_DEPLOY: "rollback", C.CONFIG_CHANGE: "rollback",
                                         C.DATA_DRIFT: "retrain"}.get(category, "escalate"))
    return Diagnosis(incident_id=incident_id, category=category, explanation="because [E1]",
                     evidence=[Evidence("E1", "metric_step", "latency jumped 40%", "metrics")],
                     confidence=confidence, hypotheses=[Hypothesis(category, confidence)],
                     suggested_action=action, **kw)


def incident(source="ci", project="proj-a", incident_id="INC-1", features=("build_failed",)) -> Incident:
    t = NOON
    alert = Alert(detector="x", source=source, feature=features[0], score=2.0, threshold=1.0, timestamp=t, value=1.0,
                  dataset={"ci": "travis", "model": "nyc", "metrics": "deploy", "logs": "hdfs"}[source],
                  details={"project": project})
    return Incident(id=incident_id, source=source, features=list(features), alert_ids=[alert.id], start=t, end=t,
                    duration_hours=1.0, peak_ratio=2.0, max_voters=1, severity=Severity.HIGH,
                    decision=Decision.INCIDENT, summary=f"{source} incident", alerts=[alert], dataset=alert.dataset)


@pytest.fixture
def log(tmp_path):
    return DecisionLog(tmp_path / "decision_log.jsonl", tmp_path / "history.jsonl")


@pytest.fixture
def agent():
    return DecisionAgent()                       # stateless: no log, no history


def decide(agent, d, inc=None, now=NOON):
    return agent.decide(d, inc, now)


# ------------------------------------------------------------------ the action risk table
def test_table_is_read_from_the_knowledge_base():
    rows = policy.load()
    assert set(rows) == set(Category)
    assert rows[C.FLAKY_TEST].action is Action.RERUN and rows[C.FLAKY_TEST].bar == 0.60
    assert rows[C.BAD_DEPLOY].bar == 0.80 and rows[C.CONFIG_CHANGE].bar == 0.80
    assert rows[C.DATA_DRIFT].bar == 0.70 and rows[C.DATA_DRIFT].daytime_only
    for never in (C.CODE_REGRESSION, C.INFRA, C.DATA_QUALITY, C.UNKNOWN):
        assert rows[never].never_alone and rows[never].action is Action.ESCALATE
    assert rows[C.CODE_REGRESSION].detail == "author"


def test_unreadable_table_raises_instead_of_guessing():
    good = (policy.paths.ACTION_RISK_TABLE).read_text(encoding="utf-8")
    with pytest.raises(policy.PolicyError, match="unknown cause"):
        policy.parse(good.replace("| flaky_test ", "| flaky_tests "))
    with pytest.raises(policy.PolicyError, match="risk"):
        policy.parse(good.replace("| none   |", "| nope   |", 1))
    with pytest.raises(policy.PolicyError, match="no row for"):
        policy.parse("\n".join(line for line in good.splitlines() if "| infra " not in line))


# ------------------------------------------------------------------ the 4 scenarios of the deck
def test_scenario1_flaky_test_is_rerun_alone(agent):
    d = decide(agent, diag(C.FLAKY_TEST, 0.9), incident("ci"))
    assert (d.action, d.route, d.status) == (Action.RERUN, Route.ACT, ActionStatus.APPROVED)
    assert d.risk == "none" and d.decided_by == "policy"
    assert any("safe to do alone" in r for r in d.reasons)


def test_scenario2_bad_deploy_rollback_waits_for_a_human(agent):
    d = decide(agent, diag(C.BAD_DEPLOY, 0.93, suspect="release-058"), incident("metrics"))
    assert (d.action, d.route, d.status) == (Action.ROLLBACK, Route.APPROVAL, ActionStatus.PENDING_APPROVAL)
    assert d.target == "release-058" and d.rule == "gate:rollback"
    assert any("needs a human" in r for r in d.reasons)


def test_scenario3_drift_retrains_by_day_but_a_human_approves_the_deploy(agent):
    d = decide(agent, diag(C.DATA_DRIFT, 0.8), incident("model"))
    assert (d.action, d.route, d.status) == (Action.RETRAIN, Route.ACT, ActionStatus.APPROVED)
    assert d.constraints["human_before_deploy"] is True


def test_scenario4_unknown_failure_escalates(agent):
    d = decide(agent, diag(C.UNKNOWN, 0.35, needs_human=True, suggested_action="escalate"), incident("logs"))
    assert (d.action, d.route) == (Action.ESCALATE, Route.ESCALATE)
    assert d.rule.startswith("floor") and d.status is ActionStatus.APPROVED     # notifying is safe: P5 may do it
    assert any("< 0.50" in r for r in d.reasons)


# ------------------------------------------------------------------ the table's rules
def test_auto_rollback_switch_follows_the_table(agent):
    on = DecisionAgent(DecisionConfig(auto_rollback=True))
    assert decide(on, diag(C.BAD_DEPLOY, 0.85, suspect="r1")).route is Route.ACT
    assert decide(on, diag(C.CONFIG_CHANGE, 0.79, suspect="c1")).route is Route.APPROVAL      # under the 0.80 bar


@pytest.mark.parametrize("cause", [C.CODE_REGRESSION, C.INFRA, C.DATA_QUALITY, C.UNKNOWN])
def test_never_alone_causes_always_escalate(agent, cause):
    d = decide(agent, diag(cause, 0.99))
    assert (d.action, d.route) == (Action.ESCALATE, Route.ESCALATE) and d.rule.endswith("never_alone")


def test_code_regression_tells_p5_to_notify_the_author(agent):
    assert decide(agent, diag(C.CODE_REGRESSION, 0.9)).constraints["notify"] == "author"


def test_below_the_bar_asks_a_human_with_the_proposed_action(agent):
    d = decide(agent, diag(C.FLAKY_TEST, 0.55), incident("ci"))
    assert (d.action, d.route) == (Action.RERUN, Route.APPROVAL) and d.rule == "table:below_bar"


def test_floor_beats_everything(agent):
    d = decide(agent, diag(C.FLAKY_TEST, 0.49), incident("ci"))
    assert (d.action, d.route) == (Action.ESCALATE, Route.ESCALATE) and d.risk == "none"


def test_needs_human_means_ask_a_human_whatever_the_confidence(agent):
    d = decide(agent, diag(C.FLAKY_TEST, 0.95, needs_human=True), incident("ci"))
    assert d.route is Route.APPROVAL and d.rule == "human:needs_human"


def test_llm_disagreement_goes_to_a_human_with_both_causes(agent):
    d = decide(agent, diag(C.FLAKY_TEST, 0.9, warnings=["LLM argues for code_regression: the test is deterministic"]),
               incident("ci"))
    assert d.route is Route.APPROVAL and d.rule == "human:llm_disagrees"
    assert any("code_regression" in r for r in d.reasons)
    assert d.snapshot["warnings"][0].startswith("LLM argues for")


def test_the_table_wins_over_the_suggestion_of_diagnosis(agent):
    d = decide(agent, diag(C.FLAKY_TEST, 0.9, suggested_action="rollback"), incident("ci"))
    assert d.action is Action.RERUN and any("the table wins" in r for r in d.reasons)


def test_retrain_at_night_waits_and_below_bar_asks(agent):
    assert decide(agent, diag(C.DATA_DRIFT, 0.9), incident("model"), NIGHT).rule == "gate:night"
    assert decide(agent, diag(C.DATA_DRIFT, 0.65), incident("model")).rule == "table:below_bar"


# ------------------------------------------------------------------ preconditions
def test_rollback_without_a_suspect_escalates(agent):
    d = decide(agent, diag(C.BAD_DEPLOY, 0.95, suspect=""))
    assert (d.action, d.route, d.rule) == (Action.ESCALATE, Route.ESCALATE, "precondition")


def test_rerun_only_for_ci_and_retrain_only_for_models(agent):
    assert decide(agent, diag(C.FLAKY_TEST, 0.9), incident("metrics")).rule == "precondition"
    assert decide(agent, diag(C.DATA_DRIFT, 0.9), incident("ci")).rule == "precondition"


# ------------------------------------------------------------------ no-fault: never silence a real fault
def nofault(conf=0.9, **kw):
    return diag(C.UNKNOWN, conf, sub_cause=kw.pop("sub_cause", "recovery"), suggested_action="none", **kw)


def test_recognised_no_fault_is_closed_and_nobody_is_paged(agent):
    d = decide(agent, nofault(0.9))
    assert (d.action, d.route, d.status) == (Action.NONE, Route.NONE, ActionStatus.CLOSED)


@pytest.mark.parametrize("bad", [nofault(0.6), nofault(0.9, sub_cause=""), nofault(0.9, needs_human=True)])
def test_unsure_no_fault_goes_to_a_human(agent, bad):
    assert decide(agent, bad).route is Route.ESCALATE


# ------------------------------------------------------------------ guards that remember
def test_rerun_budget_a_second_failure_is_a_real_bug(log):
    ag = DecisionAgent(log=log)
    first = ag.run([diag(C.FLAKY_TEST, 0.9, incident_id="A")], [incident("ci", "proj-a", "A")], NOON)[0]
    assert first.action is Action.RERUN
    again = ag.run([diag(C.FLAKY_TEST, 0.9, incident_id="B")], [incident("ci", "proj-a", "B")], NOON + timedelta(minutes=20))[0]
    assert (again.action, again.route, again.rule) == (Action.ESCALATE, Route.ESCALATE, "guard:rerun_budget")
    other = ag.run([diag(C.FLAKY_TEST, 0.9, incident_id="C")], [incident("ci", "proj-b", "C")], NOON + timedelta(minutes=30))[0]
    assert other.action is Action.RERUN                               # another project is not affected


def test_same_release_is_not_asked_twice_in_one_batch(agent):
    a = diag(C.BAD_DEPLOY, 0.95, incident_id="A", suspect="release-1")
    b = diag(C.BAD_DEPLOY, 0.85, incident_id="B", suspect="release-1")
    out = agent.run([a, b], now=NOON)
    assert [d.incident_id for d in out] == ["A", "B"]                  # input order is kept
    assert out[0].route is Route.APPROVAL and (out[1].route, out[1].rule) == (Route.NONE, "guard:duplicate")


def test_cooldown_expires(log):
    ag = DecisionAgent(DecisionConfig(auto_rollback=True), log=log)
    d = lambda i: diag(C.BAD_DEPLOY, 0.95, incident_id=i, suspect="release-1")
    assert ag.run([d("A")], now=NOON)[0].route is Route.ACT
    assert ag.run([d("B")], now=NOON + timedelta(minutes=30))[0].rule == "guard:duplicate"
    assert ag.run([d("C")], now=NOON + timedelta(minutes=90))[0].route is Route.ACT


def test_circuit_breaker_after_too_many_automatic_actions(log):
    ag = DecisionAgent(DecisionConfig(max_auto_per_hour=2), log=log)
    ds = [diag(C.FLAKY_TEST, 0.9, incident_id=f"I{n}") for n in range(3)]
    incs = [incident("ci", f"proj-{n}", f"I{n}") for n in range(3)]
    out = ag.run(ds, incs, NOON)
    assert [d.route for d in out].count(Route.ACT) == 2
    tripped = [d for d in out if d.rule == "valve:circuit_breaker"]
    assert len(tripped) == 1 and tripped[0].status is ActionStatus.PENDING_APPROVAL


@pytest.mark.parametrize("cfg,rule", [(DecisionConfig(dry_run=True), "valve:dry_run"),
                                      (DecisionConfig(enabled=False), "valve:disabled")])
def test_kill_switch_and_dry_run_stop_every_automatic_action(cfg, rule):
    d = decide(DecisionAgent(cfg), diag(C.FLAKY_TEST, 0.99), incident("ci"))
    assert (d.route, d.status, d.rule) == (Route.APPROVAL, ActionStatus.PENDING_APPROVAL, rule)


def test_disabled_agent_does_not_silence_a_no_fault():
    assert decide(DecisionAgent(DecisionConfig(enabled=False)), nofault(0.95)).route is Route.ESCALATE


# ------------------------------------------------------------------ inbox: the hand-off to P5 and the human gate
def test_inbox_lifecycle_pending_approved_executed(tmp_path, log):
    inbox = ActionInbox(tmp_path / "inbox.json", log)
    d = decide(DecisionAgent(), diag(C.BAD_DEPLOY, 0.93, suspect="release-058"))
    inbox.add([d])
    assert [x.id for x in inbox.pending()] == [d.id] and inbox.ready() == []        # P5 sees nothing yet
    ok = inbox.resolve(d.id, "approve", "yassine", "looks right", now=NOON)
    assert ok.status is ActionStatus.APPROVED and ok.decided_by == "yassine" and ok.decided_at == NOON
    assert [x.id for x in inbox.ready()] == [d.id]
    done = inbox.report(d.id, "executed", "rolled back, error rate back to normal")
    assert done.status is ActionStatus.EXECUTED and inbox.ready() == []
    events = [e["event"] for e in log.events()]
    assert events == ["approved", "executed"] and log.grants()[0]["key"] == "release-058"


def test_rejection_executes_nothing_and_a_resolved_order_cannot_be_resolved_again(tmp_path, log):
    inbox = ActionInbox(tmp_path / "inbox.json", log)
    d = decide(DecisionAgent(), diag(C.BAD_DEPLOY, 0.93, suspect="release-058"))
    inbox.add([d])
    assert inbox.resolve(d.id, "reject", "yassine", "it was infra").status is ActionStatus.REJECTED
    assert inbox.ready() == [] and log.grants() == []
    with pytest.raises(InboxError):
        inbox.resolve(d.id, "approve")
    with pytest.raises(InboxError):
        inbox.report(d.id, "executed")                                 # P5 cannot execute a rejected order


def test_p5_cannot_execute_what_is_still_pending(tmp_path, log):
    inbox = ActionInbox(tmp_path / "inbox.json", log)
    d = decide(DecisionAgent(), diag(C.BAD_DEPLOY, 0.93, suspect="release-058"))
    inbox.add([d])
    with pytest.raises(InboxError):
        inbox.report(d.id, "executed")
    with pytest.raises(ValueError):
        inbox.report(d.id, "approved")


def test_action_decision_round_trips_through_json():
    d = decide(DecisionAgent(), diag(C.BAD_DEPLOY, 0.93, suspect="release-058"), incident("metrics"))
    back = ActionDecision.from_dict(__import__("json").loads(__import__("json").dumps(d.to_dict())))
    assert back == replace(d)                                          # every field survives, enums included


# ------------------------------------------------------------------ node
def test_node_contract_and_idempotence(tmp_path, log):
    inbox = ActionInbox(tmp_path / "inbox.json", log)
    node = make_decision_node(DecisionAgent(log=log), inbox)
    state = {"diagnoses": [diag(C.FLAKY_TEST, 0.9, incident_id="A"), diag(C.BAD_DEPLOY, 0.93, incident_id="B", suspect="r1"),
                           diag(C.UNKNOWN, 0.3, incident_id="C", needs_human=True)],
             "to_diagnosis": [incident("ci", "p", "A"), incident("metrics", "", "B"), incident("logs", "", "C")]}
    out = node(state)
    assert len(out["decisions"]) == 3 and len(out["log"]) == 3
    assert {d.incident_id for d in out["to_action"]} == {"A", "C"}             # act alone + escalate
    assert [d.incident_id for d in out["pending_approval"]] == ["B"]
    assert route_after_decision(out) == "action" and route_after_decision({"to_action": []}) == "end"
    assert len(inbox.all()) == 3
    out2 = node(state)                       # the same diagnoses again (a re-run of the pipeline): nothing doubled
    assert not out2["pending_approval"] and all(d.rule.startswith(("guard", "valve")) or d.route is Route.ESCALATE
                                                  for d in out2["decisions"])


def test_node_without_diagnoses_does_not_fail(tmp_path, log):
    out = make_decision_node(DecisionAgent(log=log), ActionInbox(tmp_path / "i.json", log))({})
    assert out["decisions"] == [] and out["log"] == [{"agent": "decision", "decided": 0}]


def test_every_decision_is_logged_with_its_reasons(tmp_path, log):
    DecisionAgent(log=log).run([diag(C.FLAKY_TEST, 0.9)], [incident("ci")], NOON)
    row = log.events()[0]
    assert row["event"] == "decided" and row["action"] == "rerun" and row["reasons"] and row["data_used"]["evidence"] == ["E1"]


# ------------------------------------------------------------------ evaluation scoring and the pipeline wiring
def test_evaluation_counts_real_mistakes():
    from agents.decision.evaluate import score
    row = lambda truth, action, how, category=None, suspect="", truth_suspect="": dict(
        truth=truth, action=action, how=how, category=category or truth, suspect=suspect, truth_suspect=truth_suspect)
    rows = [row(C.FLAKY_TEST, Action.RERUN, "alone"),                                   # right
            row(C.CODE_REGRESSION, Action.RERUN, "alone", category=C.FLAKY_TEST),        # a real bug "fixed" by a re-run: wrong
            row(C.BAD_DEPLOY, Action.ROLLBACK, "alone", suspect="r2", truth_suspect="r1"),   # wrong release: wrong
            row(C.INFRA, Action.NONE, "alone", category=C.UNKNOWN),                      # a fault silenced
            row(C.BAD_DEPLOY, Action.ROLLBACK, "human", suspect="r1", truth_suspect="r1")]
    s = score(rows)
    assert (s["n"], s["wrong_auto_actions"], s["silenced_faults"], s["right_auto_pct"]) == (5, 2, 1, 20.0)
    assert s["act_alone_pct"] == 60.0 and s["human_load_pct"] == 20.0


def test_pipeline_graph_runs_decision_and_hands_over_to_action(tmp_path, log):
    pytest.importorskip("langgraph")
    pytest.importorskip("pandas")
    pytest.importorskip("ollama")
    from pipeline import build_graph

    seen = {}
    ds = [diag(C.FLAKY_TEST, 0.9, incident_id="A"), diag(C.BAD_DEPLOY, 0.93, incident_id="B", suspect="r1")]
    action = lambda state: seen.update(got=[d.incident_id for d in state["to_action"]]) or {"log": [{"agent": "action"}]}
    graph = build_graph(lambda s: {"alerts": [], "log": []}, lambda s: {"log": []},
                        diagnosis_node=lambda s: {"diagnoses": ds, "log": []},
                        decision_node=make_decision_node(DecisionAgent(log=log), ActionInbox(tmp_path / "i.json", log)),
                        action_node=action)
    out = graph.invoke({"log": []})
    assert [d.incident_id for d in out["pending_approval"]] == ["B"] and seen["got"] == ["A"]
    assert out["log"][-1] == {"agent": "action"}
