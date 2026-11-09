"""Tests for the Action agent (P5). No Ollama, no data files, no sleep: hand-made orders, tmp_path, a virtual clock.

Each test pins one promise of the agent: the deck's success and failure branches, and the hard safety rules
(only approved orders, never twice, never a guessed target, never a deploy, simulation by default, honest reports).
"""
from datetime import datetime, timedelta, timezone

import pytest

from agents.action.agent import ActionAgent, SafetyError
from agents.action.backends import (BackendError, Backends, Comparison, FileNotifier, SimCI, SimDeploy,
                                    SimModel)
from agents.action.clock import RealClock, SimClock
from agents.action.config import ActionConfig
from agents.action.executors import job_refs_from_incidents
from agents.action.node import make_action_node
from agents.decision.inbox import ActionInbox
from agents.decision.log import DecisionLog
from common.schemas import (Action, ActionDecision, ActionStatus, Alert, Category, Decision, Incident, Route,
                            Severity)

T0 = datetime(2026, 1, 20, 10, 0, tzinfo=timezone.utc)
C = Category
JOB = {"INC-1": {"project": "shop-api", "build_id": "48211", "commit": "a1b2c3d"}}


# ------------------------------------------------------------------ helpers
def order(action=Action.RERUN, status=ActionStatus.APPROVED, incident_id="INC-1", target="", **kw) -> ActionDecision:
    cat = {Action.RERUN: C.FLAKY_TEST, Action.ROLLBACK: C.BAD_DEPLOY, Action.RETRAIN: C.DATA_DRIFT,
           Action.ESCALATE: C.UNKNOWN, Action.NONE: C.UNKNOWN}[action]
    key = {Action.RERUN: "ci:shop-api", Action.ROLLBACK: target}.get(action, f"x:{incident_id}")
    constraints = {"max_retries": 2, "verify_after_min": 15, "revert_if_worse": action in
                   (Action.RERUN, Action.ROLLBACK, Action.RETRAIN), "human_before_deploy": action is Action.RETRAIN}
    if action is Action.RERUN:
        constraints["reruns"] = 1
    constraints.update(kw.pop("constraints", {}))
    route = {Action.ESCALATE: Route.ESCALATE, Action.NONE: Route.NONE}.get(action, Route.ACT)
    return ActionDecision(incident_id=incident_id, diagnosis_id="diag-1", category=cat, action=action, route=route,
                          status=status, confidence=0.9, risk="low", target=target, rule="test",
                          reasons=["because the test says so"], summary=f"{cat.value} on {incident_id}",
                          snapshot={"action_key": kw.pop("action_key", key)}, constraints=constraints,
                          created=T0, **kw)


@pytest.fixture
def inbox(tmp_path):
    return ActionInbox(tmp_path / "inbox.json", DecisionLog(tmp_path / "log.jsonl", tmp_path / "history.jsonl"))


def make_agent(tmp_path, inbox, backends=None, cfg=None, clock=None):
    return ActionAgent(cfg or ActionConfig(), backends or Backends(), inbox, clock or SimClock(T0 + timedelta(minutes=1)),
                       claims_dir=tmp_path / "claims", notifications=tmp_path / "notes.jsonl",
                       deploy_requests=tmp_path / "deploy_requests.jsonl")


def run_one(tmp_path, inbox, o, backends=None, job_refs=JOB, **kw):
    inbox.add([o])
    agent = make_agent(tmp_path, inbox, backends, **kw)
    return agent.run([o], job_refs)[0], agent


def notes(tmp_path):
    return FileNotifier(tmp_path / "notes.jsonl").messages()


def deploy_world(**kw):
    return Backends(deploy=SimDeploy(["release-v2.2", "release-v2.3"], **kw))


# ------------------------------------------------------------------ scenario 1: flaky test (rerun)
def test_rerun_passes_is_executed_and_nobody_is_paged(tmp_path, inbox):
    ci = SimCI({"48211": ["passed"]})
    run, _ = run_one(tmp_path, inbox, order(), Backends(ci=ci))
    assert run.outcome == "executed" and run.attempts == 1 and run.notified == []
    assert inbox.all()[0].status is ActionStatus.EXECUTED
    assert list(ci.runs.values())[0]["build_id"] == "48211"            # the build of the incident, not another one
    assert notes(tmp_path) == []


def test_rerun_red_again_fails_is_not_rerun_and_the_author_is_told(tmp_path, inbox):
    ci = SimCI({"48211": ["failed", "passed"]})
    run, _ = run_one(tmp_path, inbox, order(), Backends(ci=ci))
    assert run.outcome == "failed" and run.attempts == 1                 # rerun budget: one re-run, never two
    assert len(ci.runs) == 1
    assert inbox.all()[0].status is ActionStatus.FAILED
    assert [m["to"] for m in notes(tmp_path)] == ["author"] and notes(tmp_path)[0]["commit"] == "a1b2c3d"


def test_rerun_without_a_build_id_is_not_guessed(tmp_path, inbox):
    ci = SimCI()
    run, _ = run_one(tmp_path, inbox, order(), Backends(ci=ci), job_refs={})
    assert run.outcome == "failed" and "no build id" in run.detail and run.attempts == 0
    assert ci.runs == {}                                                  # nothing was re-run
    assert [m["to"] for m in notes(tmp_path)] == ["author"]                # ... and a human is told


def test_job_refs_come_from_the_ci_alerts_of_the_incident():
    def alert(build, minutes):
        return Alert(detector="flaky_model", source="ci", feature="build_failed", score=2, threshold=1,
                     timestamp=T0 + timedelta(minutes=minutes), value=0.9, dataset="travis",
                     details={"project": "p", "build_id": build, "commit": f"c{build}"})
    inc = Incident(id="INC-9", source="ci", features=["build_failed"], alert_ids=[], start=T0, end=T0,
                   duration_hours=1, peak_ratio=2, max_voters=1, severity=Severity.HIGH, decision=Decision.INCIDENT,
                   summary="", alerts=[alert(1, 0), alert(2, 5)])
    assert job_refs_from_incidents([inc]) == {"INC-9": {"project": "p", "build_id": "2", "commit": "c2",
                                                        "alert_id": inc.alerts[1].id}}


# ------------------------------------------------------------------ scenario 2: bad deploy (rollback)
def test_rollback_back_in_band_is_executed(tmp_path, inbox):
    world = deploy_world(effects={"release-v2.3": (0.06, 40.0)})
    run, agent = run_one(tmp_path, inbox, order(Action.ROLLBACK, target="release-v2.3"), world)
    assert run.outcome == "executed"
    assert world.deploy.live == ["release-v2.2"]
    assert agent.clock.now() - (T0 + timedelta(minutes=1)) == timedelta(minutes=15)   # waited verify_after_min, virtually
    assert run.data["after"]["error_rate"] <= 0.02


def test_rollback_still_bad_fails_and_on_call_takes_over(tmp_path, inbox):
    world = deploy_world(effects={"release-v2.3": (0.005, 5.0)}, ambient=(0.05, 50.0))
    run, _ = run_one(tmp_path, inbox, order(Action.ROLLBACK, target="release-v2.3"), world)
    assert run.outcome == "failed" and run.attempts == 1                  # rolling back twice changes nothing
    assert world.deploy.live == ["release-v2.2"]                         # not undone: the previous version is stable
    assert [m["to"] for m in notes(tmp_path)] == ["on-call"]


def test_rollback_worse_is_undone_and_reported_reverted(tmp_path, inbox):
    world = deploy_world(effects={"release-v2.3": (-0.03, -20.0)}, ambient=(0.06, 40.0))
    run, _ = run_one(tmp_path, inbox, order(Action.ROLLBACK, target="release-v2.3"), world)
    assert run.outcome == "reverted"
    assert "release-v2.3" in world.deploy.live                           # put back
    assert inbox.all()[0].status is ActionStatus.REVERTED


def test_worse_without_revert_if_worse_is_failed_not_undone(tmp_path, inbox):
    world = deploy_world(effects={"release-v2.3": (-0.03, -20.0)}, ambient=(0.06, 40.0))
    o = order(Action.ROLLBACK, target="release-v2.3", constraints={"revert_if_worse": False})
    run, _ = run_one(tmp_path, inbox, o, world)
    assert run.outcome == "failed" and "release-v2.3" not in world.deploy.live


def test_rollback_without_target_is_not_guessed(tmp_path, inbox):
    world = deploy_world(effects={"release-v2.3": (0.06, 40.0)})
    run, _ = run_one(tmp_path, inbox, order(Action.ROLLBACK, target=""), world)
    assert run.outcome == "failed" and "without a target" in run.detail
    assert world.deploy.live == ["release-v2.2", "release-v2.3"]          # nothing rolled back
    assert [m["to"] for m in notes(tmp_path)] == ["on-call"]


def test_rollback_of_a_release_that_is_not_live_is_refused(tmp_path, inbox):
    world = deploy_world()
    run, _ = run_one(tmp_path, inbox, order(Action.ROLLBACK, target="release-v9.9"), world)
    assert run.outcome == "failed" and "not live" in run.detail
    assert world.deploy.live == ["release-v2.2", "release-v2.3"]


# ------------------------------------------------------------------ retries (slide 11: max 2)
def test_api_down_is_retried_at_most_max_retries_then_failed(tmp_path, inbox):
    ci = SimCI(api_errors=10)
    run, _ = run_one(tmp_path, inbox, order(), Backends(ci=ci))
    assert run.outcome == "failed" and run.attempts == 3                  # 1 + max_retries (2)
    assert ci.runs == {}


def test_api_down_once_then_ok_is_executed_on_the_second_attempt(tmp_path, inbox):
    run, _ = run_one(tmp_path, inbox, order(), Backends(ci=SimCI({"48211": ["passed"]}, api_errors=1)))
    assert run.outcome == "executed" and run.attempts == 2


def test_order_cannot_ask_for_more_retries_than_the_cap(tmp_path, inbox):
    run, _ = run_one(tmp_path, inbox, order(constraints={"max_retries": 10}), Backends(ci=SimCI(api_errors=10)))
    assert run.attempts == 3


class LossyNotifier(FileNotifier):
    """Sends, but the first `lose` messages never arrive (verify says not delivered)."""
    def __init__(self, path, lose):
        super().__init__(path)
        self.lose, self.sent = lose, 0

    def delivered(self, message_id):
        self.sent += 1
        return self.sent > self.lose


def test_verify_fails_is_retried_up_to_max_then_failed(tmp_path, inbox):
    lossy = LossyNotifier(tmp_path / "notes.jsonl", lose=99)
    run, _ = run_one(tmp_path, inbox, order(Action.ESCALATE), Backends(notifier=lossy))
    assert run.outcome == "failed" and run.attempts == 3 and lossy.sent == 3
    assert inbox.all()[0].status is ActionStatus.FAILED


def test_verify_fails_once_then_delivered_is_executed(tmp_path, inbox):
    run, _ = run_one(tmp_path, inbox, order(Action.ESCALATE), Backends(notifier=LossyNotifier(tmp_path / "n.jsonl", 1)))
    assert run.outcome == "executed" and run.attempts == 2


def test_missing_constraint_is_not_guessed(tmp_path, inbox):
    o = order()
    del o.constraints["verify_after_min"]
    run, _ = run_one(tmp_path, inbox, o, Backends(ci=SimCI()))
    assert run.outcome == "failed" and "verify_after_min" in run.detail and run.attempts == 0


# ------------------------------------------------------------------ scenario 3: data drift (retrain)
def test_retrain_better_is_executed_but_never_deployed(tmp_path, inbox):
    model = SimModel(2.10, 1.70)
    run, _ = run_one(tmp_path, inbox, order(Action.RETRAIN), Backends(model=model))
    assert run.outcome == "executed"
    assert model.live == "champion"                                      # the live model did not change
    assert not hasattr(model, "deploy")                                  # and there is no way to deploy from P5
    req = (tmp_path / "deploy_requests.jsonl").read_text(encoding="utf-8")
    assert '"awaiting_human"' in req
    assert [m["to"] for m in notes(tmp_path)] == ["ml-engineer"]


def test_retrain_worse_keeps_the_champion_and_fails(tmp_path, inbox):
    model = SimModel(2.10, 2.25)
    run, _ = run_one(tmp_path, inbox, order(Action.RETRAIN), Backends(model=model))
    assert run.outcome == "failed" and run.attempts == 1
    assert model.challengers == {}                                       # the losing challenger is thrown away
    assert not (tmp_path / "deploy_requests.jsonl").exists()
    assert [m["to"] for m in notes(tmp_path)] == ["ml-engineer"]


def test_retrain_order_that_would_allow_a_deploy_is_refused(tmp_path, inbox):
    model = SimModel()
    run, _ = run_one(tmp_path, inbox, order(Action.RETRAIN, constraints={"human_before_deploy": False}), Backends(model=model))
    assert run.outcome == "failed" and "never deploys" in run.detail and model.challengers == {}


def test_sklearn_retrain_on_a_drifted_table(tmp_path):
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from agents.action.retrain_sklearn import SklearnNYCModel
    rng = np.random.default_rng(0)
    ts = pd.date_range("2025-10-01", periods=24 * 90, freq="h", tz="UTC")
    dist = rng.uniform(1, 6, len(ts))
    price = np.where(ts >= pd.Timestamp("2025-12-01", tz="UTC"), 4.5, 3.0)    # a price change: the fare drift
    df = pd.DataFrame({"timestamp": ts, "trip_count": rng.integers(20, 120, len(ts)), "avg_distance": dist,
                       "avg_fare": 2.5 + price * dist + rng.normal(0, 0.3, len(ts))})
    cmp = SklearnNYCModel(df, out_dir=tmp_path).retrain_and_compare("o1", T0, 28, 7)
    assert isinstance(cmp, Comparison) and cmp.challenger < cmp.champion and cmp.gain > 0.5
    assert (tmp_path / "challenger-o1.pkl").exists()


def test_sklearn_retrain_without_data_is_refused_not_faked(tmp_path, inbox):
    from agents.action.retrain_sklearn import SklearnNYCModel
    model = SklearnNYCModel(path=tmp_path / "missing.parquet", out_dir=tmp_path)
    run, _ = run_one(tmp_path, inbox, order(Action.RETRAIN), Backends(model=model))
    assert run.outcome == "failed" and "not found" in run.detail and run.attempts == 1


# ------------------------------------------------------------------ scenario 4: escalate
def test_escalate_notifies_and_changes_nothing(tmp_path, inbox):
    world = Backends(ci=SimCI(), deploy=SimDeploy(["release-v2.3"]), model=SimModel())
    run, _ = run_one(tmp_path, inbox, order(Action.ESCALATE, constraints={"notify": "author"}), world)
    assert run.outcome == "executed"
    assert world.ci.runs == {} and world.deploy.live == ["release-v2.3"] and world.model.challengers == {}
    m = notes(tmp_path)
    assert [x["to"] for x in m] == ["author"] and "because the test says so" in m[0]["body"]


def test_escalate_without_notify_goes_to_on_call_and_says_why(tmp_path, inbox):
    run, _ = run_one(tmp_path, inbox, order(Action.ESCALATE), Backends())
    assert notes(tmp_path)[0]["to"] == "on-call" and "names nobody" in run.steps[1]


# ------------------------------------------------------------------ hard safety rules
@pytest.mark.parametrize("status", [ActionStatus.PENDING_APPROVAL, ActionStatus.REJECTED, ActionStatus.CLOSED,
                                    ActionStatus.EXECUTED, ActionStatus.FAILED, ActionStatus.REVERTED])
def test_only_approved_orders_are_executed(tmp_path, inbox, status):
    ci = SimCI()
    run, _ = run_one(tmp_path, inbox, order(status=status), Backends(ci=ci))
    assert run.outcome == "skipped" and ci.runs == {}
    assert inbox.all()[0].status is status                               # untouched


def test_a_stale_copy_is_not_trusted_the_inbox_is(tmp_path, inbox):
    """P5 may be handed a copy (state["to_action"]) that says approved while the inbox says otherwise: the inbox wins."""
    ci = SimCI()
    real = order(status=ActionStatus.REJECTED)
    inbox.add([real])
    copy = order()                                                         # approved in the copy only
    copy.id = real.id
    run = make_agent(tmp_path, inbox, Backends(ci=ci)).run([copy], JOB)[0]
    assert run.outcome == "skipped" and "rejected" in run.detail and ci.runs == {}


def test_action_none_is_never_executed(tmp_path, inbox):
    run, _ = run_one(tmp_path, inbox, order(Action.NONE, status=ActionStatus.APPROVED), Backends())
    assert run.outcome == "skipped"


def test_an_order_not_in_the_inbox_is_not_run(tmp_path, inbox):
    ci = SimCI()
    run = make_agent(tmp_path, inbox, Backends(ci=ci)).run([order()], JOB)[0]
    assert run.outcome == "skipped" and ci.runs == {}


def test_no_double_execution(tmp_path, inbox):
    ci = SimCI()
    o = order()
    inbox.add([o])
    agent = make_agent(tmp_path, inbox, Backends(ci=ci))
    first = agent.run([o, o], JOB)
    assert [r.outcome for r in first] == ["executed", "skipped"] and len(ci.runs) == 1


def test_claim_blocks_a_second_run_even_if_the_first_never_reported(tmp_path, inbox):
    """A crash between execute and report leaves the order 'approved': it must still never run twice."""
    ci = SimCI()
    o = order()
    inbox.add([o])
    make_agent(tmp_path, inbox, Backends(ci=ci)).claims.take(o.id, T0)   # = a first process that died
    run = make_agent(tmp_path, inbox, Backends(ci=ci)).run([o], JOB)[0]
    assert run.outcome == "skipped" and "already started" in run.detail and ci.runs == {}
    assert inbox.all()[0].status is ActionStatus.APPROVED                 # left for a human to check


def test_reports_go_through_the_inbox_and_the_decision_log(tmp_path, inbox):
    run, _ = run_one(tmp_path, inbox, order(), Backends(ci=SimCI()))
    d = inbox.all()[0]
    assert d.status is ActionStatus.EXECUTED and "passed" in d.result and "[simulation]" in d.result
    events = [e["event"] for e in inbox.log.events()]
    assert events[0] == "action_started" and "action_execute" in events and "action_verify" in events
    assert events[-1] == "executed"
    assert all(e.get("agent") == "action" for e in inbox.log.events() if e["event"].startswith("action_"))


def test_an_unexpected_error_is_reported_failed_not_hidden(tmp_path, inbox):
    class Broken(SimCI):
        def result(self, run_id, now):
            raise ZeroDivisionError("bug")
    run, _ = run_one(tmp_path, inbox, order(), Backends(ci=Broken()))
    assert run.outcome == "failed" and "ZeroDivisionError" in run.detail
    assert inbox.all()[0].status is ActionStatus.FAILED


# ------------------------------------------------------------------ the simulation switch
def test_simulation_is_the_default():
    cfg = ActionConfig()
    assert cfg.simulate is True
    assert isinstance(ActionAgent(cfg, inbox=ActionInbox("unused.json")).clock, SimClock)


def test_real_mode_needs_real_backends_explicitly(tmp_path):
    with pytest.raises(SafetyError, match="no real backends"):
        ActionAgent(ActionConfig(simulate=False), inbox=ActionInbox(tmp_path / "i.json"))


def test_simulation_refuses_a_backend_that_touches_production(tmp_path):
    class RealCI(SimCI):
        touches_production = True
    with pytest.raises(SafetyError, match="touch production"):
        ActionAgent(ActionConfig(), Backends(ci=RealCI()), ActionInbox(tmp_path / "i.json"))
    agent = ActionAgent(ActionConfig(simulate=False), Backends(ci=RealCI()), ActionInbox(tmp_path / "i.json"),
                        claims_dir=tmp_path / "c", notifications=tmp_path / "n.jsonl")
    assert isinstance(agent.clock, RealClock)                             # real mode, real time


def test_unknown_backend_is_assumed_to_touch_production(tmp_path):
    class Mystery:
        def rerun(self, *a): ...
    with pytest.raises(SafetyError):
        ActionAgent(ActionConfig(), Backends(ci=Mystery()), ActionInbox(tmp_path / "i.json"))


# ------------------------------------------------------------------ KPIs
def test_kpis_mttr_and_auto_resolution(tmp_path, inbox):
    from agents.action import kpis
    agent = make_agent(tmp_path, inbox, Backends(ci=SimCI({"48211": ["passed"]})))
    o_ok, o_esc = order(), order(Action.ESCALATE, incident_id="INC-2")
    inbox.add([o_ok, o_esc])
    agent.run([o_ok, o_esc], JOB)
    k = kpis.compute(inbox.all(), inbox.log.events())
    assert k["orders_run"] == 2 and k["auto_resolved_pct"] == 50.0 and k["fix_rate_pct"] == 100.0
    assert k["mttr_min_median"] == 16.0                                   # created 10:00, started 10:01, +15 min verify
    assert k["simulated"] is True and "MTTR" in kpis.report(k)


# ------------------------------------------------------------------ the graph
def test_node_reads_to_action_and_appends_to_the_log(tmp_path, inbox):
    o = order()
    inbox.add([o])
    node = make_action_node(make_agent(tmp_path, inbox, Backends(ci=SimCI())))
    out = node({"to_action": [o], "to_diagnosis": []})
    assert out["actions_done"][0].outcome == "failed"                     # no incident given: no build id, not guessed
    assert out["log"][0]["agent"] == "action"


def test_node_standalone_reads_the_inbox(tmp_path, inbox):
    o = order(Action.ESCALATE)
    inbox.add([o, order(Action.ROLLBACK, status=ActionStatus.PENDING_APPROVAL, target="r1", incident_id="INC-3")])
    out = make_action_node(make_agent(tmp_path, inbox))({})
    assert [r.order_id for r in out["actions_done"]] == [o.id]


def test_pipeline_graph_ends_with_the_action_node(tmp_path):
    """monitor -> triage -> diagnosis -> decision -> action, through LangGraph, every file in tmp_path."""
    pytest.importorskip("langgraph")
    from pipeline import build_graph, make_layer2
    from agents.triage.memory import Memory
    from agents.decision.agent import DecisionAgent
    from agents.decision.node import make_decision_node
    from agents.diagnosis.node import make_diagnosis_node

    alert = {"id": "alert-ci-1", "timestamp": T0.isoformat(), "source": "ci", "signal": "build_failed", "score": 3.0,
             "kind": "anomaly", "dataset": "travis",                       # = P1's monitor_node format for a CI alert
             "details": {"flaky_probability": 0.97, "project": "shop-api", "build_id": 48211, "commit": "a1b2c3d",
                         "tests_failed": 1.0, "evidence": {"project_flaky_rate": 0.8, "recent_fail_rate": 0.1,
                                                           "previous_build_failed": 0.0}}}
    monitor = lambda state: {"alerts": [alert], "log": [{"node": "monitor"}]}
    dlog = DecisionLog(tmp_path / "log.jsonl", tmp_path / "history.jsonl")
    inbox = ActionInbox(tmp_path / "inbox.json", dlog)
    decision = make_decision_node(DecisionAgent(log=dlog), inbox)
    action = make_action_node(make_agent(tmp_path, inbox, Backends(ci=SimCI())))
    graph = build_graph(monitor, make_layer2(Memory(), use_llm=False), make_diagnosis_node(), decision, action)
    out = graph.invoke({"log": []})
    nodes = [x.get("node", x.get("agent")) for x in out["log"]]
    assert nodes[-2:] == ["decision", "action"]
    assert [d.action for d in out["to_action"]] == [Action.RERUN]          # flaky -> rerun, approved by the policy
    run = out["actions_done"][0]
    assert run.outcome == "executed" and "build 48211" in run.detail        # the build id came from P1's alert
    assert inbox.get(run.order_id).status is ActionStatus.EXECUTED


def test_pipeline_graph_without_action_node_is_unchanged(tmp_path):
    """No action_node: the graph still ends after Decision (the other modes of pipeline.py)."""
    pytest.importorskip("langgraph")
    from pipeline import build_graph, make_layer2
    from agents.triage.memory import Memory
    from agents.decision.agent import DecisionAgent
    from agents.decision.node import make_decision_node
    from agents.diagnosis.node import make_diagnosis_node
    dlog = DecisionLog(tmp_path / "log.jsonl", tmp_path / "history.jsonl")
    graph = build_graph(lambda s: {"alerts": [], "log": [{"node": "monitor"}]}, make_layer2(Memory(), use_llm=False),
                        make_diagnosis_node(), make_decision_node(DecisionAgent(log=dlog), ActionInbox(tmp_path / "i.json", dlog)))
    out = graph.invoke({"log": []})
    assert "actions_done" not in out


def test_standalone_run_gives_each_approved_rollback_its_own_simulated_service(tmp_path, inbox):
    """Six approved rollbacks are six separate incidents: rolling back one must not look like a failure only because
    the other five releases were still live in one shared simulated service (seen on a real inbox)."""
    from agents.action.run import run_ready
    inbox.add([order(Action.ROLLBACK, incident_id=f"INC-{i}", target=f"release-{i}") for i in range(3)])
    runs = run_ready(inbox, claims_dir=tmp_path / "claims", notifications=tmp_path / "notes.jsonl",
                     deploy_requests=tmp_path / "deploy_requests.jsonl")
    assert [r.outcome for r in runs] == ["executed"] * 3
    assert {d.status for d in inbox.all()} == {ActionStatus.EXECUTED}
