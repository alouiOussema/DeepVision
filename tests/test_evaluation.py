"""Evaluation tests: hand-made alerts, an oracle Diagnosis and tmp_path. No Ollama, no data files."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agents.action.agent import ActionAgent
from agents.action.clock import SimClock
from agents.action.config import ActionConfig
from agents.decision.agent import DecisionAgent
from agents.decision.inbox import ActionInbox
from agents.decision.log import DecisionLog
from agents.diagnosis.context import Change, ContextStore
from agents.evaluation.config import EvalConfig
from agents.evaluation.injector import Episode, Injection
from agents.evaluation.report import incident_report
from agents.evaluation.score import kpis, records
from agents.evaluation.trace import Tracer, timeline
from agents.evaluation.world import backends_for
from common.schemas import Alert, Category, Decision, Diagnosis, Incident, Severity

T0 = datetime(2026, 1, 20, 10, 0, tzinfo=timezone.utc)
C = Category


def alert(aid: str, minutes: int = 5, feature: str = "error_rate", score: float = 5.0) -> Alert:
    start = T0 + timedelta(minutes=minutes)
    return Alert(detector="vote(rolling)", source="metrics", feature=feature, score=score, threshold=1.0,
                 timestamp=start, value=0.06, dataset="deploy", id=aid,
                 details={"voters": ["rolling"], "start": str(start), "end": str(start + timedelta(minutes=30)),
                          "n_points": 31})


def incident(iid: str, alerts: list[Alert], decision=Decision.INCIDENT) -> Incident:
    return Incident(id=iid, source="metrics", features=[a.feature for a in alerts], alert_ids=[a.id for a in alerts],
                    start=min(a.start for a in alerts), end=max(a.end for a in alerts), duration_hours=0.5,
                    peak_ratio=5.0, max_voters=1, severity=Severity.HIGH, decision=decision, summary="error_rate up",
                    reasons=["grouped by time", "incident / high: strong anomaly"], alerts=alerts, dataset="deploy")


def bad_deploy_episode(alerts=None) -> Episode:
    """release-001 is the faulty one; release-000 went out the day before and is harmless."""
    alerts = alerts or [alert("a1"), alert("a2", 6, "latency_ms")]
    ctx = ContextStore([Change(T0 - timedelta(days=1), "deploy", "release-000"), Change(T0, "deploy", "release-001")])
    inj = Injection("INJ-1", "deploy", C.BAD_DEPLOY, "error_regression", "release-001", T0, T0 + timedelta(hours=2),
                    [a.id for a in alerts], "simulation")
    return Episode("unit", "deploy", 0, alerts, ctx, [inj])


def oracle_diagnosis(ep: Episode, suspect: str | None = None):
    """A Diagnosis node that knows the truth (optionally blaming the wrong release): isolates Decision + Action."""
    def node(state):
        out = []
        for inc in state.get("to_diagnosis", []):
            inj = ep.injection_for(inc.alert_ids)
            out.append(Diagnosis(incident_id=inc.id, category=inj.truth, explanation="oracle", evidence=[],
                                 confidence=0.95, sub_cause=inj.truth_sub, suggested_action="rollback",
                                 suspect=suspect if suspect is not None else inj.truth_suspect, method="oracle"))
        return {"diagnoses": out, "log": [{"agent": "diagnosis", "oracle": len(out)}]}
    return node


def test_incident_takes_the_truth_of_most_of_its_alerts():
    ep = Episode("x", "deploy", 0, [], ContextStore(), [
        Injection("F", "deploy", C.BAD_DEPLOY, alert_ids=["a1", "a2"]),
        Injection("N", "deploy", C.UNKNOWN, "no_fault", alert_ids=["a3", "a4"])])
    assert ep.injection_for(["a1", "a2", "a3"]).id == "F"
    assert ep.injection_for(["a1", "a3"]).id == "F"
    assert ep.injection_for(["zz"]) is None
    assert not ep.injections[1].is_fault and ep.injections[0].is_fault


def test_injection_round_trips_through_json(tmp_path):
    ep = bad_deploy_episode()
    path = ep.save(tmp_path)
    import json
    back = Injection.from_dict(json.loads(path.read_text().splitlines()[0]))
    assert back == ep.injections[0]


def test_deploy_episode_maps_every_alert_to_its_fault():
    pytest.importorskip("sklearn")
    from agents.evaluation.injector import deploy_episode
    ep = deploy_episode(4)
    of = ep.injection_of()
    assert set(of) == {a.id for a in ep.alerts}
    faults = [i for i in ep.injections if i.is_fault]
    assert faults and all(i.fault_start for i in faults)
    for i in faults:
        for a in (x for x in ep.alerts if x.id in i.alert_ids):
            assert i.fault_start - timedelta(minutes=5) <= a.start <= i.fault_end
    assert {c.truth_sub for c in ep.injections if not c.is_fault} <= {"no_fault", "recovery"}


def _order(tmp_path, ep, inc, suspect):
    log = DecisionLog(tmp_path / "log.jsonl", tmp_path / "hist.jsonl")
    inbox = ActionInbox(tmp_path / "inbox.json", log)
    diag = oracle_diagnosis(ep, suspect)({"to_diagnosis": [inc]})["diagnoses"][0]
    o = DecisionAgent(log=log).decide(diag, inc, inc.start)
    inbox.add([o])
    o = inbox.resolve(o.id, "approve", now=inc.start)
    return inbox, o


@pytest.mark.parametrize("suspect,outcome", [("release-001", "executed"), ("release-000", "failed")])
def test_world_only_lets_the_right_rollback_work(tmp_path, suspect, outcome):
    ep = bad_deploy_episode()
    inc = incident("INC-1", ep.alerts)
    inbox, o = _order(tmp_path, ep, inc, suspect)
    agent = ActionAgent(ActionConfig(), backends_for(ep.injections[0], inc, ep.context), inbox, SimClock(inc.start),
                        claims_dir=tmp_path / "claims", notifications=tmp_path / "n.jsonl",
                        deploy_requests=tmp_path / "d.jsonl")
    assert agent.run([o])[0].outcome == outcome


def test_world_ci_and_model_follow_the_truth():
    flaky = backends_for(Injection("F", "ci", C.FLAKY_TEST), None, ContextStore())
    real = backends_for(Injection("R", "ci", C.CODE_REGRESSION), None, ContextStore())
    assert flaky.ci.default == "passed" and real.ci.default == "failed"
    drift = backends_for(Injection("D", "model", C.DATA_DRIFT), None, ContextStore())
    calm = backends_for(None, None, ContextStore())
    assert drift.model.challenger_mae < drift.model.champion_mae < calm.model.challenger_mae
    infra = backends_for(Injection("I", "deploy", C.INFRA), None, ContextStore())
    assert infra.deploy.ambient == EvalConfig().infra_ambient and not infra.deploy.effects


def test_tracer_tells_one_story_per_incident_in_order():
    ep = bad_deploy_episode()
    inc = incident("INC-1", ep.alerts)
    diag = oracle_diagnosis(ep)({"to_diagnosis": [inc]})["diagnoses"][0]
    tr = Tracer("unit")
    tr.consume([{"monitor": {"alerts": ep.alerts}}, {"layer1_rules": {"incidents": [inc], "final": [inc]}},
                {"to_diagnosis": {"to_diagnosis": [inc]}}, {"diagnosis": {"diagnoses": [diag]}}])
    story = tr.timeline("INC-1")
    assert [e.agent for e in story] == ["monitor", "triage", "diagnosis"]
    assert story[0].at == T0 + timedelta(minutes=5) and "2 alert(s)" in story[0].action
    assert "bad_deploy" in story[2].action and "release-001" in story[2].action
    assert story[0].incident_id == "unit:INC-1" and timeline(tr.events, "unit:INC-1") == story


def _run(tmp_path, suspect=None):
    pytest.importorskip("langgraph")
    from agents.evaluation.runner import run_episode
    ep = bad_deploy_episode()
    tr, state, files = run_episode(ep, tmp_path / "run", EvalConfig(), diagnosis_node=oracle_diagnosis(ep, suspect))
    return ep, tr, state, files


def test_end_to_end_right_rollback_is_fixed_and_timed(tmp_path):
    from agents.decision import paths as decision_paths
    real = decision_paths.ACTION_INBOX
    before = real.stat().st_mtime if real.exists() else None
    ep, tr, state, files = _run(tmp_path)
    recs = records(ep, state, files.inbox.all(), key=tr.key)
    assert len(recs) == 1
    r = recs[0]
    assert r.sent and r.diagnosis_right and r.action == "rollback" and r.route == "approval"
    assert r.fixed and not r.fixed_alone and not r.wrong_fix
    agents = [e.agent for e in tr.timeline(r.incident_id)]
    assert agents == ["monitor", "triage", "diagnosis", "decision", "human", "action"]
    k = kpis(recs, ep.injections)
    assert k["mttd_min"] == 5.0
    assert k["mttr_min"] == 25.0
    assert k["resolved_pct"] == 100.0 and k["auto_resolved_pct"] == 0.0 and k["false_action_pct"] == 0.0
    assert k["noise_reduction_pct"] == 50.0
    assert (real.stat().st_mtime if real.exists() else None) == before


def test_end_to_end_wrong_release_is_a_false_action(tmp_path):
    ep, tr, state, files = _run(tmp_path, suspect="release-000")
    r = records(ep, state, files.inbox.all(), key=tr.key)[0]
    assert r.diagnosis_right is False and r.wrong_fix and not r.fixed and r.outcome == "failed"
    k = kpis([r], ep.injections)
    assert k["false_action_pct"] == 100.0 and k["wrong_auto_actions"] == 0
    assert k["resolved_pct"] == 0.0 and k["mttr_min"] is None


def test_unscored_injections_run_but_are_not_graded(tmp_path):
    ep, tr, state, files = _run(tmp_path)
    ep.injections[0].scored = False
    recs = records(ep, state, files.inbox.all(), key=tr.key)
    k = kpis(recs, ep.injections)
    assert k["incidents"] == 0 and k["faults_injected"] == 0 and k["diagnosis_accuracy_pct"] is None


def test_missed_and_silenced_faults_are_counted():
    ep = Episode("x", "deploy", 0, [], ContextStore(), [
        Injection("SEEN", "deploy", C.BAD_DEPLOY, alert_ids=["a1"], fault_start=T0),
        Injection("NEVER", "deploy", C.BAD_DEPLOY, fault_start=T0)])
    inc = incident("INC-1", [alert("a1")], decision=Decision.WATCH)
    k = kpis(records(ep, {"final": [inc]}, []), ep.injections)
    assert k["missed_by_monitor"] == 1 and k["missed_by_triage"] == 1 and k["sent_to_diagnosis"] == 0


def test_report_shows_cause_action_verification_and_truth(tmp_path):
    ep, tr, state, files = _run(tmp_path)
    r = records(ep, state, files.inbox.all(), key=tr.key)[0]
    md = incident_report(r, tr.timeline(r.incident_id))
    for text in ("fixed and verified", "bad_deploy", "rollback release-001", "back in the normal band",
                 "Truth (injected)", "INJ-1", "| human | approved"):
        assert text in md


def test_dashboard_page_loads(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from agents.evaluation import paths
    from agents.evaluation.run import evaluate
    import agents.evaluation.run as run_mod

    ep = bad_deploy_episode()
    monkeypatch.setattr(run_mod, "episodes", lambda cfg: [ep])
    evaluate(EvalConfig(), tmp_path / "latest", injections_dir=None, diagnosis_node=oracle_diagnosis(ep),
             verbose=False)
    monkeypatch.setattr(paths, "RUN_DIR", tmp_path / "latest")
    page = Path(__file__).resolve().parents[1] / "ui" / "evaluation_app.py"
    at = AppTest.from_file(str(page), default_timeout=30).run()
    assert not at.exception
    assert any(m.label == "Diagnosis accuracy" and m.value == "100.0 %" for m in at.metric)


DEMO_PAGES = ["p0_project", "p0_data", "p1_monitor", "p2_triage", "p3_diagnosis", "p4_decision", "p5_human", "p6_action",
              "p7_follow", "p8_results"]


@pytest.mark.parametrize("page", DEMO_PAGES)
def test_every_demo_page_loads(tmp_path, monkeypatch, page):
    """The final demo app (ui/demo_app.py): every page renders on a small run, without errors."""
    pytest.importorskip("streamlit")
    import sys
    from streamlit.testing.v1 import AppTest
    import agents.evaluation.run as run_mod
    from agents.evaluation.run import evaluate

    ui = Path(__file__).resolve().parents[1] / "ui"
    sys.path.insert(0, str(ui))
    import demo_pages.common as common

    ep = bad_deploy_episode()
    monkeypatch.setattr(run_mod, "episodes", lambda cfg: [ep])
    cfg = EvalConfig()
    cfg.families = ("deploy",)
    evaluate(cfg, tmp_path / "run", injections_dir=None, diagnosis_node=oracle_diagnosis(ep), verbose=False)
    monkeypatch.setitem(common.RUNS, "full", ("Full run", "test", tmp_path / "run"))
    at = AppTest.from_file(str(ui / "demo_app.py"), default_timeout=60).run()
    at.switch_page(f"demo_pages/{page}.py").run()
    assert not at.exception, [e.message for e in at.exception]
