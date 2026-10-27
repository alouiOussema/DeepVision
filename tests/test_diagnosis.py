"""Tests for the Diagnosis agent (P3). No Ollama, no benchmark: small hand-made situations.

Each test pins one behaviour the rest of the team relies on:
the contract, the cause and the release to blame, when to ask a human, and the LLM guardrails.
"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from agents.diagnosis.agent import DiagnosisAgent
from agents.diagnosis.context import Change, ContextStore
from agents.diagnosis.evidence.metrics import shape
from agents.diagnosis.feedback import incident_from_dict, record_review
from agents.diagnosis.knowledge import KnowledgeBase, parse
from agents.diagnosis.llm import LLMExplainer
from agents.diagnosis.node import make_diagnosis_node
from agents.triage.agent import RuleTriage
from common.schemas import Alert, Category, Diagnosis

T0 = datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc)
KB = KnowledgeBase(embed_model=None)


# ------------------------------------------------------------------ helpers
def metrics(lat_from=None, lat_mult=1.0, err_from=None, err_mult=1.0, ramp=False, until=None, hours=8):
    """Minute metrics around T0 with optional degradation (step or ramp) from a given minute offset."""
    rng = np.random.default_rng(0)
    t = pd.date_range(T0 - timedelta(hours=3), periods=hours * 60, freq="min", tz="UTC")
    lat = 120 * (1 + rng.normal(0, 0.01, len(t)))
    err = 0.004 * (1 + rng.normal(0, 0.03, len(t)))
    m = ((t - T0).total_seconds() / 60).to_numpy()
    for start, mult, arr in ((lat_from, lat_mult, lat), (err_from, err_mult, err)):
        if start is not None:
            on = (m >= start) & (m < (until if until is not None else 1e9))
            gain = np.clip((m - start) / 60, 0, 1) if ramp else np.ones_like(m)
            arr[on] *= 1 + (mult - 1) * gain[on]
    return pd.DataFrame({"timestamp": t, "latency_ms": lat, "error_rate": err})


def series_incident(features, start_min, minutes=10, ratio=3.0):
    start = T0 + timedelta(minutes=start_min)
    alerts = [Alert.from_dict(dict(detector="vote(rolling)", source="metrics", feature=f, score=ratio, threshold=1.0,
                                   value=1.0, dataset="deploy", timestamp=start.isoformat(),
                                   details={"voters": ["rolling"], "start": str(start),
                                            "end": str(start + timedelta(minutes=minutes - 1)), "n_points": minutes}))
              for f in features]
    return RuleTriage().process(alerts)[0]


def ctx(frame, *changes):
    return ContextStore(changes=list(changes), metrics={"deploy": frame})


def deploy(minute, cid="release-001"):
    return Change(T0 + timedelta(minutes=minute), "deploy", cid)


AGENT = DiagnosisAgent(kb=KB)


# ------------------------------------------------------------------ contract
def test_diagnosis_contract_round_trip():
    inc = series_incident(["latency_ms"], 5)
    d = AGENT.diagnose(inc, ctx(metrics(lat_from=0, lat_mult=1.6), deploy(0)))
    again = Diagnosis.from_dict(json.loads(json.dumps(d.to_dict(), default=str)))
    assert again.category == d.category and again.evidence[0].id == "E1"
    assert 0.0 <= d.confidence <= 1.0 and d.incident_id == inc.id
    assert d.suggested_action in {"rerun", "rollback", "retrain", "escalate", "none"}


# ------------------------------------------------------------------ scenario 2: bad deployment
def test_metric_shape_step_vs_ramp():
    before = np.full(60, 100.0)
    assert shape(before, np.full(90, 150.0))[1] == pytest.approx(0.0)          # step
    change, ramp = shape(before, 100 + np.linspace(0, 60, 90))                   # ramp
    assert change > 0.3 and ramp > 0.6


def test_latency_step_after_release_is_a_bad_deploy():
    d = AGENT.diagnose(series_incident(["latency_ms"], 5), ctx(metrics(lat_from=0, lat_mult=1.6), deploy(0)))
    assert d.category == Category.BAD_DEPLOY and d.sub_cause == "latency_regression"
    assert d.suspect == "release-001" and d.suggested_action == "rollback"


def test_blames_the_release_before_the_onset_not_the_latest():
    """A leak starts after release A; harmless release B lands on top. Rolling back B fixes nothing."""
    frame = metrics(lat_from=0, lat_mult=1.8, err_from=0, err_mult=3.0, ramp=True)
    d = AGENT.diagnose(series_incident(["latency_ms", "error_rate"], 60),
                       ctx(frame, deploy(0, "release-A"), deploy(40, "release-B")))
    assert d.category == Category.BAD_DEPLOY and d.suspect == "release-A"
    assert d.sub_cause == "resource_leak"
    assert any(e.kind == "later_changes" and "release-B" in e.statement for e in d.evidence)


def test_errors_right_after_config_push_is_a_config_change():
    d = AGENT.diagnose(series_incident(["error_rate"], 3),
                       ctx(metrics(err_from=0, err_mult=4.0), Change(T0, "config", "config-007")))
    assert d.category == Category.CONFIG_CHANGE and d.suspect == "config-007"


def test_degradation_without_any_change_is_infra():
    frame = metrics(lat_from=0, lat_mult=1.5, err_from=0, err_mult=2.5, ramp=True)
    d = AGENT.diagnose(series_incident(["latency_ms", "error_rate"], 30), ctx(frame, deploy(-600)))
    assert d.category == Category.INFRA and d.suspect == ""
    assert d.suggested_action != "rollback"


def test_recovery_after_rollback_is_no_fault():
    frame = metrics(lat_from=-120, lat_mult=1.6, until=0)        # bad from -120 min, rolled back at 0
    d = AGENT.diagnose(series_incident(["latency_ms"], 0), ctx(frame, deploy(-120)))
    assert d.category == Category.UNKNOWN and d.sub_cause == "recovery"
    assert any(e.kind == "metric_recovery" for e in d.evidence)


# ------------------------------------------------------------------ ci and logs
def ci_alerts(project, p, n=1, when=T0, commit=None):
    return [Alert.from_dict(dict(detector="flaky_model", source="ci", feature="build_failed", score=p / 0.15,
                                 threshold=1.0, value=p, dataset="travis",
                                 timestamp=(when + timedelta(minutes=5 * k)).isoformat(),
                                 details={"flaky_probability": p, "project": project, "build_id": k,
                                          "commit": commit or f"c{k:07d}",
                                          "evidence": {"project_flaky_rate": 0.5 if p > 0.5 else 0.1,
                                                       "recent_fail_rate": 0.1, "previous_build_failed": 0.0}}))
            for k in range(n)]


def test_ci_flaky_and_ci_infrastructure():
    lone = RuleTriage().process(ci_alerts("org/flaky", 0.85))
    assert AGENT.run(lone)[0].category == Category.FLAKY_TEST
    outage = RuleTriage().process([a for k in range(5) for a in ci_alerts(f"org/p{k}", 0.4)])
    diags = AGENT.run(outage)              # a batch: every project sees the others failing
    assert all(d.category == Category.INFRA for d in diags)


def test_novel_log_pattern_goes_to_a_human():
    alerts = [Alert.from_dict(dict(detector="block_pca", source="logs", feature="block_events", score=6.0,
                                   threshold=1.0, value=6.0, dataset="hdfs", timestamp=T0.isoformat(),
                                   details={"block_id": f"blk_{k}", "top_events": [{"event": e, "share": 0.33}
                                                                                     for e in ("E1", "E19", "E27")]}))
              for k in range(3)]
    d = AGENT.run(RuleTriage().process(alerts))[0]
    assert d.needs_human and d.suggested_action == "escalate"
    assert any("never seen" in w for w in d.warnings)


# ------------------------------------------------------------------ LLM guardrails
class FakeChat:
    def __init__(self, answer):
        self.answer = answer

    def __call__(self, model, messages):
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer)


def _llm_diagnose(answer):
    agent = DiagnosisAgent(kb=KB, llm=LLMExplainer("fake", chat=FakeChat(answer)))
    return agent.diagnose(series_incident(["latency_ms"], 5), ctx(metrics(lat_from=0, lat_mult=1.6), deploy(0)))


def test_llm_answer_is_used_when_it_checks_out():
    d = _llm_diagnose({"explanation": "Latency jumped right after the release [E2] [E3]; roll back [RB-bad-deploy].",
                       "agree": True, "alternative": None, "next_checks": ["roll back release-001"]})
    assert d.explanation.startswith("Latency jumped") and not d.warnings


@pytest.mark.parametrize("answer, warning", [
    ({"explanation": "Caused by the cache [E9].", "agree": True}, "unknown id"),
    ({"explanation": "Latency rose by 437 % [E2].", "agree": True}, "found nowhere"),
    ("I think it is a bad deploy.", "not JSON"),
])
def test_llm_answer_rejected_falls_back_to_template(answer, warning):
    d = _llm_diagnose(answer)
    assert d.explanation.startswith("Most likely cause")       # the template, not the LLM's text
    assert any(warning in w for w in d.warnings)
    assert d.category == Category.BAD_DEPLOY


def test_llm_disagreement_lowers_confidence_but_keeps_the_cause():
    base = _llm_diagnose({"explanation": "A bad release [E2].", "agree": True})
    d = _llm_diagnose({"explanation": "Probably the hosts [E3].", "agree": False, "alternative": "infra",
                       "alternative_reason": "could be saturation"})
    assert d.category == base.category == Category.BAD_DEPLOY
    assert d.confidence < base.confidence and any("argues for infra" in w for w in d.warnings)


def test_llm_down_is_not_fatal():
    def down(model, messages):
        raise ConnectionError("no Ollama")
    agent = DiagnosisAgent(kb=KB, llm=LLMExplainer("fake", chat=down))
    d = agent.diagnose(series_incident(["latency_ms"], 5), ctx(metrics(lat_from=0, lat_mult=1.6), deploy(0)))
    assert d.category == Category.BAD_DEPLOY and any("unavailable" in w for w in d.warnings)


# ------------------------------------------------------------------ knowledge base and learning
def test_knowledge_base_parse_and_source_filter():
    meta, body = parse("---\nid: X\ntags: [a:b, c:d]\nconfirmed: true\n---\nbody")
    assert meta == {"id": "X", "tags": ["a:b", "c:d"], "confirmed": True} and body == "body"
    hits = KB.search("latency jumped after the release, roll back", k=3, source="metrics")
    assert hits[0][0].doc.id == "RB-bad-deploy"
    assert all(c.doc.id != "RB-hdfs-datanode" for c, _ in hits)
    assert KB.runbook_for(Category.INFRA, "logs").id == "RB-hdfs-datanode"
    assert KB.runbook_for(Category.INFRA, "metrics").id == "RB-infra"


def test_human_correction_becomes_a_past_incident(tmp_path):
    kb = KnowledgeBase(embed_model=None)
    kb.root = tmp_path
    (tmp_path / "incidents").mkdir()
    inc = series_incident(["latency_ms"], 5)
    c = ctx(metrics(lat_from=0, lat_mult=1.6), deploy(0))
    d = DiagnosisAgent(kb=kb).diagnose(inc, c)
    path = record_review(inc, d, "correct", Category.INFRA, "slow_dependency", note="the DB was slow",
                         kb=kb, reviews=tmp_path / "reviews.jsonl", incidents_dir=tmp_path / "incidents")
    assert path.exists() and "category: infra" in path.read_text()
    again = DiagnosisAgent(kb=kb).diagnose(inc, c)            # the next similar incident remembers it
    assert any(e.kind == "similar_incident" and "confirmed by a human" in e.statement for e in again.evidence)
    assert incident_from_dict(json.loads(json.dumps(inc.to_dict(), default=str))).id == inc.id


def test_langgraph_node_contract():
    inc = series_incident(["latency_ms"], 5)
    out = make_diagnosis_node(AGENT)({"to_diagnosis": [inc],
                                      "context": ctx(metrics(lat_from=0, lat_mult=1.6), deploy(0))})
    assert [d.incident_id for d in out["diagnoses"]] == [inc.id]
    assert out["log"][0]["category"] == "bad_deploy"


def test_pipeline_graph_runs_monitor_triage_diagnosis():
    """The team graph (pipeline.py): monitor -> triage -> to_diagnosis -> diagnosis, through LangGraph."""
    pytest.importorskip("langgraph")
    from pipeline import build_graph, make_layer2
    from agents.triage.memory import Memory

    start = T0 + timedelta(minutes=5)
    alert = {"id": "alert-1", "timestamp": start.isoformat(), "source": "metrics", "signal": "latency_ms",
             "score": 5.0, "kind": "anomaly", "dataset": "deploy",            # = P1's monitor_node format
             "details": {"voters": ["rolling"], "start": str(start), "end": str(start + timedelta(minutes=30)),
                         "n_points": 31}}
    monitor = lambda state: {"alerts": [alert], "context": ctx(metrics(lat_from=0, lat_mult=1.6), deploy(0)),
                             "log": [{"node": "monitor"}]}
    graph = build_graph(monitor, make_layer2(Memory(), use_llm=False), make_diagnosis_node(AGENT))
    out = graph.invoke({"log": []})
    assert [i.id for i in out["to_diagnosis"]] == [d.incident_id for d in out["diagnoses"]]
    d = out["diagnoses"][0]
    assert d.category == Category.BAD_DEPLOY and d.suspect == "release-001"
    assert [x.get("node", x.get("agent")) for x in out["log"]][-1] == "diagnosis"
