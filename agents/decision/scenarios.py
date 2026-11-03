"""The deck's four scenarios (slides 13-16) as ready-made Diagnoses, plus three cases the table also covers.

Used by the demo and the approval page's "load demo" button. Numbers are the deck's (0.9 flaky, release v2.3,
PSI 0.31 on trip_distance, confidence 0.35 for an unknown failure).
"""
from __future__ import annotations

from datetime import datetime, timezone

from common.schemas import (Alert, Category, Decision, Diagnosis, Evidence, Hypothesis, Incident,
                            Severity)

C = Category
T0 = datetime(2026, 1, 20, 10, 0, tzinfo=timezone.utc)      # a weekday morning: retraining is allowed


def _incident(iid: str, source: str, dataset: str, features: list[str], summary: str, project: str = "") -> Incident:
    a = Alert(detector="demo", source=source, feature=features[0], score=3.0, threshold=1.0, timestamp=T0, value=1.0,
              dataset=dataset, details={"project": project} if project else {})
    return Incident(id=iid, source=source, features=features, alert_ids=[a.id], start=T0, end=T0, duration_hours=0.5,
                    peak_ratio=3.0, max_voters=1, severity=Severity.HIGH, decision=Decision.INCIDENT,
                    summary=summary, alerts=[a], dataset=dataset)


def _diag(iid, cat, conf, action, explanation, evidence, sub="", suspect="", needs_human=False, warnings=(), alt=None):
    ev = [Evidence(f"E{n}", "demo", text, src) for n, (text, src) in enumerate(evidence, 1)]
    hyp = [Hypothesis(cat, conf, sub)] + ([Hypothesis(alt, round(1 - conf, 2))] if alt else [])
    return Diagnosis(incident_id=iid, category=cat, sub_cause=sub, suspect=suspect, explanation=explanation, evidence=ev,
                     confidence=conf, hypotheses=hyp, suggested_action=action, needs_human=needs_human,
                     warnings=list(warnings), method="demo")


def deck_scenarios() -> list[tuple[str, Incident, Diagnosis]]:
    """(name, incident, diagnosis)"""
    out = []
    i = _incident("DEMO-1", "ci", "travis", ["build_failed"], "6 failed builds of project shop-api on one commit", "shop-api")
    out.append(("1 · Flaky test in CI", i, _diag(
        "DEMO-1", C.FLAKY_TEST, 0.90, "rerun", "The same test passed and failed on the same commit [E1]: flaky.",
        [("The test passed on the same code 2 h earlier (flaky probability 0.93).", "ci")], "flaky_timing")))
    i = _incident("DEMO-2", "metrics", "deploy", ["error_rate"], "error_rate +372% five minutes after release v2.3")
    out.append(("2 · Bad deployment", i, _diag(
        "DEMO-2", C.BAD_DEPLOY, 0.93, "rollback", "Errors jumped 5 min after v2.3 [E1][E2]; the previous version is stable.",
        [("Release v2.3 went out at 14:02; error_rate jumped +372% at 14:07 (step).", "change_log"),
         ("Past incident PI-001 looks like this one (similarity 0.92).", "knowledge_base")],
        "error_regression", "release-v2.3")))
    i = _incident("DEMO-3", "model", "nyc", ["avg_distance"], "avg_distance drifted: PSI 0.31 > 0.2")
    out.append(("3 · Data drift on the ML model", i, _diag(
        "DEMO-3", C.DATA_DRIFT, 0.82, "retrain", "trip_distance moved (PSI 0.31 > 0.2) and it is not seasonal [E1].",
        [("PSI of trip_distance is 0.31 over the last 7 days; the same weeks last year were stable.", "metrics")])))
    i = _incident("DEMO-4", "logs", "hdfs", ["block_events"], "a log pattern never seen before")
    out.append(("4 · Unknown failure: knowing its limits", i, _diag(
        "DEMO-4", C.UNKNOWN, 0.35, "escalate", "No known cause fits; no runbook or past incident matches.",
        [("No runbook or past incident is similar to this log pattern.", "knowledge_base")], needs_human=True,
        warnings=["pattern never seen in the knowledge base: confidence lowered"])))
    # ---- the table also covers these
    i = _incident("DEMO-5", "metrics", "deploy", ["error_rate"], "error_rate +300% right after config push config-003")
    out.append(("5 · Bad config push", i, _diag(
        "DEMO-5", C.CONFIG_CHANGE, 0.86, "rollback", "Config push config-003 broke the endpoint [E1].",
        [("Config push config-003 at 15:28; metrics degraded at 15:28.", "change_log")], "bad_config", "config-003")))
    i = _incident("DEMO-6", "ci", "travis", ["build_failed"], "failed build of project shop-api, second time today", "shop-api")
    out.append(("6 · The re-run failed again (same project as 1)", i, _diag(
        "DEMO-6", C.FLAKY_TEST, 0.88, "rerun", "Looks flaky again [E1].",
        [("Flaky probability 0.88 on the same project.", "ci")])))
    i = _incident("DEMO-7", "ci", "travis", ["build_failed"], "failed build of project billing", "billing")
    out.append(("7 · LLM and classifier disagree", i, _diag(
        "DEMO-7", C.FLAKY_TEST, 0.84, "rerun", "Probably flaky [E1].", [("Flaky probability 0.84.", "ci")],
        warnings=["LLM argues for code_regression: the failing assertion changed with the last commit "
                  "(cause kept, confidence lowered)"], alt=C.CODE_REGRESSION)))
    return out
