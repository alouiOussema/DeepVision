"""The deck's four scenarios (slides 13-16), each with its SUCCESS and its FAILURE branch, for the Action agent.

Every case is an Incident + a Diagnosis (as P2 / P3 would send them) + the simulated world P5 acts on
(what the CI re-run returns, what the service does after a rollback, how good the retrained model is).
The demo sends them through the real Decision agent first, so the orders P5 receives are real ActionDecisions.
Numbers are the deck's: flaky 0.9, release v2.3, PSI 0.31 on trip_distance, confidence 0.35 for an unknown failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from common.schemas import Alert, Category, Decision, Diagnosis, Evidence, Hypothesis, Incident, Severity

from .backends import Backends, SimCI, SimDeploy, SimModel

C = Category
T0 = datetime(2026, 1, 20, 10, 0, tzinfo=timezone.utc)      # a weekday morning: retraining is allowed


@dataclass
class Case:
    name: str
    story: str                              # what should happen, in one line
    incident: Incident
    diagnosis: Diagnosis
    world: Callable[[], Backends]           # a fresh simulated world for this case
    approver: str = ""                      # who approves on the approval page, when Decision asks a human
    reject: bool = False                    # the human says no
    expected: str = ""                      # executed | failed | reverted | skipped
    tags: list[str] = field(default_factory=list)


def _incident(iid: str, source: str, dataset: str, feature: str, summary: str, details: dict | None = None) -> Incident:
    a = Alert(detector="demo", source=source, feature=feature, score=3.0, threshold=1.0, timestamp=T0, value=1.0,
              dataset=dataset, details=details or {})
    return Incident(id=iid, source=source, features=[feature], alert_ids=[a.id], start=T0, end=T0, duration_hours=0.5,
                    peak_ratio=3.0, max_voters=1, severity=Severity.HIGH, decision=Decision.INCIDENT,
                    summary=summary, alerts=[a], dataset=dataset)


def _diag(iid, cat, conf, action, explanation, evidence, sub="", suspect="", needs_human=False, warnings=()):
    ev = [Evidence(f"E{n}", "demo", text, src) for n, (text, src) in enumerate(evidence, 1)]
    return Diagnosis(incident_id=iid, category=cat, sub_cause=sub, suspect=suspect, explanation=explanation, evidence=ev,
                     confidence=conf, hypotheses=[Hypothesis(cat, conf, sub)], suggested_action=action,
                     needs_human=needs_human, warnings=list(warnings), method="demo")


def ci_case(iid, project, build_id, outcome, name, story, expected) -> Case:
    inc = _incident(iid, "ci", "travis", "build_failed", f"6 failed-test alerts of project {project} on one commit",
                    {"project": project, "build_id": build_id, "commit": "a1b2c3d", "flaky_probability": 0.93})
    diag = _diag(iid, C.FLAKY_TEST, 0.90, "rerun", "The same test passed and failed on the same commit [E1]: flaky.",
                 [("The test passed on the same code 2 h earlier (flaky probability 0.93).", "ci")], "flaky_timing")
    return Case(name, story, inc, diag, lambda: Backends(ci=SimCI({str(build_id): [outcome]})), expected=expected)


def deploy_case(iid, target, world, name, story, expected) -> Case:
    inc = _incident(iid, "metrics", "deploy", "error_rate", f"error_rate +372% five minutes after {target}")
    diag = _diag(iid, C.BAD_DEPLOY, 0.93, "rollback", f"Errors jumped 5 min after {target} [E1]; the previous version is stable.",
                 [(f"{target} went out at 14:02; error_rate jumped +372% at 14:07 (step).", "change_log")],
                 "error_regression", target)
    return Case(name, story, inc, diag, world, approver="yassine (on-call SRE)", expected=expected)


def drift_case(iid, champion, challenger, name, story, expected, feature="avg_distance") -> Case:
    inc = _incident(iid, "model", "nyc", feature, f"{feature} drifted: PSI 0.31 > 0.2")
    diag = _diag(iid, C.DATA_DRIFT, 0.82, "retrain", "trip_distance moved (PSI 0.31 > 0.2) [E1].",
                 [("PSI of trip_distance is 0.31 over the last 7 days.", "metrics")])
    return Case(name, story, inc, diag, lambda: Backends(model=SimModel(champion, challenger)), expected=expected)


def deck_cases() -> list[Case]:
    v22, v23 = "release-v2.2", "release-v2.3"
    cases = [
        ci_case("ACT-1", "shop-api", 48211, "passed", "1 · Flaky test: the re-run passes",
                "re-run once -> green -> nobody paged", "executed"),
        ci_case("ACT-1b", "billing-api", 48377, "failed", "1b · Flaky test: the re-run is red again",
                "red twice = a real bug -> failed -> the author is told", "failed"),
        deploy_case("ACT-2", v23, lambda: Backends(deploy=SimDeploy([v22, v23], {v23: (0.06, 40.0)})),
                    "2 · Bad deployment: a human approves the rollback, it works",
                    "approved by the on-call SRE -> rollback -> metrics back in band after 15 min", "executed"),
        deploy_case("ACT-2b", "release-v3.1", lambda: Backends(deploy=SimDeploy(["release-v3.0", "release-v3.1"],
                                                                                {"release-v3.1": (0.005, 5.0)},
                                                                                ambient=(0.05, 50.0))),
                    "2b · Bad deployment: the rollback does not fix it",
                    "errors still there after the rollback -> failed -> on-call takes over (infra?)", "failed"),
        deploy_case("ACT-2c", "release-v2.5", lambda: Backends(deploy=SimDeploy(["release-v2.4", "release-v2.5"],
                                                                                {"release-v2.5": (-0.03, -20.0)},
                                                                                ambient=(0.06, 40.0))),
                    "2c · Bad deployment: the rollback makes it worse",
                    "v2.5 was carrying a fix -> worse after the rollback -> redeployed -> reverted", "reverted"),
        drift_case("ACT-3", 2.10, 1.70, "3 · Data drift: the challenger beats the champion",
                   "retrain -> challenger better -> NOT deployed, the ML engineer approves the deploy", "executed"),
        drift_case("ACT-3b", 2.10, 2.25, "3b · Data drift: the challenger is worse (seasonal drift)",
                   "retrain -> challenger worse -> champion kept -> failed -> ML engineer gets the drift report", "failed",
                   feature="avg_fare"),
        Case("4 · Unknown failure: knowing its limits", "confidence 0.35 -> escalate -> a human is told, nothing changes",
             _incident("ACT-4", "logs", "hdfs", "block_events", "a log pattern never seen before"),
             _diag("ACT-4", C.UNKNOWN, 0.35, "escalate", "No known cause fits; no runbook or past incident matches.",
                   [("No runbook or past incident is similar to this log pattern.", "knowledge_base")], needs_human=True),
             Backends, expected="executed"),
        deploy_case("ACT-5", v23, lambda: Backends(deploy=SimDeploy([v22, v23], {v23: (0.06, 40.0)})),
                    "5 · Safety: a human REJECTS the rollback", "rejected -> the Action agent does nothing", "skipped"),
    ]
    cases[-1].reject = True
    return cases
