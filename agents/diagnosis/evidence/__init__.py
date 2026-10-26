"""Step 1 of Diagnosis: collect EVIDENCE about one incident.

Each collector looks at one thing and writes what it found in two forms:

    Evidence   a sentence with its numbers + which causes it supports (weights in log-odds)
               -> read by humans, by the rule scorer, and cited by the LLM ([E3])
    features   numbers in a fixed order (FEATURES)
               -> read by the root-cause classifier

    collector       looks at                     useful for
    incident.py     the Incident itself           everything (what, how strong, how long)
    changes.py      deploys / config pushes       bad_deploy, config_change, infra
    metrics.py      raw minute metrics            sub-cause: latency / errors / leak / saturation
    ci.py           flaky model + other projects  flaky_test, code_regression, infra
    logs.py         HDFS events -> templates      infra, unknown
    data.py         drift / model inputs          data_drift, data_quality
    history.py      similar past incidents (KB)   whatever a human confirmed before

Collectors never decide: they only report. Deciding is the scorer's and classifier's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from common.schemas import Category, Evidence, Incident

from ..config import DiagnosisConfig
from ..context import Change, ContextStore

# the classifier's input, always in this order (missing values stay at their default)
FEATURES = [
    # incident
    "src_ci", "src_logs", "src_metrics", "src_model", "kind_drift",
    "n_features", "log_peak", "log_duration_h", "max_voters", "log_n_alerts",
    # changes
    "has_deploy", "mins_since_deploy", "has_config", "mins_since_config", "no_change",
    # metric shape
    "lat_change", "err_change", "lat_ramp", "err_ramp", "lat_moved", "err_moved", "recovery",
    # ci
    "flaky_mean", "flaky_max", "project_flaky_rate", "recent_fail_rate", "prev_failed_share",
    "n_commits", "peer_projects",
    # logs
    "log_n_blocks", "share_error_events", "share_replication_events", "novel_pattern",
    # model inputs
    "psi",
]
DEFAULTS = {"mins_since_deploy": 999.0, "mins_since_config": 999.0, "recent_fail_rate": 0.5,
            "project_flaky_rate": 0.25}


@dataclass
class Findings:
    evidence: list[Evidence] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=lambda: {f: DEFAULTS.get(f, 0.0) for f in FEATURES})
    sub_causes: dict[Category, str] = field(default_factory=dict)   # finer cause, per category
    novel: bool = False                                              # looks like nothing seen before
    suspect: Change | None = None                                    # the change blamed (changes.py)
    onset: datetime | None = None                                    # when the metrics started to degrade

    def add(self, kind: str, statement: str, source: str, supports: dict | None = None, **data) -> Evidence:
        e = Evidence(id=f"E{len(self.evidence) + 1}", kind=kind, statement=statement, source=source,
                     supports={Category(c).value: round(float(w), 2) for c, w in (supports or {}).items()},
                     data=data)
        self.evidence.append(e)
        return e

    def vector(self) -> list[float]:
        return [float(self.features[f]) for f in FEATURES]


def collect(inc: Incident, ctx: ContextStore, cfg: DiagnosisConfig, kb=None) -> Findings:
    from . import changes, ci, data, history, incident, logs, metrics   # noqa: E401 (one module per collector)

    f = Findings()
    incident.collect(inc, f)
    if inc.source == "metrics":
        changes.collect(inc, ctx, cfg, f)
        metrics.collect(inc, ctx, cfg, f)
    elif inc.source == "ci":
        ci.collect(inc, ctx, cfg, f)
    elif inc.source == "logs":
        logs.collect(inc, cfg, f, kb)
    elif inc.source == "model":
        data.collect(inc, cfg, f)
    if kb is not None:
        history.collect(inc, f, kb)
    return f
