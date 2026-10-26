"""Shared contracts.

Alert     : P1's format (agents/monitor/alert.py in his repo). Do not change it here; change it with P1.
Incident  : what Triage (P2) sends to Diagnosis (P3).
Diagnosis : what Diagnosis (P3) sends to Decision (P4).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


# ---------------------------------------------------------------- Alert (P1) ---
@dataclass
class Alert:
    detector: str            # "vote(seasonal)" | "vote(rolling)" | "vote(psi)" | "block_pca" | "flaky_model"
    source: str              # signal type: "ci" | "logs" | "metrics" | "model"
    feature: str             # e.g. "trip_count", "block_events", "build_failed" ("signal" in the contract)
    score: float             # how many times above its limit (1.0 = just at the limit)
    threshold: float         # always 1.0 now
    timestamp: datetime      # UTC; for merged series alerts = start of the episode
    value: float             # observed value at the peak
    dataset: str = ""        # where the data comes from: "nyc" | "travis" | "hdfs" | "deploy"
    kind: str = "anomaly"    # "anomaly" (something strange) | "drift" (the data itself changed)
    details: dict = field(default_factory=dict)   # extra evidence, depends on the detector
    id: str = field(default_factory=lambda: f"alert-{uuid4().hex[:8]}")

    # --- helpers used by Triage (do not change P1's fields) ---
    @property
    def ratio(self) -> float:
        """How far above its limit: 1 = just fired, 2 = clearly abnormal."""
        return self.score / self.threshold if self.threshold else self.score

    @property
    def voters(self) -> list[str]:
        """Detectors that agree on this alert (1 if no voting)."""
        return list(self.details.get("voters") or [self.detector])

    @property
    def family(self) -> str:
        """Which grouping logic Triage must use for this alert."""
        if self.source == "logs":
            return "logs"      # one alert per HDFS block, no real time
        if self.source == "ci":
            return "ci"        # one alert per failed build
        return "series"        # nyc / deploy time series, already merged into episodes by P1

    @property
    def start(self) -> datetime:
        """First abnormal moment (P1 merges consecutive hours into one alert)."""
        return _to_dt(self.details.get("start", self.timestamp))

    @property
    def end(self) -> datetime:
        """Last abnormal moment (= start for single-point alerts)."""
        return _to_dt(self.details.get("end", self.start))

    @property
    def duration_hours(self) -> float:
        """Length of the episode in hours (1 point = at least one step long)."""
        n = int(self.details.get("n_points", 1))
        span = (self.end - self.start).total_seconds() / 3600
        step = span / (n - 1) if n > 1 else 1.0   # hourly data -> 1 h, minute data -> 1/60 h
        return round(span + step, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Alert":
        """Accepts a CSV row (logs/alerts.csv) or a dict from P1's monitor_node."""
        d = dict(d)
        if "feature" not in d and "signal" in d:       # monitor_node renames feature -> signal
            d["feature"] = d.pop("signal")
        d.setdefault("detector", "unknown")             # monitor_node does not send these three
        d.setdefault("threshold", 1.0)
        d.setdefault("value", math.nan)
        d["timestamp"] = _to_dt(d["timestamp"])
        if isinstance(d.get("details"), str):          # CSV: details is JSON text
            d["details"] = json.loads(d["details"] or "{}")
        for k in ("score", "threshold", "value"):
            d[k] = float(d[k])
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _to_dt(x) -> datetime:
    return x if isinstance(x, datetime) else datetime.fromisoformat(str(x))


# ---------------------------------------------------------- Incident (P2 -> P3) ---
class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Decision(str, Enum):
    INCIDENT = "incident"   # sent to Diagnosis
    WATCH = "watch"         # kept, re-checked if it continues
    NOISE = "noise"         # not sent, but logged


@dataclass
class Incident:
    id: str
    source: str                         # signal type: ci / logs / metrics / model
    features: list[str]                 # features abnormal in this incident
    alert_ids: list[str]
    start: datetime                     # first abnormal moment
    end: datetime                       # last abnormal moment
    duration_hours: float
    peak_ratio: float                   # max score/threshold
    max_voters: int                     # how many detectors agreed at best
    severity: Severity
    decision: Decision
    summary: str
    reasons: list[str] = field(default_factory=list)   # why grouped / why this decision
    alerts: list[Alert] = field(default_factory=list)
    dataset: str = ""                   # nyc / travis / hdfs / deploy
    kind: str = "anomaly"               # "drift" if every alert in it is drift

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"], d["end"] = self.start.isoformat(), self.end.isoformat()
        d["severity"], d["decision"] = self.severity.value, self.decision.value
        d["alerts"] = [a.to_dict() for a in self.alerts]
        return d

# -------------------------------------------------------- Diagnosis (P3 -> P4) ---
class Category(str, Enum):
    """Root causes the Diagnosis agent can name (the team plan's list, plus what the data needed)."""
    FLAKY_TEST = "flaky_test"             # the test is unreliable, the code is fine -> re-run
    CODE_REGRESSION = "code_regression"   # a commit really broke the build -> the author fixes it
    BAD_DEPLOY = "bad_deploy"             # a new release degraded the service -> roll back
    CONFIG_CHANGE = "config_change"       # a config push (not a code release) broke something -> revert it
    INFRA = "infra"                       # machines, network, storage: nothing we shipped
    DATA_DRIFT = "data_drift"             # model inputs slowly changed -> retraining question
    DATA_QUALITY = "data_quality"         # the data itself is broken or strange (missing, outlier burst)
    UNKNOWN = "unknown"                   # the evidence does not point anywhere: a human must look


@dataclass
class Evidence:
    """One verifiable fact the diagnosis rests on. The LLM may only cite these ids."""
    id: str                     # "E1", "E2", ... (unique inside one Diagnosis)
    kind: str                   # what was checked, e.g. "deploy_before", "metric_shape", "flaky_probability"
    statement: str              # one sentence, with the numbers, a human can check
    source: str                 # where it comes from: "alert" | "change_log" | "metrics" | "logs" | "knowledge_base"
    supports: dict = field(default_factory=dict)   # {category value: weight}: + for, - against (log-odds)
    data: dict = field(default_factory=dict)       # the raw numbers behind the statement


@dataclass
class Hypothesis:
    category: Category
    probability: float          # 0-1, all hypotheses of one diagnosis sum to 1
    sub_cause: str = ""         # finer cause when the evidence allows it, e.g. "resource_leak"


@dataclass
class Diagnosis:
    incident_id: str
    category: Category
    explanation: str                    # short cause, cites evidence ids like [E1] and runbooks like [RB-bad-deploy]
    evidence: list[Evidence]
    confidence: float                   # 0-1: how sure P3 is of `category` (P4 multiplies it by the action risk)
    sub_cause: str = ""
    suspect: str = ""                   # the change to undo when there is one (release / config id) -> P5's target
    hypotheses: list[Hypothesis] = field(default_factory=list)   # ranked alternatives, best first
    suggested_action: str = "escalate"  # hint for P4: rerun / rollback / retrain / escalate / none
    needs_human: bool = False           # True when P3 itself thinks nobody should act on this alone
    runbooks: list[str] = field(default_factory=list)            # knowledge-base ids that were used
    method: str = ""                    # how it was produced, e.g. "rules+ml+llm(llama3.1:8b)"
    warnings: list[str] = field(default_factory=list)            # guardrails that fired (LLM misread, ...)
    id: str = field(default_factory=lambda: f"diag-{uuid4().hex[:8]}")
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["hypotheses"] = [{**asdict(h), "category": h.category.value} for h in self.hypotheses]
        d["created"] = self.created.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Diagnosis":
        d = dict(d)
        d["category"] = Category(d["category"])
        d["evidence"] = [e if isinstance(e, Evidence) else Evidence(**e) for e in d.get("evidence", [])]
        d["hypotheses"] = [h if isinstance(h, Hypothesis) else Hypothesis(**{**h, "category": Category(h["category"])})
                           for h in d.get("hypotheses", [])]
        if "created" in d:
            d["created"] = _to_dt(d["created"])
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
