"""Shared contracts.

Alert    : P1's format (agents/monitor/alert.py in his repo). Do not change it here; change it with P1.
Incident : what Triage (P2) sends to Diagnosis (P3).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
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