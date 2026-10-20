import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from config import settings

# the team contract: where the signal comes from
VALID_SOURCES = {"ci", "logs", "metrics", "model"}
VALID_KINDS = {"anomaly", "drift"}


@dataclass
class Alert:
    detector: str
    source: str            # contract field: ci / logs / metrics / model
    feature: str           # the signal name (called "signal" in the contract)
    score: float           # how many times above its limit (1.0 = just at the limit)
    threshold: float
    timestamp: datetime
    value: float
    dataset: str = ""      # where the data came from: nyc / travis / hdfs ...
    kind: str = "anomaly"  # "anomaly" (something strange) or "drift" (the data itself changed)
    details: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"alert-{uuid4().hex[:8]}")

    def __post_init__(self):
        if not isinstance(self.score, (int, float)):
            raise ValueError("score must be numeric")
        if self.threshold is None or self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.score < 0:
            raise ValueError("score must be >= 0")
        if self.source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}, got '{self.source}'")
        if self.kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}, got '{self.kind}'")

    def to_contract(self) -> dict:
        """The key fields of the team's Alert contract."""
        return {"id": self.id, "timestamp": self.timestamp, "source": self.source,
                "signal": self.feature, "score": round(self.score, 4)}

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "source": self.source,
            "dataset": self.dataset,
            "detector": self.detector,
            "feature": self.feature,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "value": self.value,
            "details": json.dumps(self.details, default=str),
        }


def reset_log():
    """Start a fresh logs/alerts.csv."""
    settings.ALERTS_CSV.unlink(missing_ok=True)


def log_alert(alert):
    """Append one alert to logs/alerts.csv (the header is written once)."""
    row = alert.to_row()
    new_file = not settings.ALERTS_CSV.exists()
    with open(settings.ALERTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)
