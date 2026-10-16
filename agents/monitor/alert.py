from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class Alert:
    detector: str
    source: str
    feature: str
    score: float
    threshold: float
    timestamp: datetime
    value: float
    details: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"alert-{uuid4().hex[:8]}")

    def __post_init__(self):
        if not isinstance(self.score, (int, float)):
            raise ValueError("score must be numeric")
        if self.threshold is None or self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.score < 0:
            raise ValueError("score must be >= 0")

    def to_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "detector": self.detector,
            "source": self.source,
            "feature": self.feature,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "value": self.value,
        }