from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Alert(BaseModel):
    id: str = Field(default_factory=lambda: f"alert-{uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Literal["ci", "logs", "metrics", "model"]
    signal: str
    score: float = Field(ge=0)
    threshold: float = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)