"""What the Diagnosis agent can look at besides the Incident itself.

Triage hands over WHAT is abnormal. To say WHY, Diagnosis must look around it:

    changes   what changed just before? (deploys, config pushes)       -> change log of the pipeline
    metrics   what did the raw metrics do around it? (step, ramp, ...)  -> minute metrics
    peers     what else failed at the same time? (other incidents)     -> the current batch + history

The store only holds what is observable in production. In the simulation, P1's deploy table also
says which deploys are bad and how: those truth columns are dropped here (scenarios.py keeps
them for scoring only), so the agent can never peek at the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from common.schemas import Incident


@dataclass(frozen=True)
class Change:
    timestamp: datetime
    kind: str                 # "deploy" (new code) | "config" (config push, no new code)
    id: str                   # release / commit / config version
    service: str = "app"
    note: str = ""


@dataclass
class ContextStore:
    changes: list[Change] = field(default_factory=list)
    metrics: dict[str, pd.DataFrame] = field(default_factory=dict)   # dataset -> timestamp + signal columns
    peers: list[Incident] = field(default_factory=list)

    # ------------------------------------------------------------------ changes
    def changes_before(self, t: datetime, window_min: float) -> list[Change]:
        """Changes in [t - window, t], most recent first."""
        lo = t - timedelta(minutes=window_min)
        return sorted((c for c in self.changes if lo <= c.timestamp <= t), key=lambda c: c.timestamp, reverse=True)

    # ------------------------------------------------------------------ metrics
    def metric_window(self, dataset: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Rows with start <= timestamp < end. Binary search (called many times per incident):
        the frames are sorted by timestamp (P1's loaders and the simulator write them that way)."""
        df = self.metrics.get(dataset)
        if df is None or df.empty:
            return pd.DataFrame()
        t = df["timestamp"]
        return df.iloc[t.searchsorted(pd.Timestamp(start)): t.searchsorted(pd.Timestamp(end))]

    # ------------------------------------------------------------------ peers
    def peers_near(self, inc: Incident, hours: float) -> list[Incident]:
        """Other incidents of the same source that overlap [start - h, end + h]."""
        w = timedelta(hours=hours)
        return [p for p in self.peers if p.id != inc.id and p.source == inc.source
                and p.start <= inc.end + w and p.end >= inc.start - w]

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_deploy_table(cls, metrics: pd.DataFrame, deploys: pd.DataFrame, dataset: str = "deploy",
                          config_pushes: pd.DataFrame | None = None) -> "ContextStore":
        """From P1's ingest/deploy.py output. Only the deploy TIME is kept: kind / bad / size are the truth."""
        changes = [Change(pd.Timestamp(t).to_pydatetime(), "deploy", f"release-{k + 1:03d}")
                   for k, t in enumerate(deploys["timestamp"])]
        if config_pushes is not None:
            changes += [Change(pd.Timestamp(t).to_pydatetime(), "config", f"config-{k + 1:03d}")
                        for k, t in enumerate(config_pushes["timestamp"])]
        return cls(changes=sorted(changes, key=lambda c: c.timestamp), metrics={dataset: metrics})

    def with_peers(self, incidents: list[Incident]) -> "ContextStore":
        return ContextStore(self.changes, self.metrics, list(incidents))
