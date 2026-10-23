"""Step 1 of Triage: put alerts that describe the SAME problem into one group.

Each family of alerts needs its own rule:
  A. series : episodes of the same dataset that overlap (or almost touch) in time
  B. logs   : HDFS blocks with the same top-3 events (they failed the same way)
  C. ci     : failed builds of the same project on the same day

Every group keeps a list of reasons, so a human can see WHY alerts were put together.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

from common.schemas import Alert

from .config import TriageConfig


@dataclass
class Group:
    family: str                 # "series" | "logs" | "ci"
    key: str                    # what the alerts have in common (dataset, pattern, project/day)
    alerts: list[Alert] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ A. series
def group_series(alerts: list[Alert], cfg: TriageConfig) -> list[Group]:
    """Sweep through the episodes of each dataset in time order.
    An episode joins the current group if it starts before the group ends (+ a small gap).
    Drift and anomaly are grouped SEPARATELY: a drift can last months and would
    otherwise swallow every unrelated anomaly that happens during it."""
    groups: list[Group] = []
    by_dataset = defaultdict(list)
    for a in alerts:
        by_dataset[(a.dataset, a.kind)].append(a)

    for (ds, kind), items in by_dataset.items():
        gap = timedelta(hours=cfg.series_gap_hours.get(ds, 1.0))
        current, current_end = None, None
        for a in sorted(items, key=lambda x: x.start):
            if current is not None and a.start <= current_end + gap:
                current.alerts.append(a)
                why = "overlaps" if a.start <= current_end else f"starts within {gap} of"
                current.reasons.append(f"{a.feature} {a.start:%m-%d %H:%M}: {why} the open episode")
                current_end = max(current_end, a.end)
            else:
                current = Group("series", ds, [a],
                                [f"{a.feature} {a.start:%m-%d %H:%M}: new {kind} incident on {ds}"])
                current_end = a.end
                groups.append(current)
    return groups


# ------------------------------------------------------------------ B. logs
def log_pattern(a: Alert) -> str:
    """The 3 events that made the block look strange, sorted -> e.g. 'E2+E3+E4'."""
    return "+".join(sorted(e["event"] for e in a.details.get("top_events", [])))


def group_logs(alerts: list[Alert], cfg: TriageConfig) -> list[Group]:
    groups: dict[str, Group] = {}
    for a in alerts:
        p = log_pattern(a)
        if p not in groups:
            groups[p] = Group("logs", p, reasons=[f"blocks whose main events are {p}"])
        groups[p].alerts.append(a)
    for g in groups.values():
        g.reasons.append(f"{len(g.alerts)} block(s) failed with the same event pattern")
    return list(groups.values())


# ------------------------------------------------------------------ C. ci
def group_ci(alerts: list[Alert], cfg: TriageConfig) -> list[Group]:
    groups: dict[str, Group] = {}
    for a in alerts:
        k = f"{a.details.get('project', '?')} {a.timestamp:%Y-%m-%d}"
        if k not in groups:
            groups[k] = Group("ci", k, reasons=[f"failed builds of {k}"])
        groups[k].alerts.append(a)
    for g in groups.values():
        commits = {a.details.get("commit") for a in g.alerts}
        g.reasons.append(f"{len(g.alerts)} failed build(s) on {len(commits)} commit(s), same project, same day")
    return list(groups.values())


# ------------------------------------------------------------------ all
def group_all(alerts: list[Alert], cfg: TriageConfig) -> list[Group]:
    by_family = defaultdict(list)
    for a in alerts:
        by_family[a.family].append(a)
    return (group_series(by_family["series"], cfg)
            + group_logs(by_family["logs"], cfg)
            + group_ci(by_family["ci"], cfg))