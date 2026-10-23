"""Rule-based Triage: alerts -> incidents, in 4 steps.

  1. group     alerts that describe the same problem   (grouping.py)
  2. measure   strength, width, duration, agreement     (rules.py)
  3. judge     severity + decision incident/watch/noise (rules.py)
  4. explain   a summary and the reasons of every step, kept in the Incident

P1's monitor sends a whole batch of alerts at once, so Triage works on a batch too.
"""
from __future__ import annotations

from collections import Counter
from itertools import count

from common.schemas import Alert, Incident

from .config import TriageConfig
from .grouping import Group, group_all
from .rules import Measures, judge


class RuleTriage:
    def __init__(self, cfg: TriageConfig | None = None, id_prefix: str = "INC"):
        self.cfg = cfg or TriageConfig()
        self.id_prefix = id_prefix              # e.g. "HIST" for the history run used by the tools
        self._ids = count(1)
        self.assignment: dict[str, str] = {}     # alert id -> incident id (used by evaluation)
        self.stats = Counter()

    def process(self, alerts: list[Alert]) -> list[Incident]:
        incidents = [self._close(g) for g in group_all(alerts, self.cfg)]       # step 1
        self.stats["alerts_in"] += len(alerts)
        self.stats.update(f"decision_{i.decision.value}" for i in incidents)
        return sorted(incidents, key=lambda i: i.start)

    # ------------------------------------------------------------ steps 2-4 ---
    def _close(self, g: Group) -> Incident:
        sev, dec, m, why = judge(g, self.cfg)                                   # steps 2-3
        inc_id = f"{self.id_prefix}-{next(self._ids):05d}"
        for a in g.alerts:
            self.assignment[a.id] = inc_id
        start = min(a.start for a in g.alerts)
        end = max(a.end for a in g.alerts)
        return Incident(
            id=inc_id, source=g.alerts[0].source, dataset=g.alerts[0].dataset,
            kind="drift" if m.drift else "anomaly",
            features=sorted({a.feature for a in g.alerts}), alert_ids=[a.id for a in g.alerts],
            start=start, end=end, duration_hours=m.duration_hours,
            peak_ratio=round(m.peak_ratio, 2), max_voters=m.max_voters,
            severity=sev, decision=dec, summary=_summary(g, m),                 # step 4
            reasons=g.reasons + [f"measures: {m.text()}", f"{dec.value} / {sev.value}: {why}"],
            alerts=g.alerts,
        )


def _summary(g: Group, m: Measures) -> str:
    """One sentence a human (or the Diagnosis agent) can read without opening the alerts."""
    a = g.alerts
    peak = max(a, key=lambda x: x.ratio)
    if g.family == "series":
        return (f"{peak.dataset}: {', '.join(sorted({x.feature for x in a}))} abnormal "
                f"from {min(x.start for x in a):%Y-%m-%d %H:%M} for {m.duration_hours:g} h. "
                f"Peak {peak.feature}={peak.value:g} (strength {peak.ratio:.1f}). {len(a)} alert(s) -> 1 incident.")
    if g.family == "logs":
        return (f"hdfs: {len(a)} block(s) failed with events {g.key}. "
                f"Strongest block {peak.details.get('block_id')} (strength {peak.ratio:.1f}).")
    return (f"ci: {g.key}: {len(a)} failed build(s) ({m.n_real} look real, {m.n_flaky} look flaky). "
            f"Last commit {a[-1].details.get('commit', '?')[:8]}.")