"""Turns the graph's state updates into one event per agent and per incident (input, reason, action, result),
so any incident can be followed from its first alert to the verified fix.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from common.schemas import Decision

AGENTS = ("monitor", "triage", "diagnosis", "decision", "human", "action")


@dataclass
class TraceEvent:
    at: datetime | None
    agent: str
    incident_id: str
    input: str
    reason: str
    action: str
    result: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"at": self.at.isoformat() if self.at else None, "agent": self.agent, "incident_id": self.incident_id,
                "input": self.input, "reason": self.reason, "action": self.action, "result": self.result,
                "data": self.data}

    @classmethod
    def from_dict(cls, d: dict) -> "TraceEvent":
        return cls(datetime.fromisoformat(d["at"]) if d.get("at") else None, d["agent"], d["incident_id"],
                   d.get("input", ""), d.get("reason", ""), d.get("action", ""), d.get("result", ""), d.get("data", {}))


def _join(xs, n: int = 3) -> str:
    xs = [str(x) for x in xs]
    return ", ".join(xs[:n]) + (f" (+{len(xs) - n})" if len(xs) > n else "")


class Tracer:
    def __init__(self, episode: str = ""):
        self.episode = episode
        self.events: list[TraceEvent] = []
        self.alerts: dict[str, Any] = {}
        self.incidents: dict[str, Any] = {}
        self.sent: set[str] = set()

    def consume(self, updates: Iterable[dict]) -> dict:
        """Read every node's output; returns the merged final state (like pipeline.py's __main__)."""
        state: dict = {}
        for update in updates:
            for node, out in update.items():
                self.on_node(node, out or {})
                state.update(out or {})
        return state

    def on_node(self, node: str, out: dict) -> None:
        for a in out.get("alerts", []) or []:
            if hasattr(a, "id"):
                self.alerts[a.id] = a
        if node == "layer1_rules":
            for inc in out.get("incidents", []):
                self._monitor(inc)
                self._triage(inc)
        if node == "layer2_agent":
            for r in out.get("reviews", []) or []:
                iid = getattr(r, "incident_id", "")
                self.add(self.incidents[iid].start if iid in self.incidents else None, "triage", iid,
                         "an incident the rules were unsure about", getattr(r, "reason", "") or "LLM review",
                         f"LLM review: {getattr(r, 'rule_decision', '')} -> {getattr(r, 'decision', '')}", "")
        if node == "to_diagnosis":
            self.sent |= {i.id for i in out.get("to_diagnosis", [])}
        for d in out.get("diagnoses", []) or []:
            self._diagnosis(d)
        for d in out.get("decisions", []) or []:
            self._decision(d)
        for ap in out.get("approvals", []) or []:
            self.add(ap["at"], "human", ap["incident_id"], f"order {ap['decision_id']} waiting for approval",
                     ap.get("note", ""), ap["verdict"], f"status {ap['status']}", decision_id=ap["decision_id"])
        for r in out.get("actions_done", []) or []:
            self._action(r)

    def _monitor(self, inc) -> None:
        a = inc.alerts
        peak = max(a, key=lambda x: x.ratio)
        voters = sorted({v for x in a for v in x.voters})
        self.add(min(x.start for x in a), "monitor", inc.id,
                 f"{inc.dataset or inc.source} signals: {_join(sorted({x.feature for x in a}))}",
                 f"{_join(voters)} flagged it; peak {peak.ratio:.1f}x its limit ({peak.feature})",
                 f"{len(a)} alert(s) sent to Triage", _join([x.id for x in a]),
                 alerts=[x.id for x in a], first_alert=min(x.start for x in a).isoformat())

    def _triage(self, inc) -> None:
        self.incidents[inc.id] = inc
        sent = inc.decision is Decision.INCIDENT
        self.add(inc.start, "triage", inc.id, f"{len(inc.alerts)} alert(s)", " | ".join(inc.reasons[-2:]),
                 f"{inc.decision.value} ({inc.severity.value})",
                 "sent to Diagnosis" if sent else f"kept as {inc.decision.value}: nobody is paged",
                 summary=inc.summary, decision=inc.decision.value, severity=inc.severity.value)

    def _diagnosis(self, d) -> None:
        inc = self.incidents.get(d.incident_id)
        ev = " ".join(f"[{e.id}] {e.statement}" for e in d.evidence[:3])
        self.add(inc.start if inc else None, "diagnosis", d.incident_id, inc.summary if inc else "",
                 d.explanation + (f" Evidence: {ev}" if ev else ""),
                 f"{d.category.value}{' / ' + d.sub_cause if d.sub_cause else ''}"
                 f"{' suspect ' + d.suspect if d.suspect else ''}",
                 f"confidence {d.confidence:.2f}, suggests {d.suggested_action}"
                 f"{', needs a human' if d.needs_human else ''}",
                 category=d.category.value, sub_cause=d.sub_cause, suspect=d.suspect, confidence=d.confidence,
                 needs_human=d.needs_human, warnings=d.warnings)

    def _decision(self, d) -> None:
        self.add(d.created, "decision", d.incident_id, f"{d.category.value} at confidence {d.confidence:.2f}",
                 " -> ".join(d.reasons), f"{d.route.value}: {d.action.value}{' ' + d.target if d.target else ''}",
                 f"status {d.status.value} [{d.rule}]", decision_id=d.id, route=d.route.value,
                 order_action=d.action.value, status=d.status.value, target=d.target, rule=d.rule)

    def _action(self, r) -> None:
        self.add(r.started_at, "action", r.incident_id, f"order {r.order_id}: {r.action}", " | ".join(r.steps),
                 f"{r.action} ({r.attempts} attempt(s))",
                 f"{r.outcome}: {r.detail}" + (f"; told {', '.join(r.notified)}" if r.notified else ""),
                 order_id=r.order_id, outcome=r.outcome, attempts=r.attempts, notified=r.notified,
                 finished_at=r.finished_at.isoformat() if r.finished_at else None, simulated=r.simulated)

    def key(self, incident_id: str) -> str:
        """Triage numbers incidents from INC-00001 in every run: the episode name makes the key unique."""
        return f"{self.episode}:{incident_id}" if self.episode else incident_id

    def add(self, at, agent: str, incident_id: str, input: str, reason: str, action: str, result: str, **data) -> None:
        self.events.append(TraceEvent(at, agent, self.key(incident_id), input, reason, action, result,
                                      {"episode": self.episode, **data}))

    def timeline(self, incident_id: str) -> list[TraceEvent]:
        return timeline(self.events, self.key(incident_id))

    def save(self, path: Path, append: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a" if append else "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e.to_dict(), default=str) + "\n")


def timeline(events: list[TraceEvent], incident_id: str) -> list[TraceEvent]:
    """One incident's story, in time order (agents in pipeline order when two steps share a time)."""
    mine = [e for e in events if e.incident_id == incident_id]
    first = min((e.at for e in mine if e.at), default=None)
    return sorted(mine, key=lambda e: (e.at or first or datetime.min, AGENTS.index(e.agent)))


def load(path: Path) -> list[TraceEvent]:
    if not Path(path).exists():
        return []
    return [TraceEvent.from_dict(json.loads(l)) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
