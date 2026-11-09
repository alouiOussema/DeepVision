"""The decision log (traceability) and the history the guards read.

    logs/decision_log.jsonl          EVERYTHING, one line per event: decided | approved | rejected | executed | failed ...
                                     with the data used, the reasons, the action and the result (the deck's promise:
                                     "every step is recorded: data used, reason, action, result")
    data/decision/decision_history.jsonl   only actions that were GRANTED (by policy or by a human):
                                     what the guards need (rerun budget, cooldown, auto-actions per hour)
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common.schemas import ActionDecision

from . import paths


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _read(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def action_key(d: ActionDecision) -> str:
    """What an action is ABOUT, so two orders on the same thing can be recognised (set by the agent:
    the release to undo, else the CI project, else the signals)."""
    return d.snapshot.get("action_key") or d.target or d.incident_id


class DecisionLog:
    def __init__(self, log: Path = paths.DECISION_LOG, history: Path = paths.DECISION_HISTORY):
        self.log_path, self.history_path = Path(log), Path(history)

    # ------------------------------------------------------------------ write
    def event(self, event: str, d: ActionDecision, now: datetime | None = None, **extra) -> None:
        """One line in the decision log."""
        _append(self.log_path, {
            "at": (now or utcnow()).isoformat(timespec="seconds"), "event": event,
            "decision_id": d.id, "incident_id": d.incident_id, "diagnosis_id": d.diagnosis_id,
            "category": d.category.value, "confidence": d.confidence, "risk": d.risk,
            "action": d.action.value, "route": d.route.value, "status": d.status.value,
            "target": d.target, "rule": d.rule, "decided_by": d.decided_by, "reasons": d.reasons,
            "data_used": {"evidence": [e.get("id") for e in d.snapshot.get("evidence", [])],
                          "warnings": d.snapshot.get("warnings", [])},
            **extra})

    def grant(self, d: ActionDecision, now: datetime | None = None) -> None:
        """Remember that this action was allowed to run (the guards count these)."""
        _append(self.history_path, {
            "at": (now or utcnow()).isoformat(timespec="seconds"), "decision_id": d.id,
            "incident_id": d.incident_id, "key": action_key(d), "action": d.action.value,
            "auto": d.decided_by == "policy"})

    # ------------------------------------------------------------------ read
    def events(self) -> list[dict]:
        return _read(self.log_path)

    def grants(self, within_h: float | None = None, now: datetime | None = None) -> list[dict]:
        rows = _read(self.history_path)
        if within_h is None:
            return rows
        since = (now or utcnow()) - timedelta(hours=within_h)
        return [r for r in rows if datetime.fromisoformat(r["at"]) >= since]
