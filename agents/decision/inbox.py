"""The action inbox: the hand-off between Decision (P4), the human approval page, and Action (P5).

    P4   inbox.add(decisions)                         every ActionDecision, with its status
    UI   inbox.pending()  /  inbox.resolve(...)       a human approves or rejects a pending one
    P5   inbox.ready()                                what may be executed NOW (status "approved")
         inbox.report(id, "executed"|"failed"|"reverted", detail)    after acting and verifying

One JSON file (data/decision/action_inbox.json). Writes are atomic (temp file + replace) and take a small
lock file, so the pipeline, the approval page and P5 can all use it at the same time.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from common.schemas import ActionDecision, ActionStatus

from . import paths
from .log import DecisionLog, utcnow

DONE = {ActionStatus.EXECUTED, ActionStatus.FAILED, ActionStatus.REVERTED}


class InboxError(RuntimeError):
    pass


class ActionInbox:
    def __init__(self, path: Path = paths.ACTION_INBOX, log: DecisionLog | None = None):
        self.path = Path(path)
        self.log = log or DecisionLog()

    # ------------------------------------------------------------------ file access
    @contextmanager
    def _lock(self, timeout: float = 5.0):
        lock = self.path.with_suffix(".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        end = time.time() + timeout
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if time.time() > end:                       # a crashed writer left the lock: take it over
                    lock.unlink(missing_ok=True)
                    continue
                time.sleep(0.02)
        try:
            yield
        finally:
            lock.unlink(missing_ok=True)

    def all(self) -> list[ActionDecision]:
        if not self.path.exists():
            return []
        return [ActionDecision.from_dict(d) for d in json.loads(self.path.read_text(encoding="utf-8"))]

    def _write(self, items: list[ActionDecision]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([d.to_dict() for d in items], indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, decision_id: str) -> ActionDecision:
        for d in self.all():
            if d.id == decision_id:
                return d
        raise InboxError(f"no decision {decision_id!r} in the inbox")

    # ------------------------------------------------------------------ P4: add
    def add(self, decisions: list[ActionDecision]) -> None:
        """Insert or replace by id. Keeps everything already there (history of orders)."""
        with self._lock():
            by_id = {d.id: d for d in self.all()}
            by_id.update({d.id: d for d in decisions})
            self._write(list(by_id.values()))

    # ------------------------------------------------------------------ views
    def pending(self) -> list[ActionDecision]:
        """Waiting for a human."""
        return [d for d in self.all() if d.status is ActionStatus.PENDING_APPROVAL]

    def ready(self) -> list[ActionDecision]:
        """P5: what may be executed now (decided by the policy or approved by a human, not yet done)."""
        return [d for d in self.all() if d.status is ActionStatus.APPROVED]

    # ------------------------------------------------------------------ UI: the human gate
    def resolve(self, decision_id: str, verdict: str, reviewer: str = "operator", note: str = "",
                now: datetime | None = None) -> ActionDecision:
        if verdict not in {"approve", "reject"}:
            raise ValueError("verdict must be 'approve' or 'reject'")
        now = now or utcnow()
        with self._lock():
            items = self.all()
            d = next((x for x in items if x.id == decision_id), None)
            if d is None:
                raise InboxError(f"no decision {decision_id!r} in the inbox")
            if d.status is not ActionStatus.PENDING_APPROVAL:
                raise InboxError(f"{decision_id} is {d.status.value}: only a pending decision can be resolved")
            d.status = ActionStatus.APPROVED if verdict == "approve" else ActionStatus.REJECTED
            d.decided_by, d.note, d.decided_at = reviewer, note, now
            d.reasons.append(f"{reviewer} {'approved' if verdict == 'approve' else 'rejected'} it"
                             + (f": {note}" if note else ""))
            self._write(items)
        self.log.event("approved" if verdict == "approve" else "rejected", d, now, note=note)
        if verdict == "approve":
            self.log.grant(d, now)
        return d

    # ------------------------------------------------------------------ P5: report the result
    def report(self, decision_id: str, result: str, detail: str = "", now: datetime | None = None) -> ActionDecision:
        """P5 calls this after acting AND verifying. result: executed | failed | reverted."""
        try:
            status = ActionStatus(result)
        except ValueError:
            raise ValueError("result must be 'executed', 'failed' or 'reverted'") from None
        if status not in DONE:
            raise ValueError("result must be 'executed', 'failed' or 'reverted'")
        now = now or utcnow()
        with self._lock():
            items = self.all()
            d = next((x for x in items if x.id == decision_id), None)
            if d is None:
                raise InboxError(f"no decision {decision_id!r} in the inbox")
            if d.status is not ActionStatus.APPROVED:
                raise InboxError(f"{decision_id} is {d.status.value}: only an approved decision can be executed")
            d.status, d.result = status, detail
            self._write(items)
        self.log.event(status.value, d, now, result=detail)
        return d
