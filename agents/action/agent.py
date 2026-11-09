"""The Action agent (P5): approved ActionDecision -> done, verified, reported. No LLM, no new decision.

Decision (P4) already chose WHAT to do and WHETHER a human must approve it. P5 only does it, safely:

    0  gate       the inbox says "approved" right now, the order was never started (claim file), action is not none
    1  read       the order's constraints (max_retries, verify_after_min, ...): missing -> failed, never guessed
    2  precheck   what the executor needs (target, build id, human_before_deploy): missing -> failed + a human is told
    3  execute    once
    4  verify     after constraints["verify_after_min"] minutes (injected clock: tests and simulations never sleep)
    5  retry      at most max_retries times, only where a retry is safe (executors.py)
    6  undo       worse than before and revert_if_worse -> put it back -> "reverted"
    7  report     ActionInbox.report(id, executed | failed | reverted, detail): honest, failures count as false actions
    8  escalate   every failed / reverted action, and every model waiting for its deploy, is told to a person

Every step writes one line to the decision log (agent "action"): input, reason, action, result (slide 10).
"""
from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from common.schemas import Action, ActionDecision, ActionStatus
from agents.decision.inbox import ActionInbox, InboxError

from . import paths
from .backends import Backends, FileNotifier
from .clock import Clock, RealClock, SimClock
from .config import ActionConfig
from .executors import EXECUTORS, Check, Context, Executor, Result, Verdict, escalation_body


class SafetyError(RuntimeError):
    """The agent was configured in a way that could touch a real system without the explicit switch."""


@dataclass
class ActionRun:
    """What happened to one order (out of the node as state["actions_done"])."""
    order_id: str
    incident_id: str
    action: str
    outcome: str                  # executed | failed | reverted | skipped
    detail: str
    attempts: int = 0
    steps: list[str] = field(default_factory=list)       # the trail, in order, readable by a human
    notified: list[str] = field(default_factory=list)    # who was told
    ordered_at: datetime | None = None                   # when Decision created the order
    started_at: datetime | None = None
    finished_at: datetime | None = None
    simulated: bool = True
    data: dict = field(default_factory=dict)

    @property
    def mttr_min(self) -> float | None:
        """Order -> verified fix, in minutes (only for a fix that worked)."""
        if self.outcome != "executed" or not (self.ordered_at and self.finished_at):
            return None
        return round((self.finished_at - self.ordered_at).total_seconds() / 60, 1)

    def to_dict(self) -> dict:
        return {"order_id": self.order_id, "incident_id": self.incident_id, "action": self.action,
                "outcome": self.outcome, "detail": self.detail, "attempts": self.attempts, "steps": self.steps,
                "notified": self.notified, "simulated": self.simulated, "mttr_min": self.mttr_min,
                "ordered_at": self.ordered_at.isoformat() if self.ordered_at else None,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None}


class Claims:
    """One empty file per order ever started. Created with O_EXCL, so two Action processes (or a re-run of the
    pipeline) can never both execute the same order, even if the first one crashed before reporting."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def take(self, order_id: str, now: datetime) -> bool:
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.dir / f"{order_id}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(now.isoformat())
        return True

    def has(self, order_id: str) -> bool:
        return (self.dir / f"{order_id}.claim").exists()


class ActionAgent:
    def __init__(self, cfg: ActionConfig | None = None, backends: Backends | None = None,
                 inbox: ActionInbox | None = None, clock: Clock | None = None, claims_dir: Path = paths.CLAIMS_DIR,
                 notifications: Path = paths.NOTIFICATIONS, deploy_requests: Path = paths.DEPLOY_REQUESTS):
        self.cfg = cfg or ActionConfig()
        if backends is None:
            if not self.cfg.simulate:
                raise SafetyError("simulate=False but no real backends were given: this repo has no CI API, deploy "
                                  "API or model registry. Pass Backends(...) with real adapters explicitly.")
            backends = Backends()
        if backends.notifier is None:
            backends.notifier = FileNotifier(notifications)
        if self.cfg.simulate and backends.touching_production():
            raise SafetyError(f"simulation mode, but these backends touch production: {backends.touching_production()}"
                              " (set ActionConfig(simulate=False) to allow real execution)")
        self.backends = backends
        self.inbox = inbox or ActionInbox()
        self.log = self.inbox.log
        self.clock = clock or (SimClock() if self.cfg.simulate else RealClock())
        self.claims = Claims(claims_dir)
        self.deploy_requests = Path(deploy_requests)

    # ------------------------------------------------------------------ a batch
    def run(self, orders: list[ActionDecision] | None = None, job_refs: dict[str, dict] | None = None) -> list[ActionRun]:
        """orders: the graph's state["to_action"]; None = everything the inbox says is ready."""
        orders = self.inbox.ready() if orders is None else orders
        return [self.process(o, job_refs or {}) for o in orders]

    # ------------------------------------------------------------------ one order
    def process(self, order: ActionDecision, job_refs: dict[str, dict] | None = None) -> ActionRun:
        now = self.clock.now()
        run = ActionRun(order.id, order.incident_id, order.action.value, "skipped", "", ordered_at=order.created,
                        started_at=now, simulated=self.cfg.simulate)

        # 0. the gate: the inbox is the truth, not the copy we were handed
        try:
            current = self.inbox.get(order.id)
        except InboxError:
            return self._skip(run, order, f"order {order.id} is not in the inbox: it cannot be reported, so it is not run")
        if current.status is not ActionStatus.APPROVED:
            return self._skip(run, current, f"status is {current.status.value}: only 'approved' orders are executed")
        if current.action is Action.NONE:
            return self._skip(run, current, "action 'none' is never executed")
        if not self.claims.take(current.id, now):
            return self._skip(run, current, "this order was already started once (claim file exists): never executed "
                              "twice; if it never reported, a human must check what happened")
        order = current
        ex = EXECUTORS[order.action]
        ctx = Context(self.backends, self.clock, self.cfg, job_refs or {})
        self._step(run, order, "action_started", f"{order.action.value}{' ' + order.target if order.target else ''} "
                   f"on {order.incident_id} (approved by {order.decided_by}, rule {order.rule})"
                   f"{' [simulation]' if self.cfg.simulate else ''}")

        # 1-2. constraints and preconditions: what is missing is never guessed
        problem = self._constraints_problem(order) or ex.precheck(order, ctx)
        if problem:
            self._step(run, order, "action_refused", problem)
            return self._finish(run, order, "failed", f"not executed: {problem}", escalate=True)

        # 3-6. execute, verify, retry, undo
        retries = min(int(order.constraints["max_retries"]), self.cfg.max_retries_cap)
        try:
            outcome, detail = self._attempts(run, order, ex, ctx, retries)
        except Exception as e:                         # a bug or an unexpected backend error: report it, never hide it
            self._step(run, order, "action_error", f"unexpected error: {e!r}", trace=traceback.format_exc(limit=3))
            outcome, detail = "failed", f"unexpected error during {order.action.value}: {e!r}"
        return self._finish(run, order, outcome, detail, escalate=outcome != "executed")

    def _attempts(self, run: ActionRun, order: ActionDecision, ex: Executor, ctx: Context,
                  retries: int) -> tuple[str, str]:
        total = 1 + retries
        result: Result | None = None
        for attempt in range(1, total + 1):
            run.attempts = attempt
            tag = f"attempt {attempt}/{total}"
            result = ex.execute(order, ctx)
            self._step(run, order, "action_execute", f"{tag}: {result.detail}", attempt=attempt, ok=result.ok)
            run.data.update({k: v for k, v in result.data.items() if isinstance(v, (str, int, float))})
            if not result.ok:
                if result.retryable and attempt < total:
                    self._wait(run, order, self.cfg.retry_wait_min, "before retrying")
                    continue
                ex.on_fail(order, ctx, result)
                return "failed", f"{result.detail} ({tag}{', no retry left' if result.retryable else ''})"

            self._wait(run, order, ex.wait_before_verify(order), "before verifying")
            check: Check = ex.verify(order, ctx, result)
            self._step(run, order, "action_verify", f"{tag}: {check.verdict.value}: {check.detail}", attempt=attempt,
                       verdict=check.verdict.value, measured=check.data)
            run.data.update(check.data)
            if check.verdict is Verdict.FIXED:
                return "executed", f"{result.detail}; verified: {check.detail}"
            if check.verdict is Verdict.WORSE:
                return self._undo(run, order, ex, ctx, result, check)
            if check.verdict is Verdict.NOT_FIXED and ex.retry_when_not_fixed and attempt < total:
                self._wait(run, order, self.cfg.retry_wait_min, "before retrying")
                continue
            ex.on_fail(order, ctx, result)
            return "failed", f"{result.detail}; verified: {check.detail} ({tag})"
        ex.on_fail(order, ctx, result)
        return "failed", f"no attempt worked ({total} tried)"

    def _undo(self, run, order, ex: Executor, ctx, result: Result, check: Check) -> tuple[str, str]:
        if not (order.constraints.get("revert_if_worse") and result.changed):
            return "failed", f"{result.detail}; verified: {check.detail}; not undone (revert_if_worse is off)"
        back = ex.undo(order, ctx, result)
        self._step(run, order, "action_undo", back.detail, ok=back.ok)
        if not back.ok:
            return "failed", f"{check.detail}; UNDO FAILED: {back.detail}: the system may be worse than before"
        after = ctx.backends.deploy.health(ctx.clock.now()) if order.action is Action.ROLLBACK else None
        return "reverted", (f"{result.detail}; {check.detail}; {back.detail}"
                            + (f"; now error_rate {after.error_rate:.2%}, latency {after.latency_ms:.0f} ms" if after else ""))

    # ------------------------------------------------------------------ helpers
    def _constraints_problem(self, order: ActionDecision) -> str:
        c = order.constraints
        need = ["max_retries"] + (["verify_after_min"] if order.action is not Action.ESCALATE else []) \
            + (["revert_if_worse"] if order.action is Action.ROLLBACK else [])
        missing = [k for k in need if k not in c]
        if missing:
            return f"constraint(s) {missing} missing from the order: not guessing a value"
        if not isinstance(c["max_retries"], int) or c["max_retries"] < 0:
            return f"constraint max_retries={c['max_retries']!r} is not a count"
        return ""

    def _wait(self, run, order, minutes: float, why: str) -> None:
        if minutes > 0:
            self.clock.wait(minutes)
            self._step(run, order, "action_wait", f"waited {minutes:g} min {why}", log=False)

    def _step(self, run: ActionRun, order: ActionDecision, event: str, text: str, log: bool = True, **extra) -> None:
        run.steps.append(text)
        if log:
            self.log.event(event, order, self.clock.now(), agent="action", detail=text,
                           simulated=self.cfg.simulate, **extra)

    def _skip(self, run: ActionRun, order: ActionDecision, why: str) -> ActionRun:
        run.detail = why
        run.finished_at = self.clock.now()
        self._step(run, order, "action_skipped", why)
        return run

    def _finish(self, run: ActionRun, order: ActionDecision, outcome: str, detail: str, escalate: bool) -> ActionRun:
        if order.action is Action.RETRAIN and outcome == "executed":
            self._deploy_request(run, order)
        if order.action is Action.ESCALATE and outcome == "executed" and run.data.get("to"):
            run.notified.append(run.data["to"])
        if escalate and order.action is not Action.ESCALATE:
            self._tell_a_human(run, order, outcome, detail)
        elif escalate:
            self._step(run, order, "action_escalation_failed",
                       f"the escalation itself could not be delivered: {detail}. Nobody was told: check the channel")
        run.outcome, run.detail = outcome, detail
        run.finished_at = self.clock.now()
        self.inbox.report(order.id, outcome, detail + (" [simulation]" if self.cfg.simulate else ""),
                          now=run.finished_at)
        return run

    def _tell_a_human(self, run: ActionRun, order: ActionDecision, outcome: str, detail: str) -> None:
        to = self.cfg.escalate_to.get(order.action.value, self.cfg.escalate_to["escalate"])
        subject = {Action.RERUN: "the re-run failed again: treat it as a real bug",
                   Action.ROLLBACK: "the rollback did not fix it" if outcome == "failed" else "the rollback made it worse, undone",
                   Action.RETRAIN: "the retrained model is not better: champion kept, drift report"}[order.action]
        if detail.startswith("not executed"):
            subject = f"{order.action.value} could not be done safely"
        elif order.action is Action.RETRAIN and "comparison" not in run.data:
            subject = "retraining could not run"
        elif order.action is Action.RERUN and "run_id" not in run.data:
            subject = "the re-run could not be started"
        meta = {"order": order.id, "incident": order.incident_id, "outcome": outcome}
        if run.data.get("commit"):
            meta["commit"] = run.data["commit"]           # "author" = the author of this commit
        mid = self.backends.notifier.send(to, f"[{order.category.value}] {order.incident_id}: {subject}",
                                          escalation_body(order, run.steps), self.clock.now(), meta)
        run.notified.append(to)
        self._step(run, order, "action_escalated", f"told {to}: {subject} ({mid})", to=to)

    def _deploy_request(self, run: ActionRun, order: ActionDecision) -> None:
        """The new model is NOT deployed. We only record that a human must decide (open question with P4: a follow-up
        order in the action inbox, or P5's own gate). Here: a line in deploy_requests.jsonl + a message."""
        cmp = run.data.get("comparison", {})
        row = {"at": self.clock.now().isoformat(timespec="seconds"), "order": order.id, "incident": order.incident_id,
               "challenger_id": cmp.get("challenger_id"), "comparison": cmp, "status": "awaiting_human",
               "simulated": self.cfg.simulate}
        self.deploy_requests.parent.mkdir(parents=True, exist_ok=True)
        with open(self.deploy_requests, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        to = self.cfg.escalate_to["retrain"]
        mid = self.backends.notifier.send(
            to, f"[data_drift] {order.incident_id}: a better model is ready, approve its deploy?",
            escalation_body(order, run.steps), self.clock.now(), {"order": order.id, "incident": order.incident_id,
                                       "challenger": cmp.get("challenger_id")})
        run.notified.append(to)
        self._step(run, order, "action_deploy_request", f"challenger {cmp.get('challenger_id')} waits for {to} "
                   f"to approve its deploy ({mid}); the live model was not changed", to=to)
3
