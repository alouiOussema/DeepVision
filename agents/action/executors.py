"""One small class per action, behind the same interface: precheck -> execute -> verify -> (undo).

    precheck(order)          what is missing to do it SAFELY (no target, no job id...): "" = ok. Never guessed.
    execute(order)  -> Result   do it once. A BackendError (API down) means it did NOT happen: retrying is safe.
    verify(order, result) -> Check   FIXED | NOT_FIXED | WORSE | UNKNOWN  (UNKNOWN is never read as fixed)
    undo(order, result) -> Result    only when execute changed something that can be put back (rollback)

When to retry is the executor's call, because a blind retry can be harmful:
    rerun     a red re-run is NOT retried: the rerun budget is 1 (flaky_test runbook), red twice = a real bug
    rollback  not retried once done: we are already on the previous version, doing it again changes nothing
    retrain   not retried when the challenger loses: same data, same answer. Retried if training crashed
    escalate  re-sent if the message was not delivered
Every executor retries when the outside system did not answer (BackendError), up to the order's max_retries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from common.schemas import Action, ActionDecision

from .backends import Backends, BackendError, BackendRefused, Health
from .clock import Clock
from .config import ActionConfig


class Verdict(str, Enum):
    FIXED = "fixed"           # the problem is gone
    NOT_FIXED = "not_fixed"   # the action ran, the problem is still there
    WORSE = "worse"           # the action made things worse than before
    UNKNOWN = "unknown"       # could not be measured: treated as NOT fixed, never assumed fine


@dataclass
class Result:
    ok: bool                       # the action really happened
    detail: str
    retryable: bool = False        # ok=False and nothing changed: safe to try again
    changed: bool = False          # something in the system changed and could be put back (undo)
    data: dict = field(default_factory=dict)


@dataclass
class Check:
    verdict: Verdict
    detail: str
    data: dict = field(default_factory=dict)


@dataclass
class Context:
    """What an executor may use. job_refs: incident id -> {"project", "build_id", "commit"} of the failed CI build
    (read from the incident's alerts in the graph; the order itself does not carry the build id)."""
    backends: Backends
    clock: Clock
    cfg: ActionConfig
    job_refs: dict[str, dict] = field(default_factory=dict)


class Executor:
    action: Action
    retry_when_not_fixed = False     # see the module docstring

    def precheck(self, order: ActionDecision, ctx: Context) -> str:
        return ""

    def wait_before_verify(self, order: ActionDecision) -> float:
        return float(order.constraints["verify_after_min"])

    def execute(self, order: ActionDecision, ctx: Context) -> Result:
        raise NotImplementedError

    def verify(self, order: ActionDecision, ctx: Context, result: Result) -> Check:
        raise NotImplementedError

    def undo(self, order: ActionDecision, ctx: Context, result: Result) -> Result:
        return Result(False, f"{self.action.value} cannot be undone")

    def on_fail(self, order: ActionDecision, ctx: Context, result: Result | None) -> None:
        """Clean up after a failed attempt (e.g. throw away a losing challenger). Nothing by default."""


# ------------------------------------------------------------------ rerun (flaky test, deck scenario 1)
def ci_project(order: ActionDecision) -> str:
    key = order.snapshot.get("action_key", "")
    return key[3:] if key.startswith("ci:") else ""


def job_ref(order: ActionDecision, ctx: Context) -> dict | None:
    """The failed build to re-run. The order carries it if Decision ever adds it to the snapshot ("job");
    otherwise it is read from the incident's alerts (ctx.job_refs). Nothing else: no guessing."""
    ref = order.snapshot.get("job") or ctx.job_refs.get(order.incident_id)
    return ref if ref and ref.get("build_id") not in (None, "") else None


class RerunExecutor(Executor):
    action = Action.RERUN

    def precheck(self, order, ctx) -> str:
        if not ci_project(order):
            return (f"no CI project in the order (snapshot action_key={order.snapshot.get('action_key')!r}): "
                    "cannot tell which pipeline to re-run")
        ref = job_ref(order, ctx)
        if ref is None:
            return (f"no build id for incident {order.incident_id}: the order does not carry it and no CI alert of this "
                    "incident was given to the Action agent; re-running 'some' build of the project would be a guess")
        if ref.get("project") and ref["project"] != ci_project(order):
            return f"build {ref['build_id']} belongs to {ref['project']}, the order is about {ci_project(order)}"
        reruns = order.constraints.get("reruns")
        if not isinstance(reruns, int) or reruns < 1:
            return f"constraint 'reruns' is {reruns!r}: no re-run is allowed"
        return ""

    def execute(self, order, ctx) -> Result:
        ref, project = job_ref(order, ctx), ci_project(order)
        try:
            run_id = ctx.backends.ci.rerun(project, str(ref["build_id"]), ctx.clock.now())
        except BackendError as e:
            return Result(False, f"re-run of build {ref['build_id']} not started: {e}", retryable=True)
        return Result(True, f"re-ran build {ref['build_id']} of {project} (commit {ref.get('commit') or '?'}) as {run_id}",
                      data={"run_id": run_id, "build_id": str(ref["build_id"]), "project": project,
                            "commit": ref.get("commit", "")})

    def verify(self, order, ctx, result) -> Check:
        outcome = ctx.backends.ci.result(result.data["run_id"], ctx.clock.now())
        if outcome == "passed":
            return Check(Verdict.FIXED, f"{result.data['run_id']} passed: build green, the failure was flaky "
                         f"(test tagged flaky, nobody paged)", {"ci": outcome})
        if outcome == "failed":
            return Check(Verdict.NOT_FIXED, f"{result.data['run_id']} failed again on the same code: a real bug, "
                         "not a flaky test (rerun budget used, not re-run again)", {"ci": outcome})
        return Check(Verdict.UNKNOWN, f"{result.data['run_id']} is {outcome!r}: no verdict, not counted as fixed",
                     {"ci": outcome})


# ------------------------------------------------------------------ rollback (bad deploy / config, deck scenario 2)
class RollbackExecutor(Executor):
    action = Action.ROLLBACK

    def precheck(self, order, ctx) -> str:
        if not order.target:
            return "rollback order without a target: P5 never picks the release to undo itself"
        if not ctx.backends.deploy.is_live(order.target):
            return (f"{order.target} is not live any more (already rolled back, or never deployed here): "
                    "refusing to roll back something else")
        return ""

    def execute(self, order, ctx) -> Result:
        dep, now = ctx.backends.deploy, ctx.clock.now()
        before = dep.health(now)
        try:
            dep.rollback(order.target, now)
        except BackendError as e:
            return Result(False, f"rollback of {order.target} not done: {e}", retryable=True)
        except BackendRefused as e:
            return Result(False, f"rollback of {order.target} refused: {e}")
        return Result(True, f"rolled back {order.target}; before: error_rate {before.error_rate:.2%}, "
                      f"latency {before.latency_ms:.0f} ms", changed=True, data={"before": before})

    def verify(self, order, ctx, result) -> Check:
        dep = ctx.backends.deploy
        before: Health = result.data["before"]
        after = dep.health(ctx.clock.now())
        m = ctx.cfg.worse_margin
        data = {"before": before.as_dict(), "after": after.as_dict(),
                "band": {"error_rate_max": dep.band.error_rate_max, "latency_ms_max": dep.band.latency_ms_max}}
        now_txt = f"error_rate {after.error_rate:.2%}, latency {after.latency_ms:.0f} ms"
        if dep.band.contains(after):
            return Check(Verdict.FIXED, f"{now_txt}: back in the normal band (<= {dep.band.error_rate_max:.0%}, "
                         f"<= {dep.band.latency_ms_max:.0f} ms)", data)
        if after.error_rate > before.error_rate * (1 + m) or after.latency_ms > before.latency_ms * (1 + m):
            return Check(Verdict.WORSE, f"{now_txt}: worse than before the rollback "
                         f"({before.error_rate:.2%}, {before.latency_ms:.0f} ms)", data)
        return Check(Verdict.NOT_FIXED, f"{now_txt}: still out of the band after the rollback: the release was "
                     "probably not the cause (infrastructure?)", data)

    def undo(self, order, ctx, result) -> Result:
        try:
            ctx.backends.deploy.redeploy(order.target, ctx.clock.now())
        except (BackendError, BackendRefused) as e:
            return Result(False, f"could not redeploy {order.target}: {e}")
        return Result(True, f"redeployed {order.target} (undo of the rollback)")


# ------------------------------------------------------------------ retrain (data drift, deck scenario 3)
class RetrainExecutor(Executor):
    action = Action.RETRAIN

    def precheck(self, order, ctx) -> str:
        if order.constraints.get("human_before_deploy") is not True:
            return ("retrain order without human_before_deploy=True: P5 never deploys a model alone, "
                    "so it refuses an order that would let it")
        return ""

    def execute(self, order, ctx) -> Result:
        try:
            cmp = ctx.backends.model.retrain_and_compare(order.id, ctx.clock.now(), ctx.cfg.retrain_window_days,
                                                          ctx.cfg.retrain_holdout_days)
        except BackendError as e:
            return Result(False, f"retraining failed: {e}", retryable=True)
        except BackendRefused as e:
            return Result(False, f"retraining impossible: {e}")
        return Result(True, f"trained {cmp.challenger_id} on the last {ctx.cfg.retrain_window_days} days; "
                      f"live model untouched", data={"comparison": cmp})

    def verify(self, order, ctx, result) -> Check:
        cmp = result.data["comparison"]
        txt = (f"{cmp.metric}: champion {cmp.champion:.3f} vs challenger {cmp.challenger:.3f} "
               f"({cmp.gain:+.1%}) on {cmp.n_holdout} recent rows")
        if cmp.gain >= ctx.cfg.retrain_min_gain:
            return Check(Verdict.FIXED, f"{txt}: the challenger is better; NOT deployed, a human approves the deploy",
                         {"comparison": cmp.as_dict()})
        return Check(Verdict.NOT_FIXED, f"{txt}: the challenger is not better by {ctx.cfg.retrain_min_gain:.0%}; "
                     "the champion stays live (the drift may be seasonal)", {"comparison": cmp.as_dict()})

    def on_fail(self, order, ctx, result) -> None:
        cmp = (result.data.get("comparison") if result else None)
        if cmp is not None:
            ctx.backends.model.discard(cmp.challenger_id)


# ------------------------------------------------------------------ escalate (deck scenario 4 + every failure)
class EscalateExecutor(Executor):
    action = Action.ESCALATE
    retry_when_not_fixed = True       # a message that did not arrive is simply sent again

    def wait_before_verify(self, order) -> float:
        return 0.0                    # a notification is checked at once, not 15 minutes later

    def recipient(self, order, ctx) -> tuple[str, str]:
        to = order.constraints.get("notify")
        if to:
            return to, "constraints['notify']"
        return ctx.cfg.escalate_to["escalate"], "ActionConfig.escalate_to['escalate'] (the order names nobody)"

    def execute(self, order, ctx) -> Result:
        if ctx.backends.notifier is None:
            return Result(False, "no notification channel configured")
        to, why = self.recipient(order, ctx)
        body = escalation_body(order)
        try:
            mid = ctx.backends.notifier.send(to, f"[{order.category.value}] {order.incident_id}: a human takes over",
                                             body, ctx.clock.now(), {"order": order.id, "incident": order.incident_id})
        except BackendError as e:
            return Result(False, f"notification to {to} not sent: {e}", retryable=True)
        return Result(True, f"notified {to} (recipient from {why}); nothing changed in the system",
                      data={"message_id": mid, "to": to})

    def verify(self, order, ctx, result) -> Check:
        if ctx.backends.notifier.delivered(result.data["message_id"]):
            return Check(Verdict.FIXED, f"message {result.data['message_id']} delivered to {result.data['to']}")
        return Check(Verdict.NOT_FIXED, f"message {result.data['message_id']} not delivered")


def escalation_body(order: ActionDecision, extra: list[str] | None = None) -> str:
    lines = [order.summary or f"{order.category.value} on {order.incident_id}", "", "Decision trail:"]
    lines += [f"- {r}" for r in order.reasons]
    if extra:
        lines += ["", "What the Action agent did:"] + [f"- {s}" for s in extra]
    return "\n".join(lines)


EXECUTORS: dict[Action, Executor] = {e.action: e for e in
                                     (RerunExecutor(), RollbackExecutor(), RetrainExecutor(), EscalateExecutor())}


def job_refs_from_incidents(incidents: list[Any]) -> dict[str, dict]:
    """incident id -> the failed CI build to re-run (the most recent CI alert that names a build_id).
    P1's CI monitor puts project / build_id / commit in Alert.details."""
    out: dict[str, dict] = {}
    for inc in incidents or []:
        builds = [a for a in getattr(inc, "alerts", []) if a.source == "ci" and a.details.get("build_id") is not None]
        if builds:
            a = max(builds, key=lambda x: x.timestamp)
            out[inc.id] = {"project": str(a.details.get("project", "")), "build_id": str(a.details["build_id"]),
                           "commit": str(a.details.get("commit", "")), "alert_id": a.id}
    return out
