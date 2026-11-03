"""KPIs of the Decision agent, computed from the orders in the action inbox (slide 10: KPIs).

    act_alone_pct       orders P5 may run without a human (route act)
    human_load_pct      orders that reached a person: approval page + escalations
    auto_resolved_pct   routed act AND verified good by P5 (status executed), out of all decisions
    false_action_rate   executed actions (rerun / rollback / retrain) that FAILED or were REVERTED, out of those P5 reported
    override_rate       what humans rejected, out of what they resolved on the approval page

`suggest_thresholds` only REPORTS where the logged mistakes say a bar looks wrong ("thresholds re-tuned from
logged mistakes", slide 16). Nothing changes by itself: the bars live in knowledge_base/action_risk_table.md and a
person edits them.
"""
from __future__ import annotations

from collections import defaultdict

from common.schemas import Action, ActionDecision, ActionStatus, Route

EXECUTABLE = {Action.RERUN, Action.ROLLBACK, Action.RETRAIN}
BAD = {ActionStatus.FAILED, ActionStatus.REVERTED}
DONE = {ActionStatus.EXECUTED} | BAD


def _pct(n: int, total: int) -> float | None:
    return round(100 * n / total, 1) if total else None


def compute(items: list[ActionDecision]) -> dict:
    n = len(items)
    by_route = {r: [d for d in items if d.route is r] for r in Route}
    acted = [d for d in items if d.action in EXECUTABLE and d.status in DONE]
    humans = [d for d in items if d.status in (ActionStatus.REJECTED,) or (d.decided_by != "policy" and d.status in
              ({ActionStatus.APPROVED} | DONE))]
    rejected = [d for d in humans if d.status is ActionStatus.REJECTED]
    per_cause: dict = defaultdict(lambda: {"decisions": 0, "act": 0, "approval": 0, "escalate": 0, "none": 0, "bad": 0})
    for d in items:
        c = per_cause[d.category.value]
        c["decisions"] += 1
        c[d.route.value] += 1
        c["bad"] += d.action in EXECUTABLE and d.status in BAD
    return {
        "decisions": n,
        "act_alone_pct": _pct(len(by_route[Route.ACT]), n),
        "human_load_pct": _pct(len(by_route[Route.APPROVAL]) + len(by_route[Route.ESCALATE]), n),
        "auto_resolved_pct": _pct(sum(d.status is ActionStatus.EXECUTED and d.route is Route.ACT for d in items), n),
        "waiting_for_human": sum(d.status is ActionStatus.PENDING_APPROVAL for d in items),
        "reported_by_p5": len(acted),
        "false_action_rate": _pct(sum(d.status in BAD for d in acted), len(acted)),
        "override_rate": _pct(len(rejected), len(humans)),
        "per_cause": dict(per_cause),
    }


def suggest_thresholds(items: list[ActionDecision], bars: dict[str, float | None], min_cases: int = 3) -> list[str]:
    """Plain-language hints, per cause, from the mistakes seen so far. Report only."""
    out = []
    for cause, bar in bars.items():
        if bar is None:
            continue
        mine = [d for d in items if d.category.value == cause and d.action in EXECUTABLE]
        bad = [d for d in mine if d.status in BAD]
        rejected_above = [d for d in mine if d.status is ActionStatus.REJECTED and d.confidence >= bar]
        if bad and len([d for d in mine if d.status in DONE]) >= min_cases:
            top = max(d.confidence for d in bad)
            out.append(f"{cause}: {len(bad)} executed action(s) failed or were reverted, the worst at confidence "
                       f"{top:.2f} (bar {bar:.2f}): consider raising the bar above {top:.2f}")
        if len(rejected_above) >= min_cases:
            out.append(f"{cause}: humans rejected {len(rejected_above)} order(s) that were above the bar {bar:.2f}: "
                       "the diagnosis is over-confident for this cause")
    return out
