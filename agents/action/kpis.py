"""KPIs of the Action agent (slide 11: "MTTR · % auto-resolved"), from the action inbox + the decision log.

    mttr_min            order created -> verified fix (status executed), median and mean, in minutes.
                        For a human-approved order this includes the time it waited on the approval page:
                        that is what the on-call engineer lives through.
    action_time_min     "action_started" -> report, for the same orders: the part that is P5's own speed
    auto_resolved_pct   incidents fixed with NO human at all (route act + executed), out of every order P5 ran
                        (escalations included: an escalation is a human's work, so it is not "auto-resolved")
    fix_rate_pct        rerun / rollback / retrain verified good, out of those P5 ran
    false_action_pct    rerun / rollback / retrain that FAILED or were REVERTED, out of those P5 ran
    simulated           True when the log says the runs were simulations (then the times are virtual)
"""
from __future__ import annotations

from datetime import datetime
from statistics import mean, median

from common.schemas import Action, ActionDecision, ActionStatus, Route

FIXES = {Action.RERUN, Action.ROLLBACK, Action.RETRAIN}
REPORTED = (ActionStatus.EXECUTED, ActionStatus.FAILED, ActionStatus.REVERTED)     # ordered: stable report


def _pct(n: int, total: int) -> float | None:
    return round(100 * n / total, 1) if total else None


def compute(orders: list[ActionDecision], events: list[dict]) -> dict:
    ran = [o for o in orders if o.status in REPORTED]
    fixes = [o for o in ran if o.action in FIXES]
    finished = {e["decision_id"]: datetime.fromisoformat(e["at"]) for e in events
                if e.get("event") in {s.value for s in REPORTED}}
    started = {e["decision_id"]: datetime.fromisoformat(e["at"]) for e in events if e.get("event") == "action_started"}
    good = [o for o in fixes if o.status is ActionStatus.EXECUTED and o.id in finished]
    mttr = [(finished[o.id] - o.created).total_seconds() / 60 for o in good]
    own = [(finished[o.id] - started[o.id]).total_seconds() / 60 for o in good if o.id in started]
    sim = [e.get("simulated") for e in events if e.get("agent") == "action" and "simulated" in e]
    per_action = {a.value: {s.value: sum(o.action is a and o.status is s for o in ran) for s in REPORTED}
                  for a in Action if a is not Action.NONE}
    return {
        "orders_run": len(ran),
        "mttr_min_median": round(median(mttr), 1) if mttr else None,
        "mttr_min_mean": round(mean(mttr), 1) if mttr else None,
        "action_time_min_median": round(median(own), 1) if own else None,
        "auto_resolved_pct": _pct(sum(o.route is Route.ACT and o.status is ActionStatus.EXECUTED for o in ran), len(ran)),
        "fix_rate_pct": _pct(len([o for o in fixes if o.status is ActionStatus.EXECUTED]), len(fixes)),
        "false_action_pct": _pct(len([o for o in fixes if o.status is not ActionStatus.EXECUTED]), len(fixes)),
        "escalations_sent": sum(o.action is Action.ESCALATE and o.status is ActionStatus.EXECUTED for o in ran),
        "simulated": bool(sim) and all(sim),
        "per_action": per_action,
    }


def report(k: dict) -> str:
    """The KPIs as a few lines a person can read."""
    def f(v, unit=""):
        return "-" if v is None else f"{v}{unit}"
    lines = [f"orders run by the Action agent : {k['orders_run']}" + ("  (SIMULATION: times are virtual)" if k["simulated"] else ""),
             f"MTTR (order -> verified fix)   : median {f(k['mttr_min_median'], ' min')}, mean {f(k['mttr_min_mean'], ' min')}",
             f"time spent by P5 itself        : median {f(k['action_time_min_median'], ' min')}",
             f"resolved with no human at all  : {f(k['auto_resolved_pct'], ' %')}",
             f"fixes that worked              : {f(k['fix_rate_pct'], ' %')}   false actions (failed / reverted): "
             f"{f(k['false_action_pct'], ' %')}",
             f"escalations delivered          : {k['escalations_sent']}"]
    for a, c in k["per_action"].items():
        if any(c.values()):
            lines.append(f"  {a:9s} " + "  ".join(f"{s} {n}" for s, n in c.items()))
    return "\n".join(lines)
