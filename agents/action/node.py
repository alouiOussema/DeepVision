"""LangGraph node for the orchestrator.

    from agents.action.node import make_action_node
    graph = build_graph(..., action_node=make_action_node())        # pipeline.py: decision -> action -> END

State in : "to_action"      the orders Decision approved in this run (status "approved"). Absent -> the whole
                            inbox is read (ActionInbox().ready()): the node also works on its own, e.g. after a
                            human approved orders on ui/decision_app.py
           "to_diagnosis"   optional: the incidents, to find the failed CI build behind a rerun (Alert.details.build_id)
           "context"        optional, simulation only: P3's ContextStore; its change log tells the simulated deploy
                            backend which releases / config pushes are live
State out: "actions_done"   one ActionRun per order (executed | failed | reverted | skipped)
           "log"            one entry per order (declare it Annotated[list, operator.add])
Side effects: ActionInbox.report(...) for every order run, the decision log, data/action/ (claims, notifications).
"""
from __future__ import annotations

from typing import Any

from common.schemas import Action

from .agent import ActionAgent
from .backends import SimDeploy
from .executors import job_refs_from_incidents


def _feed_simulated_world(agent: ActionAgent, state: dict[str, Any], orders) -> None:
    """Simulation only: make the simulated deploy backend know the changes of the pipeline's change log.
    Assumption (stated in the README): in the simulation the release Diagnosis blamed is the one that hurts,
    so rolling it back fixes it. Real backends read the real state instead."""
    dep = agent.backends.deploy
    ctx = state.get("context")
    if not (agent.cfg.simulate and isinstance(dep, SimDeploy) and not dep.live and ctx is not None):
        return
    dep.live = [c.id for c in getattr(ctx, "changes", [])]
    for o in orders or []:
        if o.action is Action.ROLLBACK and o.target in dep.live:
            dep.effects.setdefault(o.target, (0.05, 60.0))


def make_action_node(agent: ActionAgent | None = None):
    """Defaults: simulation (no real side effect), the real action inbox and decision log."""
    agent = agent or ActionAgent()

    def action_node(state: dict[str, Any]) -> dict[str, Any]:
        orders = state.get("to_action")
        orders = agent.inbox.ready() if orders is None else orders
        incidents = state.get("to_diagnosis") or state.get("incidents") or []
        _feed_simulated_world(agent, state, orders)
        runs = agent.run(orders, job_refs_from_incidents(incidents))
        rows = [{"agent": "action", "incident": r.incident_id, "order": r.order_id, "action": r.action,
                 "outcome": r.outcome, "attempts": r.attempts, "notified": r.notified, "detail": r.detail,
                 "simulated": r.simulated} for r in runs]
        return {"actions_done": runs, "log": rows or [{"agent": "action", "executed": 0}]}

    return action_node
