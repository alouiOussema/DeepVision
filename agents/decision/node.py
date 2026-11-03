"""LangGraph node for the orchestrator.

    from agents.decision.node import make_decision_node, route_after_decision
    graph.add_node("decision", make_decision_node())
    graph.add_edge("diagnosis", "decision")
    graph.add_conditional_edges("decision", route_after_decision, {"action": "action", "end": END})   # P5 plugs in

State in : "diagnoses"        one Diagnosis per incident (Diagnosis node)
           "to_diagnosis"     optional: the incidents Diagnosis received (gives the source / project to the guards)
           "incidents"        optional: same, when Triage's own node is used
State out: "decisions"        one ActionDecision per diagnosis
           "to_action"        the decisions P5 may execute now (status "approved": act alone, or escalate)
           "pending_approval" the decisions waiting for the approval page (ui/decision_app.py)
           "log"              one entry per decision (declare it Annotated[list, operator.add])
Side effects: every order is written to the action inbox (P5 reads it there too) and to the decision log.
"""
from __future__ import annotations

from typing import Any

from common.schemas import ActionStatus

from .agent import DecisionAgent
from .inbox import ActionInbox
from .log import DecisionLog

OPEN = (ActionStatus.PENDING_APPROVAL, ActionStatus.APPROVED)       # orders not finished yet


def make_decision_node(agent: DecisionAgent | None = None, inbox: ActionInbox | None = None):
    """Defaults: policy from the knowledge base, decision log + history in logs/ and data/decision/."""
    log = DecisionLog()
    agent = agent or DecisionAgent(log=log)
    inbox = inbox or ActionInbox(log=agent.log or log)

    def decision_node(state: dict[str, Any]) -> dict[str, Any]:
        diagnoses = state.get("diagnoses", [])
        incidents = state.get("to_diagnosis") or state.get("incidents") or []
        # orders still open from earlier runs: the same action on the same thing is not asked twice
        open_orders = {(d.snapshot.get("action_key") or d.incident_id, d.action): d.id
                       for d in inbox.all() if d.status in OPEN}
        decisions = agent.run(diagnoses, incidents, open_orders=open_orders)
        inbox.add(decisions)
        log_rows = [{"agent": "decision", "incident": d.incident_id, "decision": d.id, "category": d.category.value,
                     "confidence": d.confidence, "action": d.action.value, "route": d.route.value,
                     "status": d.status.value, "rule": d.rule, "target": d.target, "why": d.reasons}
                    for d in decisions]
        return {"decisions": decisions,
                "to_action": [d for d in decisions if d.status is ActionStatus.APPROVED],
                "pending_approval": [d for d in decisions if d.status is ActionStatus.PENDING_APPROVAL],
                "log": log_rows or [{"agent": "decision", "decided": 0}]}

    return decision_node


def route_after_decision(state: dict[str, Any]) -> str:
    """Conditional edge: "action" when there is something P5 may execute now, else "end"."""
    return "action" if state.get("to_action") else "end"
