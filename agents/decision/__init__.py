"""Decision agent (P4): Diagnosis -> ActionDecision (act alone, ask a human, or escalate).

    from agents.decision import DecisionAgent, make_decision_node
See agents/decision/README.md for the agent card and the hand-off contract for the Action agent (P5).
"""
from .agent import DecisionAgent
from .node import make_decision_node

__all__ = ["DecisionAgent", "make_decision_node"]
