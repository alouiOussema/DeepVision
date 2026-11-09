"""Action agent (P5): approved ActionDecision -> executed, verified, reported (simulation by default).

    from agents.action import ActionAgent, make_action_node
See agents/action/README.md for the agent card.
"""
from .agent import ActionAgent
from .node import make_action_node

__all__ = ["ActionAgent", "make_action_node"]
