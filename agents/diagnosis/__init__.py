"""Diagnosis agent (P3): Incident -> Diagnosis (root cause, cited evidence, calibrated confidence).

    from agents.diagnosis import DiagnosisAgent, make_diagnosis_node
See agents/diagnosis/README.md for the agent card.
"""
from .agent import DiagnosisAgent
from .node import make_diagnosis_node

__all__ = ["DiagnosisAgent", "make_diagnosis_node"]
