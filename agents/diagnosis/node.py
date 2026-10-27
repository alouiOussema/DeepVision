"""LangGraph node for the orchestrator (P4).

    from agents.diagnosis.node import make_diagnosis_node
    graph.add_node("diagnosis", make_diagnosis_node())
    graph.add_edge("to_diagnosis", "diagnosis")

State in : "to_diagnosis"  incidents sent by Triage (pipeline.py), or "incidents" (agents/triage/node.py)
           "context"       optional ContextStore: change log, raw metrics (live mode: filled by P5's loaders)
State out: "diagnoses"     one Diagnosis per incident -> Decision (P4)
           "log"           one entry per diagnosis (declare it Annotated[list, operator.add])
"""
from __future__ import annotations

from typing import Any

from .agent import DiagnosisAgent
from .classifier import RootCauseClassifier
from .context import ContextStore
from .knowledge import KnowledgeBase
from .llm import LLMExplainer


def make_diagnosis_node(agent: DiagnosisAgent | None = None, llm_model: str | None = None):
    """llm_model=None -> no LLM (template explanations, fast); e.g. "llama3.1:8b" to explain with Ollama."""
    if agent is None:
        agent = DiagnosisAgent(kb=KnowledgeBase(), classifier=RootCauseClassifier.load(),
                               llm=LLMExplainer(llm_model) if llm_model else None)

    def diagnosis_node(state: dict[str, Any]) -> dict[str, Any]:
        incidents = state.get("to_diagnosis", state.get("incidents", []))
        diagnoses = agent.run(incidents, state.get("context") or ContextStore())
        log = [{"agent": "diagnosis", "incident": d.incident_id, "category": d.category.value,
                "sub_cause": d.sub_cause, "confidence": d.confidence, "needs_human": d.needs_human,
                "action_hint": d.suggested_action} for d in diagnoses]
        return {"diagnoses": diagnoses, "log": log or [{"agent": "diagnosis", "diagnosed": 0}]}

    return diagnosis_node
