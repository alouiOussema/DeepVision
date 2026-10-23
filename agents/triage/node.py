"""LangGraph node for the orchestrator (P4).

    from agents.triage.node import triage_node
    graph.add_node("triage", triage_node)
    graph.add_edge("monitor", "triage")

State in : "alerts"    list of dicts from P1's monitor_node (or Alert objects)
State out: "incidents" decision == incident -> for Diagnosis (P3)
           "watch"     ambiguous cases, for the LLM step / a later look
           "log"       one entry per incident, noise included (declare it in the graph
                       state as Annotated[list, operator.add] so LangGraph appends)
"""
from __future__ import annotations

from typing import Any

from common.schemas import Alert, Decision

from .agent import RuleTriage
from .config import TriageConfig

CONFIG = TriageConfig()


def triage_node(state: dict[str, Any]) -> dict[str, Any]:
    alerts = [a if isinstance(a, Alert) else Alert.from_dict(a) for a in state.get("alerts", [])]
    incidents = RuleTriage(CONFIG).process(alerts)
    log = [{"agent": "triage", "incident": i.id, "source": i.source, "dataset": i.dataset,
            "decision": i.decision.value, "severity": i.severity.value,
            "alerts": len(i.alert_ids), "reasons": i.reasons} for i in incidents]
    return {"incidents": [i for i in incidents if i.decision == Decision.INCIDENT],
            "watch": [i for i in incidents if i.decision == Decision.WATCH],
            "log": log}