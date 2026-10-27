"""The pipeline as a LangGraph graph: Monitor (P1) -> Triage layer 1 -> [layer 2] -> Diagnosis (P3).

    monitor ──> layer1_rules ──┬── any WATCH or HDFS pattern? ──> layer2_agent ──> to_diagnosis ──> diagnosis
                               └── no ──────────────────────────────────────────> to_diagnosis ──> diagnosis

Three ways to feed the graph:
  - REPLAY (works now): the monitor node replays P1's real alerts from data/alerts.csv for a
    time window, exactly as his monitor_node would emit them.
  - LIVE (once our code sits in the team repo next to agents/monitor): P1's own monitor_node
    is the first node. Nothing else changes, because both speak the same Alert contract.
  - SIM (no download needed): P1's deploy simulator run through P1's rolling detector, with the
    change log and the metrics, so Diagnosis (P3) can blame a release.

Diagnosis also reads state["context"] (change log + raw metrics, a ContextStore) when the monitor
node provides it; without it, it still diagnoses, but cannot point to the release to roll back.

    py pipeline.py                 # runs the "Deploy · Jan 15-18" window, prints each node
    py pipeline.py sim             # the same graph on a held-out deploy simulation (no data files needed)
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, StateGraph

from agents.triage import paths
from agents.triage.agent import RuleTriage
from agents.triage.config import TriageConfig
from agents.triage.io import load_alerts
from agents.triage.llm_agent import LLMTriage
from agents.triage.memory import Memory
from agents.diagnosis.context import ContextStore
from agents.diagnosis.node import make_diagnosis_node
from common.schemas import Alert, Decision, Incident


class PipelineState(TypedDict, total=False):
    signals: dict                         # LIVE mode, input of P1's monitor_node: {"nyc": hourly DataFrame, ...}
    blocks: tuple                         # LIVE mode: (block_ids, event_count_matrix) of HDFS blocks
    builds: tuple                         # LIVE mode: (builds table, since timestamp) for CI
    alerts: list                          # out of the monitor (dicts in P1's format, or Alert objects)
    incidents: list[Incident]             # out of layer 1 (every incident, with the rules' decision)
    reviews: list                         # out of layer 2 (one per reviewed incident)
    final: list[Incident]                 # after layer 2
    to_diagnosis: list[Incident]          # what P3 receives
    context: ContextStore                 # change log + raw metrics around the alerts (read by P3)
    diagnoses: list                       # out of P3: one Diagnosis per incident -> Decision (P4)
    log: Annotated[list, operator.add]    # every node appends what it did


# ------------------------------------------------------------------ demo windows of P1's data
@dataclass
class Window:
    name: str
    story: str
    keep: Callable[[Alert], bool]


WINDOWS = {
    "deploy": Window("Deploy · Jan 15-18 (bad deployments)",
                     "Minute-level pipeline metrics around new deployments (P1's simulated deploys).",
                     lambda a: a.dataset == "deploy" and date(2026, 1, 15) <= a.start.date() <= date(2026, 1, 18)),
    "nyc": Window("NYC taxi · New Year's Day 2026",
                  "Hourly NYC taxi trips: the night of Jan 1 moves several features at once.",
                  lambda a: a.dataset == "nyc" and a.kind == "anomaly" and a.start.date() == date(2026, 1, 1)),
    "ci": Window("CI · Aug 2, 2016 (Travis builds)",
                 "Every failed build of one day, with P1's flaky probability.",
                 lambda a: a.family == "ci" and a.timestamp.date() == date(2016, 8, 2)),
    "hdfs": Window("HDFS · 400 blocks of one scoring run",
                   "Blocks P1's log detector found strange, grouped by event pattern.",
                   lambda a: a.family == "logs" and abs(int(a.details["block_id"][4:])) % 34 == 0),
}


def replay_monitor(window: Window, path=paths.ALERTS, context: ContextStore | None = None):
    """A stand-in for P1's monitor_node: emits his real alerts for one window, in time order,
    in the same dict format as his node (feature is called "signal" there).
    context: the change log + metrics behind those alerts, passed on to Diagnosis (P3)."""
    def monitor(state: PipelineState) -> dict:
        alerts = sorted((a for a in load_alerts(path) if window.keep(a)), key=lambda a: a.start)
        out = [{"id": a.id, "timestamp": a.timestamp.isoformat(), "source": a.source, "signal": a.feature,
                "score": a.score, "kind": a.kind, "dataset": a.dataset, "details": a.details} for a in alerts]
        return {"alerts": out, **({"context": context} if context else {}),
                "log": [{"node": "monitor", "alerts": len(out), "window": window.name}]}
    return monitor


def deploy_context() -> ContextStore | None:
    """The deploy list + minute metrics P1's 01_build_data wrote (data/deploy/), if they exist.
    Only the deploy TIMES reach Diagnosis: the truth columns are dropped by ContextStore."""
    import pandas as pd
    from config import settings
    if not (settings.DEPLOY_METRICS_FILE.exists() and settings.DEPLOY_EVENTS_FILE.exists()):
        return None
    return ContextStore.from_deploy_table(pd.read_parquet(settings.DEPLOY_METRICS_FILE),
                                          pd.read_parquet(settings.DEPLOY_EVENTS_FILE))


def simulated_monitor(seed: int = 4):
    """No data files needed: P1's deploy simulator (seed 4 is never used to train Diagnosis) run
    through P1's rolling detector, emitted in his monitor_node's format, with the context."""
    from agents.diagnosis.scenarios import simulate_deploy

    def monitor(state: PipelineState) -> dict:
        alerts, ctx, _ = simulate_deploy(seed)
        out = [{**a.to_contract(), "timestamp": a.timestamp.isoformat(), "kind": a.kind,
                "dataset": a.dataset, "details": a.details} for a in alerts]   # = P1's monitor_node
        return {"alerts": out, "context": ctx,
                "log": [{"node": "monitor", "alerts": len(out), "window": f"deploy simulation (seed {seed})"}]}
    return monitor


# ------------------------------------------------------------------ triage nodes
def layer1_rules(state: PipelineState) -> dict:
    alerts = [a if isinstance(a, Alert) else Alert.from_dict(a) for a in state.get("alerts", [])]
    incidents = RuleTriage(TriageConfig()).process(alerts)
    count = {d.value: sum(i.decision == d for i in incidents) for d in Decision}
    return {"incidents": incidents, "final": incidents,
            "log": [{"node": "layer1_rules", "alerts": len(alerts), "incidents": len(incidents), **count}]}


def needs_layer2(state: PipelineState) -> str:
    """The conditional edge: only unsure cases go to the LLM agent."""
    return "layer2_agent" if any(LLMTriage.needs_review(i) for i in state["incidents"]) else "to_diagnosis"


def make_layer2(memory: Memory, model: str = "qwen2.5:7b", history: list[Incident] | None = None,
                on_review=None, use_llm: bool = True):
    def layer2_agent(state: PipelineState) -> dict:
        incidents = state["incidents"]
        if not use_llm:
            return {"reviews": [], "log": [{"node": "layer2_agent", "skipped": "LLM switched off"}]}
        # the tools may compare with older incidents (history), but never with a twin of the same alerts
        mine = {a for i in incidents for a in i.alert_ids}
        ctx = [h for h in (history or []) if not mine & set(h.alert_ids)]
        final, reviews = LLMTriage(memory, model=model).run(incidents, context=ctx, on_review=on_review)
        changed = sum(r.decision != r.rule_decision for r in reviews)
        return {"final": final, "reviews": reviews,
                "log": [{"node": "layer2_agent", "reviewed": len(reviews), "changed": changed}]}
    return layer2_agent


def to_diagnosis(state: PipelineState) -> dict:
    """Hand-off to P3: only real incidents. P3's diagnosis node is the next node."""
    sent = [i for i in state["final"] if i.decision == Decision.INCIDENT]
    return {"to_diagnosis": sent, "log": [{"node": "to_diagnosis", "sent_to_P3": len(sent)}]}


def build_graph(monitor_node, layer2_node, diagnosis_node=None):
    """diagnosis_node: P3's node; by default rules + classifier + RAG, without LLM (fast).
    make_diagnosis_node(llm_model="llama3.1:8b") adds the LLM explanations."""
    g = StateGraph(PipelineState)
    g.add_node("monitor", monitor_node)
    g.add_node("layer1_rules", layer1_rules)
    g.add_node("layer2_agent", layer2_node)
    g.add_node("to_diagnosis", to_diagnosis)
    g.add_node("diagnosis", diagnosis_node or make_diagnosis_node())
    g.set_entry_point("monitor")
    g.add_edge("monitor", "layer1_rules")
    g.add_conditional_edges("layer1_rules", needs_layer2, ["layer2_agent", "to_diagnosis"])
    g.add_edge("layer2_agent", "to_diagnosis")
    g.add_edge("to_diagnosis", "diagnosis")
    g.add_edge("diagnosis", END)
    return g.compile()


def live_monitor_node():
    """LIVE mode, once our code is in the team repo: P1's real Monitor is the first node.
    His node reads {"signals", "blocks", "builds"} from the state and writes {"alerts"}."""
    from agents.monitor.node import monitor_node      # P1's code (team repo)
    return monitor_node


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "deploy"
    if mode != "sim" and not paths.ALERTS.exists():
        print(f"{paths.ALERTS} not found (P1's alerts are not in git): running the 'sim' mode instead.\n")
        mode = "sim"
    if mode == "sim":
        graph = build_graph(simulated_monitor(), make_layer2(Memory()))
        print("Window: deploy simulation (P1's simulator -> P1's rolling detector)\n")
    else:
        w = WINDOWS[mode]
        history = RuleTriage(id_prefix="HIST").process(load_alerts(paths.ALERTS))
        ctx = deploy_context() if mode == "deploy" else None
        graph = build_graph(replay_monitor(w, context=ctx), make_layer2(Memory(), history=history))
        print(f"Window: {w.name}\n")
    final = {}
    for update in graph.stream({"log": []}, stream_mode="updates"):
        for node, out in update.items():
            final.update(out)
            print(f"[{datetime.now():%H:%M:%S}] {node:13s} -> {out['log'][-1]}")
    print("\nDiagnoses sent to Decision (P4):")
    for d in final.get("diagnoses", []):
        print(f"  {d.incident_id}  {d.category.value:14s} {d.sub_cause:20s} {d.suspect or '-':12s} "
              f"confidence {d.confidence:.2f} -> {d.suggested_action}{'  [needs a human]' if d.needs_human else ''}")