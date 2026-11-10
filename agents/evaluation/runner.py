"""Runs the team's graph on an injected episode.

Decision, the simulated human and Action are replayed one incident at a time, at the moment the incident was
detected. In real life incidents arrive hours apart; deciding a whole batch at once would trip the circuit breaker
and close later incidents as duplicates. Every agent writes into the run's own folder, never the real inbox.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from agents.action.agent import ActionAgent
from agents.action.clock import SimClock
from agents.action.config import ActionConfig
from agents.action.executors import job_refs_from_incidents
from agents.decision.agent import DecisionAgent
from agents.decision.inbox import ActionInbox
from agents.decision.log import DecisionLog, action_key
from common.schemas import Action, ActionStatus

from .config import EvalConfig
from .injector import Episode
from .trace import Tracer
from .world import backends_for

OPEN = (ActionStatus.PENDING_APPROVAL, ActionStatus.APPROVED)


class AgentFiles:
    """Where the run's agents write: one directory per episode, nothing shared with the real pipeline."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.log = DecisionLog(self.dir / "decision_log.jsonl", self.dir / "decision_history.jsonl")
        self.inbox = ActionInbox(self.dir / "action_inbox.json", self.log)
        self.claims = self.dir / "claims"
        self.notifications = self.dir / "notifications.jsonl"
        self.deploy_requests = self.dir / "deploy_requests.jsonl"


def monitor_node(ep: Episode):
    """Stands in for P1's monitor_node: emits the episode's alerts (already in the team's Alert contract)."""
    def monitor(state: dict) -> dict:
        return {"alerts": list(ep.alerts), "context": ep.context,
                "log": [{"node": "monitor", "alerts": len(ep.alerts), "window": ep.name}]}
    return monitor


def replay_node(ep: Episode, files: AgentFiles, cfg: EvalConfig, tracer: Tracer,
                decision_agent: DecisionAgent | None = None):
    """Decision -> human approval -> Action, for each incident in time order.

    Approvals are sent to the tracer directly because PipelineState has no field for them.
    """
    agent = decision_agent or DecisionAgent(log=files.log)

    def decision(state: dict[str, Any]) -> dict[str, Any]:
        incidents = {i.id: i for i in state.get("to_diagnosis", [])}
        diagnoses = sorted(state.get("diagnoses", []), key=lambda d: incidents[d.incident_id].start)
        job_refs = job_refs_from_incidents(list(incidents.values()))
        decisions, approvals, runs = [], [], []
        for diag in diagnoses:
            inc = incidents[diag.incident_id]
            now = inc.start
            open_orders = {(d.snapshot.get("action_key") or action_key(d), d.action): d.id
                           for d in files.inbox.all() if d.status in OPEN}
            order = agent.run([diag], [inc], now=now, open_orders=open_orders)[0]
            files.inbox.add([order])
            decisions.append(order)
            t = now
            if order.status is ActionStatus.PENDING_APPROVAL:
                t = now + timedelta(minutes=cfg.human_delay_min)
                order = files.inbox.resolve(order.id, "approve", reviewer=cfg.human_name, now=t,
                                            note="evaluation: every order sent to the approval page is approved")
                approvals.append({"at": t, "incident_id": inc.id, "decision_id": order.id, "verdict": "approved",
                                  "status": order.status.value, "note": f"after {cfg.human_delay_min:g} min"})
            if order.status is ActionStatus.APPROVED and order.action is not Action.NONE:
                inj = ep.injection_for(inc.alert_ids)
                act = ActionAgent(ActionConfig(), backends_for(inj, inc, ep.context, cfg), files.inbox, SimClock(t),
                                  claims_dir=files.claims, notifications=files.notifications,
                                  deploy_requests=files.deploy_requests)
                runs += act.run([order], job_refs)
        tracer.on_node("human", {"approvals": approvals})
        return {"decisions": decisions, "to_action": [], "actions_done": runs,
                "pending_approval": [], "log": [{"agent": "decision+action (replay)", "decided": len(decisions),
                                                 "approved_by_human": len(approvals), "actions_run": len(runs)}]}
    return decision


def run_episode(ep: Episode, directory: Path, cfg: EvalConfig | None = None, diagnosis_node=None,
                decision_agent: DecisionAgent | None = None) -> tuple[Tracer, dict, AgentFiles]:
    """Returns the tracer (every agent's steps), the final graph state and the run's agent files (inbox, log)."""
    from pipeline import build_graph, make_layer2
    from agents.triage.memory import Memory

    cfg = cfg or EvalConfig()
    files = AgentFiles(directory)
    tracer = Tracer(ep.name)
    graph = build_graph(monitor_node(ep), make_layer2(Memory(), use_llm=False), diagnosis_node=diagnosis_node,
                        decision_node=replay_node(ep, files, cfg, tracer, decision_agent))
    state = tracer.consume(graph.stream({"log": []}, stream_mode="updates"))
    return tracer, state, files
