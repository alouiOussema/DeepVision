"""The tools the LLM agent can call to investigate one incident.

Each tool is a plain Python function with a clear docstring: Ollama reads the name, the
arguments and the docstring to decide WHEN to call it. Every tool returns short JSON text,
so the model reads facts, not raw data.

  get_incident(id)      what is this incident? (summary, measures, rule decision)
  find_related(id)      what else happened on the same system around the same time?
  compare_history(id)   is this unusual compared with similar incidents of this run?
  check_memory(id)      did a human already give feedback on this kind of incident?

make_tools() binds the functions to the incidents of the current run and to the memory.
facts_checklist() reads the same tool outputs in code: it is the fact-check of the LLM.
"""
from __future__ import annotations

import json
from datetime import timedelta

from common.schemas import Incident

from .memory import Memory, key_for


def _short(i: Incident) -> dict:
    return {"id": i.id, "features": i.features, "decision": i.decision.value,
            "severity": i.severity.value, "peak_strength": i.peak_ratio,
            "start": i.start.isoformat(timespec="minutes"), "hours": i.duration_hours}


def make_tools(incidents: list[Incident], memory: Memory) -> dict:
    by_id = {i.id: i for i in incidents}

    def _get(incident_id: str) -> Incident:
        if incident_id not in by_id:
            raise KeyError(f"unknown incident {incident_id}")
        return by_id[incident_id]

    # ------------------------------------------------------------------ tool 1
    def get_incident(incident_id: str) -> str:
        """Get the facts of one incident: what was abnormal, how strong, how long, and why the
        rules gave it their decision.

        Args:
            incident_id: the incident id, e.g. INC-00042
        """
        i = _get(incident_id)
        out = {**_short(i), "source": i.source, "dataset": i.dataset, "kind": i.kind,
               "n_alerts": len(i.alert_ids), "summary": i.summary, "rule_reasons": i.reasons[-2:]}
        if i.source in ("model", "metrics"):
            n = len(i.features)
            out["fact"] = (f"{n} different features are abnormal together in this incident: {', '.join(i.features)}"
                           if n > 1 else f"only 1 feature is abnormal: {i.features[0]}")
            if i.dataset == "deploy" and {"error_rate", "latency_ms"} <= set(i.features):
                out["fact"] += ". Errors AND latency rising together is the typical sign of a bad deployment"
        a = i.alerts[0].details
        if i.source == "logs":
            out["top_events"] = a.get("top_events")
        if i.source == "ci":
            out["flaky_probabilities"] = [x.details.get("flaky_probability") for x in i.alerts][:10]
            out["fail_rate_before"] = a.get("evidence", {}).get("recent_fail_rate")
        return json.dumps(out)

    # ------------------------------------------------------------------ tool 2
    def find_related(incident_id: str, hours: float = 6.0) -> str:
        """Find other incidents on the SAME system (same dataset or same CI project) that happened
        close in time. Several features going wrong together is stronger evidence of a real problem.

        Args:
            incident_id: the incident id, e.g. INC-00042
            hours: how many hours before and after to look (default 6)
        """
        i = _get(incident_id)
        if i.source == "logs":
            return json.dumps({"note": "HDFS blocks have no usable time; use compare_history instead"})
        w = timedelta(hours=hours)
        same = (lambda j: j.alerts[0].details.get("project") == i.alerts[0].details.get("project")) \
            if i.source == "ci" else (lambda j: j.dataset == i.dataset)
        near = [j for j in incidents if j.id != i.id and j.source == i.source and same(j)
                and j.kind == i.kind                        # a months-long drift is not the same event
                and j.start <= i.end + w and j.end >= i.start - w]
        other_feats = sorted({f for j in near for f in j.features} - set(i.features))
        fact = (f"{len(near)} other incident(s) within {hours:g} h on the same system; other features abnormal "
                f"nearby: {', '.join(other_feats)}" if other_feats else
                f"no OTHER feature was abnormal within {hours:g} h on the same system (isolated)")
        return json.dumps({"fact": fact, "related": [_short(j) for j in near[:5]]})

    # ------------------------------------------------------------------ tool 3
    def compare_history(incident_id: str) -> str:
        """Compare this incident with all similar incidents of the same run: is it stronger than
        usual, or is this kind of alert happening all the time (chronic noise)?

        Args:
            incident_id: the incident id, e.g. INC-00042
        """
        i = _get(incident_id)
        if i.source == "logs":
            peers = [j for j in incidents if j.source == "logs"]
            bigger = sum(len(j.alert_ids) > len(i.alert_ids) for j in peers)
            stronger = sum(j.peak_ratio < i.peak_ratio for j in peers)
            return json.dumps({"fact": (f"this pattern is on {len(i.alert_ids)} blocks: number {bigger + 1} of "
                                        f"{len(peers)} patterns by size; its strongest block is stronger than "
                                        f"{round(100 * stronger / len(peers))}% of the other patterns. Size and "
                                        f"strength alone cannot tell a real failure from a detector false alarm: "
                                        f"only human feedback can.")})
        if i.source == "ci":
            proj = i.alerts[0].details.get("project")
            peers = [j for j in incidents if j.source == "ci" and j.alerts[0].details.get("project") == proj]
            return json.dumps({"fact": f"project {proj} had failed builds on {len(peers)} day(s) of this run, "
                                       f"{sum(len(j.alert_ids) for j in peers)} failed builds in total"})
        peers = [j for j in incidents if j.dataset == i.dataset and set(j.features) & set(i.features)
                 and j.kind == i.kind]
        pct = round(100 * sum(j.peak_ratio < i.peak_ratio for j in peers) / max(len(peers), 1))
        word = "STRONGER than most" if pct >= 80 else "weaker than most" if pct < 50 else "about average"
        return json.dumps({"fact": f"this incident is stronger than {pct}% of {len(peers)} similar incidents: {word}"})

    # ------------------------------------------------------------------ tool 4
    def check_memory(incident_id: str) -> str:
        """Check if a human already gave feedback (false_alarm or real) on this kind of incident.
        Human feedback is the strongest evidence available.

        Args:
            incident_id: the incident id, e.g. INC-00042
        """
        k = key_for(_get(incident_id))
        found = memory.lookup(k)
        if not found:
            return json.dumps({"fact": f"no human feedback yet for {k}"})
        return json.dumps({"fact": f"a human marked {k} as {found['verdict'].upper()} "
                                   f"({found['counts']}). Note: {found['note']}"})

    return {f.__name__: f for f in (get_incident, find_related, compare_history, check_memory)}


def facts_checklist(inc: Incident, tools: dict) -> dict:
    """The evidence checklist computed in CODE from the same tool outputs the model reads.
    Used to fact-check the model: when the model and the tools disagree, the tools win."""
    fact = lambda name: json.loads(tools[name](inc.id)).get("fact", "")
    mem = fact("check_memory")
    out = {"feedback": "false_alarm" if "FALSE_ALARM" in mem else "real" if "as REAL" in mem else "none"}
    if inc.source != "logs":        # the logs comparison gives a percentage, no stronger/weaker word
        hist = fact("compare_history")
        out["strength"] = ("stronger" if "STRONGER" in hist else "weaker" if "weaker than most" in hist
                           else "average")
        rel = fact("find_related")
        others = rel.split("nearby: ")[1].split(", ") if "nearby: " in rel else []
        out["features_together"] = len(inc.features) + len(others)
        out["isolated"] = not others
        out["bad_deploy_sign"] = inc.dataset == "deploy" and {"error_rate", "latency_ms"} <= set(inc.features)
    return out