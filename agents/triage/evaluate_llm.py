"""Measure the LLM layer: rules alone vs rules + LLM agent, on the same ground truth.

  A. NYC planted anomalies : does the agent rescue real problems left in WATCH,
                             without adding too many pages?
  B. HDFS blocks           : with a little human feedback in memory, does precision go up?
  C. Reading accuracy      : how often does the model copy the tool facts correctly into its
                             checklist? (the fact-check in llm_agent logs every misread field)

Feedback protocol for HDFS (realistic, not cheating): an operator checks 5 sample blocks of
each pattern the rules paged, once. 1 or fewer really failing -> false_alarm; 4 or more ->
real; otherwise no feedback. That uses 5 labels per pattern, out of 13,703 blocks.
"""
from __future__ import annotations

import random

import pandas as pd

from . import paths
from common.schemas import Decision, Incident

from .agent import RuleTriage
from .evaluate import EVAL, eval_hdfs, eval_planted
from .io import load_alerts
from .llm_agent import LLMTriage
from .memory import Memory, key_for


# ------------------------------------------------------------------ feedback simulation
def simulate_operator(memory: Memory, incidents: list[Incident], sample: int = 5, seed: int = 0) -> int:
    lab = pd.read_csv(EVAL / "hdfs_labels.csv").set_index("block_id")["failing"].to_dict()
    rnd, n = random.Random(seed), 0
    for i in incidents:
        if i.source != "logs" or i.decision != Decision.INCIDENT:
            continue
        blocks = rnd.sample(i.alerts, min(sample, len(i.alerts)))
        failing = sum(lab.get(a.details["block_id"], 0) for a in blocks)
        if failing <= 1:
            memory.add_feedback(key_for(i), "false_alarm", f"operator checked {len(blocks)} blocks: {failing} failing")
        elif failing >= len(blocks) - 1:
            memory.add_feedback(key_for(i), "real", f"operator checked {len(blocks)} blocks: {failing} failing")
        n += 1
    return n


# ------------------------------------------------------------------ reading accuracy
FIELDS = ("feedback", "strength", "isolated", "bad_deploy_sign", "features_together")


def reading_accuracy(reviews) -> dict:
    """Share of reviews where the model copied each field like the tools said (fact-check)."""
    out = {}
    for f in FIELDS:
        checked = [r for r in reviews if r.evidence and f in r.facts]
        if checked:
            out[f] = round(1 - sum(f in r.misread for r in checked) / len(checked), 2)
    return out


# ------------------------------------------------------------------ full comparison
def compare(model: str = "qwen2.5:7b", features=("trip_count", "avg_fare", "avg_distance"), verbose=True):
    reviews_all = []

    def llm_post(memory: Memory):
        def post(incidents):
            new, reviews = LLMTriage(memory, model=model).run(incidents, verbose=verbose)
            reviews_all.extend(reviews)
            return new
        return post

    rows = []
    for f in features:
        if verbose:
            print(f"\n--- NYC {f}")
        r0 = eval_planted(f)
        r1 = eval_planted(f, post=llm_post(Memory(EVAL / "_none.json")))
        for name, r in (("rules", r0), ("rules+LLM", r1)):
            rows.append({"case": f"nyc/{f}", "system": name, "caught": r["caught_by_monitor"],
                         "paged": r["paged"], "watch": r["watch"], "noise": r["noise"],
                         "recall": r["triage_recall"], "pages": r["pages"]})

    if verbose:
        print("\n--- HDFS")
    alerts = load_alerts(paths.ALERTS)
    memory = Memory(EVAL / "_feedback.json")
    memory.clear()
    n_fb = simulate_operator(memory, RuleTriage().process([a for a in alerts if a.family == "logs"]))
    h0 = eval_hdfs(alerts)
    h1 = eval_hdfs(alerts, post=llm_post(memory))
    hdfs = [{"system": name, "incidents": h["incidents"], "recall_of_failing": h["recall_of_failing"],
             "precision": h["precision_paged"]} for name, h in (("rules", h0), ("rules+LLM", h1))]

    errors = sum(bool(r.error) for r in reviews_all)
    guard = sum(r.overridden for r in reviews_all)
    return {"nyc": pd.DataFrame(rows), "hdfs": pd.DataFrame(hdfs), "feedback_given": n_fb,
            "memory": memory.data, "reading_accuracy": reading_accuracy(reviews_all),
            "misread_reviews": sum(bool(r.misread) for r in reviews_all),
            "reviews": len(reviews_all), "errors": errors, "guardrail_used": guard}