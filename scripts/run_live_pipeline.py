"""LIVE pipeline in the team repo: P1's real Monitor -> Triage (layer 1 + layer 2) -> to Diagnosis.

Put this file in DeepVision/scripts/ and run from the repo root:

    python -m scripts.run_live_pipeline

It builds the Monitor's input exactly like P1's scripts/05_node.py (NYC hours, HDFS blocks,
recent failed CI builds), then runs the same LangGraph graph as pipeline.py, with P1's
monitor_node as the first node instead of the replay.
"""
import numpy as np
import pandas as pd

from config import settings                                  # P1's settings
from ingest.hdfs import load_event_matrix, split_blocks      # P1's ingest
from agents.triage.memory import Memory                      # ours
from pipeline import build_graph, live_monitor_node, make_layer2


def monitor_input() -> dict:
    """The three inputs P1's monitor_node reads (same choice as his scripts/05_node.py)."""
    nyc = pd.read_parquet(settings.NYC_HOURLY_FILE).sort_values("timestamp").tail(60 * 24)
    ids, X, y, _ = load_event_matrix()
    test = np.flatnonzero(~split_blocks(len(y)))
    rng = np.random.default_rng(1)
    pick = np.r_[rng.choice(test[y[test] == 0], 200, replace=False),
                 rng.choice(test[y[test] == 1], 100, replace=False)]
    builds = pd.read_parquet(settings.BUILDS_FILE)
    since = builds["started"].max() - pd.Timedelta(days=3)
    return {"signals": {"nyc": nyc}, "blocks": (ids[pick], X[pick]), "builds": (builds, since)}


if __name__ == "__main__":
    graph = build_graph(live_monitor_node(), make_layer2(Memory()))
    for update in graph.stream({**monitor_input(), "log": []}, stream_mode="updates"):
        for node, out in update.items():
            info = out["log"][-1] if out.get("log") else f"{len(out.get('alerts', []))} alerts"  # P1's node has no log
            print(f"{node:13s} -> {info}")