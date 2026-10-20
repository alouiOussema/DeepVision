"""Try the Monitor node exactly as P4's orchestrator will call it.

    python -m scripts.09_demo_node
"""
import json
import numpy as np
import pandas as pd
from config import settings
from ingest.hdfs_blocks import load_event_matrix, split_blocks
from agents.monitor.node import monitor_node


def main():
    # last 60 days of the NYC test period
    u = pd.read_parquet(settings.UNIFIED_FILE)
    nyc = u[u.source == "nyc"].sort_values("timestamp").tail(60 * 24)

    # 300 unseen HDFS blocks: 200 normal + 100 failing
    ids, X, y, _ = load_event_matrix()
    test = np.flatnonzero(~split_blocks(len(y)))
    rng = np.random.default_rng(1)
    pick = np.r_[rng.choice(test[y[test] == 0], 200, replace=False),
                 rng.choice(test[y[test] == 1], 100, replace=False)]

    # CI builds: alert on the failed builds of the last 3 days (the rest is history)
    builds = pd.read_parquet(settings.ROOT / "data" / "builds.parquet")
    since = builds["started"].max() - pd.Timedelta(days=3)

    out = monitor_node({"signals": {"nyc": nyc}, "blocks": (ids[pick], X[pick]),
                        "builds": (builds, since)})
    alerts = out["alerts"]
    by = pd.Series([(a["source"], a["kind"]) for a in alerts]).value_counts()
    print(f"{len(alerts)} alerts\n{by.to_string()}\n")
    for kind in ("model", "logs", "ci"):
        first = next((a for a in alerts if a["source"] == kind), None)
        if first:
            print(json.dumps(first, indent=2, default=str))


if __name__ == "__main__":
    main()