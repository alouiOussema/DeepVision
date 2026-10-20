"""Call the Monitor node exactly as P4's orchestrator will, and print one alert of each type.

    python -m scripts.05_demo_node
"""
import json
import numpy as np
import pandas as pd
from config import settings
from ingest.hdfs import load_event_matrix, split_blocks
from agents.monitor.node import monitor_node


def main():
    nyc = pd.read_parquet(settings.NYC_HOURLY_FILE).sort_values("timestamp").tail(60 * 24)   # last 60 days

    ids, X, y, _ = load_event_matrix()                         # 300 unseen blocks: 200 normal + 100 failing
    test = np.flatnonzero(~split_blocks(len(y)))
    rng = np.random.default_rng(1)
    pick = np.r_[rng.choice(test[y[test] == 0], 200, replace=False),
                 rng.choice(test[y[test] == 1], 100, replace=False)]

    builds = pd.read_parquet(settings.BUILDS_FILE)             # failed builds of the last 3 days
    since = builds["started"].max() - pd.Timedelta(days=3)

    alerts = monitor_node({"signals": {"nyc": nyc}, "blocks": (ids[pick], X[pick]),
                           "builds": (builds, since)})["alerts"]
    print(f"{len(alerts)} alerts\n{pd.Series([(a['source'], a['kind']) for a in alerts]).value_counts().to_string()}\n")
    for source in ("model", "logs", "ci"):
        first = next((a for a in alerts if a["source"] == source), None)
        if first:
            print(json.dumps(first, indent=2, default=str))


if __name__ == "__main__":
    main()
