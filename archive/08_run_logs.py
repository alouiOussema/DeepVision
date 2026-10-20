"""Replay HDFS blocks through the log detector and write Alerts (source = "logs").

Trains PCA on normal blocks of one random half, then replays the OTHER half as if the blocks
were arriving from the pipeline. Because we know the labels, it also reports how many alerts
were real failures.

Run AFTER 04_run_monitor.py (that script restarts alerts.csv; this one appends to it).
    python -m scripts.08_run_logs
"""
import pickle
import numpy as np
from config import settings
from ingest.hdfs_blocks import load_event_matrix, split_blocks
from agents.monitor.detectors.block_pca import BlockPCA
from agents.monitor.log_monitor import LogMonitor
from logs.csv_log import log_alert


def main():
    ids, X, y, events = load_event_matrix()
    is_train = split_blocks(len(y))
    det = BlockPCA(false_alarm_rate=settings.HDFS_FALSE_ALARM_RATE).fit(X[is_train & (y == 0)])
    with open(settings.MODELS_DIR / "hdfs_block_pca.pkl", "wb") as f:
        pickle.dump({"detector": det, "events": events}, f)

    test = ~is_train
    monitor = LogMonitor(det, events, dataset="hdfs")
    alerted = []
    for alert in monitor.run(ids[test], X[test]):
        log_alert(alert)
        alerted.append(alert.details["block_id"])

    truth = dict(zip(ids[test], y[test]))
    real = sum(truth[b] for b in alerted)
    n_fail = int(y[test].sum())
    print(f"replayed {monitor.checked:,} blocks ({n_fail:,} really failing)")
    print(f"alerts: {len(alerted):,}   real failures among them: {real:,} "
          f"(precision {real / max(len(alerted), 1):.0%})")
    print(f"failing blocks found: {real:,} of {n_fail:,} (recall {real / max(n_fail, 1):.0%})")
    print(f"alerts appended to {settings.ALERTS_CSV}")


if __name__ == "__main__":
    main()