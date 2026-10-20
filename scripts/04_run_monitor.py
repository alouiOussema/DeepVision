"""Step 4: replay the data through the trained detectors and write every alert to logs/alerts.csv.

    python -m scripts.04_run_monitor

  NYC test months   -> Monitor      (anomaly + drift alerts, source "model")
  HDFS test blocks  -> LogMonitor   (one alert per suspicious block, source "logs")
  CI builds         -> CIMonitor    (one alert per failed build of the last N days, with flaky probability)
alerts.csv is rebuilt on every run.
"""
import pickle
import pandas as pd
from config import settings
from ingest.hdfs import load_event_matrix, split_blocks
from agents.monitor.alert import log_alert, reset_log
from agents.monitor.ci_monitor import CIMonitor
from agents.monitor.factory import slice_period
from agents.monitor.log_monitor import LogMonitor
from agents.monitor.series_monitor import SeriesMonitor


def load(name):
    with open(settings.MODELS_DIR / name, "rb") as f:
        return pickle.load(f)


def run_series():
    total = 0
    for source, bundle in load("monitor_state.pkl").items():
        df = pd.read_parquet(settings.SERIES_FILES[source])
        df = slice_period(df, *settings.SPLITS[source]["test"])
        monitor = SeriesMonitor(bundle, dataset=source, min_votes=settings.MIN_VOTES,
                                max_gap_hours=settings.MERGE_GAP_HOURS[source])
        counts = {"anomaly": 0, "drift": 0}
        for alert in monitor.run(df):
            log_alert(alert)
            counts[alert.kind] += 1
        pct = 100 * monitor.flagged_hours / max(monitor.checked, 1)
        print(f"{source}: {sum(counts.values())} alerts [anomaly {counts['anomaly']}, drift {counts['drift']}] "
              f"| anomalous hour-features {monitor.flagged_hours} of {monitor.checked} ({pct:.2f}%)")
        total += sum(counts.values())
    return total


def run_logs():
    m = load("hdfs_block_pca.pkl")
    ids, X, y, events = load_event_matrix()
    test = ~split_blocks(len(y))
    monitor = LogMonitor(m["detector"], m["events"], dataset="hdfs")
    truth = dict(zip(ids[test], y[test]))
    hits = n = 0
    for alert in monitor.run(ids[test], X[test]):
        log_alert(alert)
        n += 1
        hits += truth[alert.details["block_id"]]
    fails = int(y[test].sum())
    print(f"hdfs: {n} alerts from {monitor.checked:,} blocks | really failing: {hits:,} "
          f"(precision {hits / max(n, 1):.0%}, recall {hits / max(fails, 1):.0%})")
    return n


def run_ci():
    builds = pd.read_parquet(settings.BUILDS_FILE)
    since = builds["started"].max() - pd.Timedelta(days=settings.CI_REPLAY_DAYS)
    n = 0
    for alert in CIMonitor(load("flaky_model.pkl")).run(builds, since):
        log_alert(alert)
        n += 1
    print(f"ci: {n} alerts (failed builds since {since.date()}, each with its flaky probability)")
    return n


if __name__ == "__main__":
    reset_log()
    total = run_series() + run_logs() + run_ci()
    print(f"\nTotal: {total} alerts -> {settings.ALERTS_CSV}")
