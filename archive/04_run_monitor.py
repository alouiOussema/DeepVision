"""Run the trained detectors over each source's OWN test period.

Run from the project root:  python -m scripts.04_run_monitor
"""
import pickle
import pandas as pd
from config import settings
from agents.monitor.monitor import Monitor
from logs.csv_log import log_alert, reset_log


def main():
    unified = pd.read_parquet(settings.UNIFIED_FILE)
    with open(settings.MODELS_DIR / "monitor_state.pkl", "rb") as f:
        all_bundles = pickle.load(f)

    reset_log()                       # alerts.csv is rebuilt on every run
    total = 0
    for source, bundle in all_bundles.items():
        start, end = settings.SPLITS[source]["test"]
        df = unified[(unified.source == source)
                     & (unified.timestamp >= pd.Timestamp(start, tz="UTC"))
                     & (unified.timestamp <  pd.Timestamp(end,   tz="UTC"))]
        if len(df) == 0:
            print(f"{source}: no rows in test period {start} -> {end}, skipped")
            continue

        monitor = Monitor(bundle, dataset=source, min_votes=settings.MIN_VOTES)
        n, by_kind = 0, {"anomaly": 0, "drift": 0}
        for alert in monitor.run(df):
            log_alert(alert)
            n += 1
            by_kind[alert.kind] += 1
        pct = 100 * monitor.flagged_hours / max(monitor.checked, 1)
        print(f"{source}: {n} alerts  [anomaly {by_kind['anomaly']}, drift {by_kind['drift']}]  "
              f"| anomalous hour-features: {monitor.flagged_hours} of {monitor.checked} ({pct:.2f}%)"
              f"  | hour-features inside a drift episode: {monitor.drift_hours}")
        total += n

    print(f"\nTotal: {total} alerts  ->  {settings.ALERTS_CSV}")


if __name__ == "__main__":
    main()