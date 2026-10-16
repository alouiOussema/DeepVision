# scripts/04_run_monitor.py
import pandas as pd
from config import settings
from ingest import nyc
from agents.monitor.detectors.zscore import ZScoreDetector
from agents.monitor.core import Monitor
from logs.csv_log import log_alert
import pickle

def main():
    unified = pd.read_parquet(settings.UNIFIED_FILE)

    with open(settings.MODELS_DIR / "monitor_state.pkl", "rb") as f:
        all_bundles = pickle.load(f)

    val_end  = pd.Timestamp(settings.VAL_END, tz="UTC")

    total_alerts = 0
    for source, bundle in all_bundles.items():
        df = unified[unified.source == source]
        df = df[df.timestamp >= val_end].sort_values("timestamp")
        if len(df) == 0:
            continue

        # Monitor accepts a dict {feature: {detector_name: detector}}
        monitor = Monitor(bundle, source=source)

        n_alerts = 0
        for alert in monitor.run(df):
            log_alert(alert)
            n_alerts += 1

        print(f"{source}: {n_alerts} alerts / {len(df)} hours "
              f"({100*n_alerts/len(df):.2f}%)")
        total_alerts += n_alerts

    print(f"\nTotal: {total_alerts} alerts")


if __name__ == "__main__":
    main()