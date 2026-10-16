"""
Quick alert smoke test.
Runs entirely on synthetic data — no training, no parquet, no HDFS parse.
"""
from datetime import datetime, timezone
import pandas as pd

from agents.monitor.alert import Alert
from agents.monitor.detectors.zscore import ZScoreDetector
from agents.monitor.core import Monitor
from logs.csv_log import log_alert


def main():
    # 1. Build a tiny hourly series: 50 normal hours + 1 huge anomaly
    hours = pd.date_range("2025-01-01", periods=51, freq="h", tz="UTC")
    values = [100.0] * 50 + [5000.0]   # last hour is 50x normal
    df = pd.DataFrame({"timestamp": hours, "value": values, "source": "synthetic"})

    # 2. Fit a z-score on the first 50 hours only
    train = df.iloc[:50]
    test  = df.iloc[50:]

    det = ZScoreDetector(threshold=3.0).fit(train["value"].values)
    print(f"Fitted: mean={det.mean_:.2f} std={det.std_:.2f} "
          f"threshold={det.threshold}")

    # 3. Run the Monitor
    monitor = Monitor({"value": {"zscore": det}}, source="synthetic")
    alerts = list(monitor.run(test))

    if not alerts:
        print("NO ALERTS — something is wrong.")
        return

    # 4. Show the alert
    alert = alerts[0]
    print()
    print("=== ALERT FIRED ===")
    print(f"  id        : {alert.id}")
    print(f"  detector  : {alert.detector}")
    print(f"  source    : {alert.source}")
    print(f"  feature   : {alert.feature}")
    print(f"  score     : {alert.score:.3f}")
    print(f"  threshold : {alert.threshold}")
    print(f"  timestamp : {alert.timestamp}")
    print(f"  value     : {alert.value}")
    print(f"  details   : {alert.details}")

    # 5. Log it
    log_alert(alert)
    print()
    print("Logged to logs/alerts.csv")

    # 6. Verify it round-trips through CSV
    logged = pd.read_csv("logs/alerts.csv").iloc[-1]
    print()
    print("=== LAST ROW IN CSV ===")
    print(logged.to_string())


if __name__ == "__main__":
    main()