"""Measure detectors against planted anomalies (known problems we inject ourselves).

Run from the project root:
    python -m scripts.03_test_planted           # NYC
    python -m scripts.03_test_planted travis    # another source
Detectors compared: settings.COMPARE_DETECTORS
"""
import sys
import pandas as pd
from config import settings
from agents.monitor.plant import plant
from agents.monitor.evaluate import evaluate
from agents.monitor.factory import make_detector, fit_detector, score_series

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "nyc"


def between(df, start, end):
    return df[(df.timestamp >= pd.Timestamp(start, tz="UTC")) &
              (df.timestamp <  pd.Timestamp(end,   tz="UTC"))]


def main():
    df = pd.read_parquet(settings.UNIFIED_FILE)
    df = df[df.source == SOURCE]
    sp = settings.SPLITS[SOURCE]
    train, test = between(df, *sp["train"]), between(df, *sp["test"])
    tz = settings.TIMEZONES[SOURCE]
    pd.set_option("display.width", 140)

    for feature in settings.FEATURES[SOURCE]:
        if feature not in df.columns:
            continue
        tr = train[["timestamp", feature]].dropna()
        te = test[["timestamp", feature]].dropna()
        if len(tr) < settings.MIN_TRAIN_ROWS[SOURCE] or len(te) < 200:
            print(f"\n{SOURCE}/{feature}: not enough data, skipped")
            continue

        series, labels = plant(te, feature, tr[feature].values, tr["timestamp"])

        tables, summaries = {}, {}
        for name in settings.COMPARE_DETECTORS:
            try:
                det = fit_detector(make_detector(name, tz), tr[feature].values, tr["timestamp"])
            except ImportError as e:
                print(f"  ({name} skipped: {e})")
                continue
            scores = score_series(det, series[feature].values, series["timestamp"])
            tables[name], summaries[name] = evaluate(
                series["timestamp"], series["planted"], scores > det.threshold, labels)

        first = next(iter(tables.values()))
        view = first[["kind", "size", "events"]].copy()
        full = view.copy()
        for name, t in tables.items():
            view[name] = t["recall"].values
            full[f"recall_{name}"] = t["recall"].values
            full[f"delay_{name}"] = t["median_delay_h"].values
            full[f"chance_{name}"] = t["chance"].values

        print(f"\n=== {SOURCE} / {feature}  ({len(labels)} planted events) ===")
        print("share of planted events caught (recall):")
        print(view.round(2).to_string(index=False))

        board = []
        for name, sm in summaries.items():
            t = tables[name]
            short = t[t.kind.isin(["spike", "drop"])]["recall"].mean()
            board.append({"detector": name,
                          "false_alerts_per_week": round(sm["false_alerts_per_week"], 1),
                          "alert_rate_clean": f"{sm['alert_rate_on_clean_hours']:.1%}",
                          "recall_overall": f"{sm['recall_overall']:.0%}",
                          "recall_spikes_drops": f"{short:.0%}"})
        print("\nscoreboard (spikes/drops last 1 hour, so luck plays no role there):")
        print(pd.DataFrame(board).to_string(index=False))

        labels.to_csv(settings.REPORTS_DIR / f"planted_labels_{SOURCE}_{feature}.csv", index=False)
        full.to_csv(settings.REPORTS_DIR / f"compare_{SOURCE}_{feature}.csv", index=False)


if __name__ == "__main__":
    main()