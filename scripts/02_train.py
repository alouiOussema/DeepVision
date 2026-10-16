import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from config import settings
from agents.monitor.detectors.zscore import ZScoreDetector
from agents.monitor.detectors.iforest import IForestDetector
from agents.monitor.detectors.autoencoder import AutoencoderDetector


FEATURES = {
    "nyc":    ["trip_count", "avg_fare", "avg_distance"],
    "hdfs":   ["log_volume", "error_count", "warn_count"],
    "travis": ["build_count", "failure_rate", "avg_duration"],
}

SPLITS = {
    "nyc":    {"train": ("2025-01-01", "2026-01-01"),
               "test":  ("2026-01-01", "2026-09-01")},
    "hdfs":   {"train": ("2008-11-09 20:00", "2008-11-10 12:00"),
               "test":  ("2008-11-10 12:00", "2008-11-11 23:59")},
    "travis": {"train": ("2011-01-01", "2016-01-01"),
               "test":  ("2016-01-01", "2018-01-01")},
}

MIN_TRAIN_ROWS = {
    "nyc":    100,
    "hdfs":   10,
    "travis": 100,
}


def make_detectors():
    return {
        "zscore":  ZScoreDetector(threshold=2.0),
        #"iforest": IForestDetector(threshold=1.5),
        #"autoenc": AutoencoderDetector(threshold=2.0, bottleneck=1, epochs=500, patience=30),
    }


def supervised_metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    out = {
        "n":              int(len(y_true)),
        "n_pos_true":     int(np.sum(y_true)),
        "n_pos_pred":     int(np.sum(y_pred)),
        "precision":      float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":         float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":             float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
    }
    if 0 < np.sum(y_true) < len(y_true):
        try:
            out["auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["auc"] = None
    return out


def unsupervised_metrics(timestamps, scores, preds):
    """For NYC/Travis: no labels, so we report shape metrics."""
    flagged   = scores[preds == 1]
    unflagged = scores[preds == 0]

    out = {
        "alert_rate": float(preds.mean()),
        "score_mean": float(scores.mean()),
        "score_std":  float(scores.std()),
        "separation_ratio": (
            float(flagged.mean() / (unflagged.mean() + 1e-9))
            if len(flagged) > 0 and len(unflagged) > 0 else None
        ),
    }

    # histogram (20 bins)
    hist, edges = np.histogram(scores, bins=20)
    out["score_histogram"] = {
        "counts": hist.tolist(),
        "edges":  [float(e) for e in edges],
    }

    # alert distribution by hour-of-day
    ts = pd.to_datetime(timestamps)
    hour_counts = pd.Series(ts[preds == 1].hour).value_counts().sort_index()
    out["alerts_by_hour"] = {int(h): int(c) for h, c in hour_counts.items()}

    # stability: monthly alert counts
    month_counts = pd.Series(1, index=ts[preds == 1]).resample("MS").sum()
    out["alerts_by_month"] = {
        str(k.date()): int(v) for k, v in month_counts.items()
    }
    # stability score: std / mean of monthly counts (lower = more stable)
    if len(month_counts) > 1 and month_counts.mean() > 0:
        out["monthly_stability"] = float(
            month_counts.std() / month_counts.mean()
        )
    else:
        out["monthly_stability"] = None

    return out


def main():
    unified = pd.read_parquet(settings.UNIFIED_FILE)

    hdfs_labels = None
    lab_path = settings.HOURLY_DIR / "hdfs_labels.parquet"
    if lab_path.exists():
        hdfs_labels = pd.read_parquet(lab_path)

    all_bundles = {}
    all_metrics = {}

    for source, features in FEATURES.items():
        print(f"\n=== {source} ===")
        df = unified[unified.source == source].copy()
        if len(df) == 0:
            print("  no data, skip")
            continue

        split = SPLITS[source]
        train_start = pd.Timestamp(split["train"][0], tz="UTC")
        train_end   = pd.Timestamp(split["train"][1], tz="UTC")
        test_start  = pd.Timestamp(split["test"][0], tz="UTC")
        test_end    = pd.Timestamp(split["test"][1], tz="UTC")

        train_df = df[(df.timestamp >= train_start) & (df.timestamp < train_end)]
        test_df  = df[(df.timestamp >= test_start)  & (df.timestamp < test_end)]

        print(f"  train rows: {len(train_df)}")
        print(f"  test rows : {len(test_df)}")

        # HDFS ground truth
        y_true = None
        if source == "hdfs" and hdfs_labels is not None:
            merged = test_df.merge(hdfs_labels, on="timestamp", how="left")
            y_true = merged["is_anomalous_hour"].fillna(0).astype(int).values
            print(f"  labeled test hours: {len(y_true)} "
                  f"({int(y_true.sum())} anomalies)")

        source_bundle = {}
        source_metrics = {}

        for feature in features:
            if feature not in train_df.columns:
                print(f"  {feature}: not in data — skip")
                continue

            train_values = train_df[feature].dropna().values
            test_values  = test_df[feature].dropna().values

            if len(train_values) < MIN_TRAIN_ROWS[source]:
                print(f"  {feature}: only {len(train_values)} train rows — skip")
                continue
            if len(test_values) == 0:
                print(f"  {feature}: no test rows — skip")
                continue

            # align test timestamps with test_values (after dropna)
            test_aligned = test_df.loc[test_df[feature].notna(), "timestamp"].values

            detectors = make_detectors()
            feature_metrics = {}

            for name, det in detectors.items():
                det.fit(train_values)
                scores = np.array([det.score(v) for v in test_values])
                preds  = (scores > det.threshold).astype(int)

                # ALWAYS: unsupervised metrics
                m = unsupervised_metrics(test_aligned, scores, preds)

                # model internals
                if name == "zscore":
                    m["model"] = {"mean": det.mean_, "std": det.std_}
                elif name == "iforest":
                    m["model"] = {"n_estimators": 200,
                                  "score_mean": det.mean_,
                                  "score_std":  det.std_}
                elif name == "autoenc":
                    m["model"] = {
                        "bottleneck": det.bottleneck,
                        "epochs_used": len(det.history),
                        "final_loss": det.history[-1] if det.history else None,
                        "err_mean": det._err_mean,
                        "err_std":  det._err_std,
                    }

                # HDFS only: supervised metrics
                if y_true is not None and len(y_true) == len(test_values):
                    m.update(supervised_metrics(y_true, preds, scores))

                feature_metrics[name] = m

                # one-line summary
                line = (f"  {feature:14s} [{name:7s}] "
                        f"alert={m['alert_rate']:.3f} "
                        f"sep={m.get('separation_ratio') or 0:.2f}")
                if "f1" in m:
                    line += (f"  P={m['precision']:.3f} "
                             f"R={m['recall']:.3f} "
                             f"F1={m['f1']:.3f}")
                    if m.get("auc") is not None:
                        line += f" AUC={m['auc']:.3f}"
                print(line)

            source_bundle[feature]  = detectors
            source_metrics[feature] = feature_metrics

        all_bundles[source] = source_bundle
        all_metrics[source] = source_metrics

    # persist
    with open(settings.MODELS_DIR / "monitor_state.pkl", "wb") as f:
        pickle.dump(all_bundles, f)
    with open(settings.MODELS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved detectors → {settings.MODELS_DIR / 'monitor_state.pkl'}")
    print(f"Saved metrics   → {settings.MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()