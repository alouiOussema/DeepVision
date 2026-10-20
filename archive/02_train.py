"""Train the detectors for every source and save them.

Run from the project root:  python -m scripts.02_train
Which detectors are trained is decided in config/settings.py (ACTIVE_DETECTORS).
Quality is NOT judged here (there are no labels for most sources): use 03_test_planted.py.
"""
import pickle
import numpy as np
import pandas as pd
from config import settings
from agents.monitor.factory import make_detector, fit_detector, score_series


def between(df, start, end):
    return df[(df.timestamp >= pd.Timestamp(start, tz="UTC")) &
              (df.timestamp <  pd.Timestamp(end,   tz="UTC"))]


def main():
    unified = pd.read_parquet(settings.UNIFIED_FILE)
    print(f"detectors: {settings.ACTIVE_DETECTORS}   min_votes: {settings.MIN_VOTES}")

    all_bundles = {}
    for source, features in settings.FEATURES.items():
        print(f"\n=== {source} ===")
        df = unified[unified.source == source]
        sp = settings.SPLITS[source]
        train = between(df, *sp["train"])
        test = between(df, *sp["test"])
        print(f"  train hours: {len(train)}   test hours: {len(test)}")
        tz = settings.TIMEZONES[source]

        bundle = {}
        for feature in features:
            tr = train[["timestamp", feature]].dropna() if feature in train.columns else []
            te = test[["timestamp", feature]].dropna() if feature in test.columns else []
            if len(tr) < settings.MIN_TRAIN_ROWS[source] or len(te) == 0:
                print(f"  {feature}: not enough data, skipped")
                continue
            detectors = {}
            for name in settings.ACTIVE_DETECTORS:
                det = fit_detector(make_detector(name, tz), tr[feature].values, tr["timestamp"])
                rate = float((score_series(det, te[feature].values, te["timestamp"])
                              > det.threshold).mean())
                print(f"  {feature:14s} [{name:8s}] limit={det.threshold:6.2f}   "
                      f"share of test hours above the limit = {rate:.1%}")
                detectors[name] = det
            bundle[feature] = detectors
        # drift detectors (PSI): reference = the training period
        for feature, days in settings.DRIFT.get(source, {}).items():
            if feature not in train.columns:
                continue
            tr = train[["timestamp", feature]].dropna()
            te = test[["timestamp", feature]].dropna()
            if len(tr) < settings.MIN_TRAIN_ROWS[source] or len(te) == 0:
                continue
            det = fit_detector(make_detector("psi", tz, window_hours=days * 24),
                               tr[feature].values, tr["timestamp"])
            rate = float((score_series(det, te[feature].values, te["timestamp"]) > det.threshold).mean())
            print(f"  {feature:14s} [psi {days:2d}d  ] limit={det.threshold:6.2f}   "
                  f"share of test hours in drift = {rate:.1%}")
            bundle.setdefault(feature, {})["psi"] = det

        all_bundles[source] = bundle

    with open(settings.MODELS_DIR / "monitor_state.pkl", "wb") as f:
        pickle.dump(all_bundles, f)
    print(f"\nSaved detectors -> {settings.MODELS_DIR / 'monitor_state.pkl'}")


if __name__ == "__main__":
    main()