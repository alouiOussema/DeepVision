"""Step 2: train every detector and save it in models/.

    python -m scripts.02_train

  NYC + deploy metrics -> seasonal anomaly detectors + PSI drift detectors   models/monitor_state.pkl
  HDFS blocks -> PCA on normal blocks                               models/hdfs_block_pca.pkl
  CI builds   -> flaky-failure model                                models/flaky_model.pkl
Quality is judged in 03_test.py, not here.
"""
import pickle
import pandas as pd
from config import settings
from ingest.hdfs import load_event_matrix, split_blocks
from agents.monitor.factory import make_detector, fit_detector, score_series, slice_period
from agents.monitor.detectors.block_pca import BlockPCA
from agents.monitor import ci_monitor


def save(name, obj):
    path = settings.MODELS_DIR / name
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  saved {path}")


def train_series():
    print(f"=== time series (detectors {settings.DETECTORS}, min_votes {settings.MIN_VOTES}) ===")
    models = {}
    for source, features in settings.FEATURES.items():
        df = pd.read_parquet(settings.SERIES_FILES[source])
        sp = settings.SPLITS[source]
        train, test = slice_period(df, *sp["train"]), slice_period(df, *sp["test"])
        tz = settings.TIMEZONES[source]
        print(f"{source}: {len(train)} training rows, {len(test)} test rows")

        bundle = {}
        for feature in features:                       # anomaly detectors, one set per feature
            tr = train[["timestamp", feature]].dropna()
            te = test[["timestamp", feature]].dropna()
            if len(tr) < settings.MIN_TRAIN_ROWS[source] or len(te) == 0:
                print(f"  {feature}: not enough data, skipped")
                continue
            bundle[feature] = {}
            for name in settings.DETECTORS[source]:
                det = fit_detector(make_detector(name, tz), tr[feature].values, tr["timestamp"])
                rate = float((score_series(det, te[feature].values, te["timestamp"]) > det.threshold).mean())
                print(f"  {feature:14s} [{name}] limit={det.threshold:5.2f}  test hours above the limit: {rate:.1%}")
                bundle[feature][name] = det
        for feature, days in settings.DRIFT.get(source, {}).items():     # drift detectors
            tr = train[["timestamp", feature]].dropna()
            te = test[["timestamp", feature]].dropna()
            det = fit_detector(make_detector("psi", tz, window_hours=days * 24), tr[feature].values, tr["timestamp"])
            rate = float((score_series(det, te[feature].values, te["timestamp"]) > det.threshold).mean())
            print(f"  {feature:14s} [psi {days:2d}d] limit={det.threshold:5.2f}  test hours in drift: {rate:.1%}")
            bundle.setdefault(feature, {})["psi"] = det
        models[source] = bundle
    save("monitor_state.pkl", models)


def train_logs():
    print("\n=== HDFS blocks (PCA on normal blocks) ===")
    ids, X, y, events = load_event_matrix()
    is_train = split_blocks(len(y))
    det = BlockPCA(false_alarm_rate=settings.HDFS_FALSE_ALARM_RATE).fit(X[is_train & (y == 0)])
    print(f"  trained on {int((is_train & (y == 0)).sum()):,} normal blocks, limit={det.threshold:.3f}")
    save("hdfs_block_pca.pkl", {"detector": det, "events": events})


def train_ci():
    print("\n=== CI builds (flaky model) ===")
    b, X, y, train_mask, test_mask, prior = ci_monitor.prepare(pd.read_parquet(settings.BUILDS_FILE),
                                                          settings.FLAKY_SPLIT_DATE)
    model = ci_monitor.fit(X[train_mask], y[train_mask])
    print(f"  trained on {int(train_mask.sum()):,} failed builds before {settings.FLAKY_SPLIT_DATE} "
          f"({int(y[train_mask].sum()):,} flaky)")
    save("flaky_model.pkl", {"model": model, "features": list(X.columns), "prior": prior,
                             "smooth": ci_monitor.SMOOTH, "base_rate_test": float(y[test_mask].mean()),
                             "split": settings.FLAKY_SPLIT_DATE})


if __name__ == "__main__":
    train_series()
    train_logs()
    train_ci()
