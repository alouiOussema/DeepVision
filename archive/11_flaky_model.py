"""Step 2 of the flaky-test detector: score every FAILED build with a flaky probability.

Trains on failed builds BEFORE the split date, tests on failed builds AFTER it (the model never
sees the future). Compares the model with two one-feature baselines, so we know what it adds.

Run from the project root:
    python -m scripts.11_flaky_model                 # split date 2015-07-01
    python -m scripts.11_flaky_model 2016-01-01      # another split date
"""
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score
from config import settings
from agents.monitor.flaky import make_features, SMOOTH

BUILDS_FILE = settings.ROOT / "data" / "builds.parquet"
SPLIT = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2015-07-01", tz="UTC")


def ranking_metrics(y, score):
    return {"AUC": round(roc_auc_score(y, score), 3), "avg_precision": round(average_precision_score(y, score), 3)}


def main():
    b = pd.read_parquet(BUILDS_FILE).sort_values(["project", "started"]).reset_index(drop=True)
    failed = b["label"].isin(["flaky_fail", "real_fail"])
    train_mask = failed & (b["started"] < SPLIT)
    test_mask = failed & (b["started"] >= SPLIT)
    prior = float((b.loc[train_mask, "label"] == "flaky_fail").mean())

    X = make_features(b, prior)
    y = (b["label"] == "flaky_fail").astype(int)
    Xtr, ytr, Xte, yte = X[train_mask], y[train_mask], X[test_mask], y[test_mask]
    base = float(yte.mean())
    print(f"split date {SPLIT.date()}")
    print(f"failed builds  train: {len(ytr):,} ({int(ytr.sum()):,} flaky = {ytr.mean():.1%})   "
          f"test: {len(yte):,} ({int(yte.sum()):,} flaky = {base:.1%})")
    print(f"random guessing would be right {base:.1%} of the time (average precision {base:.3f}, AUC 0.5)\n")

    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                           l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]

    rows = [{"scorer": "model (all features)", **ranking_metrics(yte, p)},
            {"scorer": "project history only", **ranking_metrics(yte, Xte["proj_flaky_rate"].fillna(prior))},
            {"scorer": "calm project (1 - recent fail rate)",
             **ranking_metrics(yte, 1 - Xte["fail_rate_20"].fillna(Xte["fail_rate_20"].median()))}]
    print(pd.DataFrame(rows).to_string(index=False))

    order = np.argsort(-p)
    n = len(p)
    top = []
    for share in (0.01, 0.02, 0.05, 0.10, 0.20):
        k = max(int(n * share), 1)
        hit = int(yte.to_numpy()[order[:k]].sum())
        top.append({"flag_top": f"{share:.0%} of failures", "builds_flagged": k,
                    "precision": round(hit / k, 3), "lift_vs_random": round(hit / k / base, 1),
                    "flaky_found": f"{hit / max(int(yte.sum()), 1):.0%}"})
    print("\nif we call the most suspicious failures 'flaky':")
    print(pd.DataFrame(top).to_string(index=False))

    sample = np.random.default_rng(0).choice(n, size=min(n, 20000), replace=False)
    imp = permutation_importance(model, Xte.iloc[sample], yte.iloc[sample], scoring="average_precision",
                                 n_repeats=3, random_state=0)
    imp = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False).head(8)
    print("\nwhat the model relies on (drop in average precision when the feature is shuffled):")
    print(imp.round(4).to_string())

    with open(settings.MODELS_DIR / "flaky_model.pkl", "wb") as fh:
        pickle.dump({"model": model, "features": list(X.columns), "prior": prior,
                     "smooth": SMOOTH, "base_rate_test": base, "split": str(SPLIT)}, fh)
    print(f"\nsaved {settings.MODELS_DIR / 'flaky_model.pkl'}")


if __name__ == "__main__":
    main()