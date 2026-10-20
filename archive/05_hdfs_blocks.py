"""HDFS the right way: one row per BLOCK, with REAL labels (Success / Fail).

We train ONLY on normal blocks, then check how well each method finds the failing blocks
among blocks it has never seen. Methods: one number per block, IsolationForest on all 29
event counts, and PCA on all 29 event counts (the one we use in the Monitor).
The last table shows the trade-off: a looser limit finds more failures but raises more false alarms.

Run from the project root:  python -m scripts.05_hdfs_blocks
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from config import settings
from ingest.hdfs_blocks import load_event_matrix, split_blocks
from agents.monitor.detectors.block_pca import BlockPCA

SEED = 0


def total_events(Xtr, Xte):
    a, b = Xtr.sum(1), Xte.sum(1)
    med = np.median(a)
    mad = np.median(np.abs(a - med)) * 1.4826 + 1e-9
    return np.abs(a - med) / mad, np.abs(b - med) / mad


def iforest_29(Xtr, Xte):
    A, B = np.log1p(Xtr), np.log1p(Xte)
    m = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(A)
    return -m.decision_function(A), -m.decision_function(B)


def pca_29(Xtr, Xte):
    d = BlockPCA(false_alarm_rate=settings.HDFS_FALSE_ALARM_RATE, seed=SEED).fit(Xtr)
    return d.score(Xtr), d.score(Xte)


METHODS = {"total_events": total_events, "iforest_29": iforest_29, "pca_29": pca_29}


def at_limit(s_tr, s_te, yte, rate):
    limit = np.quantile(s_tr, 1 - rate)
    pred = (s_te > limit).astype(int)
    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    return {"false_alarm_rate": f"{rate:.1%}", "precision": round(p, 3), "recall": round(r, 3),
            "F1": round(f, 3), "false_alarms": int(((pred == 1) & (yte == 0)).sum())}


def main():
    ids, X, y, _ = load_event_matrix()
    is_train = split_blocks(len(y), SEED)
    Xtr, Xte, yte = X[is_train & (y == 0)], X[~is_train], y[~is_train]
    print(f"blocks: {len(y):,}   failing: {int(y.sum()):,} ({y.mean():.1%})")
    print(f"trained on {len(Xtr):,} normal blocks; tested on {len(yte):,} unseen blocks "
          f"({int(yte.sum()):,} failing)\n")

    rows, scores = [], {}
    for name, method in METHODS.items():
        s_tr, s_te = method(Xtr, Xte)
        scores[name] = (s_tr, s_te)
        row = {"method": name, "AUC": round(roc_auc_score(yte, s_te), 3)}
        row.update(at_limit(s_tr, s_te, yte, 0.01))
        rows.append(row)
    out = pd.DataFrame(rows)
    print("all methods, limit = 1% false alarms on normal blocks:")
    print(out.to_string(index=False))

    print("\npca_29: the trade-off when the limit moves")
    s_tr, s_te = scores["pca_29"]
    trade = pd.DataFrame([at_limit(s_tr, s_te, yte, r) for r in (0.005, 0.01, 0.02, 0.05, 0.10)])
    print(trade.to_string(index=False))

    out.to_csv(settings.REPORTS_DIR / "hdfs_blocks.csv", index=False)
    trade.to_csv(settings.REPORTS_DIR / "hdfs_pca_tradeoff.csv", index=False)


if __name__ == "__main__":
    main()