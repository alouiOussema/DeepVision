"""Flaky-test model and CI Monitor, in one file.

PART 1 (features, prepare, fit): used by 02_train and 03_test to build the model.
PART 2 (CIMonitor): every FAILED build becomes an Alert (source="ci") carrying a flaky probability.

Alert.score = flaky probability / average flaky rate of failed builds
              (how many times more likely than a typical failure; 1.0 = typical).
The decision (re-run or page a human) is NOT taken here: that is P4's policy.
Every feature uses only what was known BEFORE the build.
Input of the monitor: a builds table from ingest/travis.py, with enough project history.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from config import settings
from .alert import Alert

# ---------------- PART 1: the model ----------------
SMOOTH = 5          # a project with few failures is pulled toward the global flaky rate


def make_features(b, prior, smooth=SMOOTH):
    """b is sorted by project and time. Every feature uses only what was known BEFORE the build."""
    failed = b["label"].isin(["flaky_fail", "real_fail"]).astype(int)
    flaky = (b["label"] == "flaky_fail").astype(int)
    by_proj = b["project"]
    before_failed = failed.groupby(by_proj, sort=False).cumsum() - failed     # earlier failures
    before_flaky = flaky.groupby(by_proj, sort=False).cumsum() - flaky        # ... that were flaky

    f = pd.DataFrame(index=b.index)
    f["proj_flaky_rate"] = (before_flaky + smooth * prior) / (before_failed + smooth)
    f["proj_failures_before"] = np.log1p(before_failed)
    f["fail_rate_20"] = b["fail_rate_20"]
    f["prev_failed"] = np.where(b["prev_status"].isna(), np.nan, (b["prev_status"] != "passed").astype(float))
    f["log_prev_builds"] = np.log1p(b["n_prev_builds"])
    f["log_mins_since_prev"] = np.log1p(b["mins_since_prev"].clip(lower=0))
    f["n_jobs"] = b["n_jobs"]
    if "is_pr" in b:
        f["is_pr"] = b["is_pr"].astype(float)
    if "duration" in b:
        f["log_duration"] = np.log1p(b["duration"].clip(lower=0))
    if "tests_run" in b:
        f["log_tests_run"] = np.log1p(b["tests_run"].clip(lower=0))
        f["log_tests_failed"] = np.log1p(b["tests_failed"].clip(lower=0))
        f["tests_failed_share"] = b["tests_failed"] / b["tests_run"].where(b["tests_run"] > 0)
    f["hour"] = b["started"].dt.hour
    f["weekday"] = b["started"].dt.dayofweek
    return f


def prepare(builds, split_date):
    """Sort, split failed builds in time, build features.
    Returns b, X, y (1 = flaky), train_mask, test_mask, prior."""
    split = pd.Timestamp(split_date)
    split = split.tz_localize("UTC") if split.tzinfo is None else split
    b = builds.sort_values(["project", "started"]).reset_index(drop=True)
    failed = b["label"].isin(["flaky_fail", "real_fail"])
    train_mask = failed & (b["started"] < split)
    test_mask = failed & (b["started"] >= split)
    prior = float((b.loc[train_mask, "label"] == "flaky_fail").mean())
    y = (b["label"] == "flaky_fail").astype(int)
    return b, make_features(b, prior), y, train_mask, test_mask, prior


def fit(X_train, y_train):
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, random_state=0).fit(X_train, y_train)


# ---------------- PART 2: the monitor ----------------

def _clean(v):
    return None if pd.isna(v) else float(v)


class CIMonitor:
    def __init__(self, bundle, dataset="travis"):
        self.m = bundle                       # contents of models/flaky_model.pkl
        self.dataset = dataset
        self.source = settings.DATASET_KIND[dataset]      # "ci"

    def run(self, builds, since=None):
        b = builds.sort_values(["project", "started"]).reset_index(drop=True)
        X = make_features(b, self.m["prior"], self.m["smooth"])[self.m["features"]]
        pick = b["status"] == "failed"
        if since is not None:
            since = pd.Timestamp(since)
            since = since.tz_localize("UTC") if since.tzinfo is None else since
            pick &= b["started"] >= since
        idx = np.flatnonzero(pick.to_numpy())
        if len(idx) == 0:
            return
        prob = self.m["model"].predict_proba(X.iloc[idx])[:, 1]
        base = self.m["base_rate_test"]
        for i, p in zip(idx, prob):
            r, x = b.iloc[i], X.iloc[i]
            yield Alert(
                detector="flaky_model", source=self.source, dataset=self.dataset, kind="anomaly",
                feature="build_failed", score=float(p / base), threshold=1.0,
                timestamp=r["started"], value=float(p),
                details={"flaky_probability": round(float(p), 4), "project": str(r["project"]),
                         "build_id": int(r["build_id"]), "commit": str(r["commit"]),
                         "tests_failed": _clean(r.get("tests_failed", np.nan)),
                         "evidence": {"project_flaky_rate": _clean(x["proj_flaky_rate"]),
                                      "recent_fail_rate": _clean(x["fail_rate_20"]),
                                      "previous_build_failed": _clean(x["prev_failed"])}},
            )
