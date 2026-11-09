"""A real (local) retrain backend: champion vs challenger with scikit-learn on P1's NYC hourly table.

The repo has no live prediction model and no MLflow, so "the model" is defined here, small and honest:
    target    avg_fare of the hour (the deck's January 2026 fare drift is what it must survive)
    features  hour, weekday, avg_distance, trip_count
    champion  trained on the old data: everything before the recent window   (the model built before the drift)
    challenger trained on the recent window (ActionConfig.retrain_window_days, the PSI window)
    holdout   the most recent days (retrain_holdout_days): neither model has seen them; both are scored there (MAE)

It only writes the challenger to data/action/challengers/. It has NO deploy method: the live model is never touched,
so it is safe in simulation mode (touches_production = False). Data file missing -> BackendRefused (no fake data).
"""
from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import paths
from .backends import BackendRefused, Comparison

FEATURES = ["hour", "weekday", "avg_distance", "trip_count"]
TARGET = "avg_fare"


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    X = pd.DataFrame({"hour": ts.dt.hour, "weekday": ts.dt.day_of_week,
                      "avg_distance": df["avg_distance"].to_numpy(), "trip_count": df["trip_count"].to_numpy()})
    return X, df[TARGET].reset_index(drop=True)


def _fit(df: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, y = _xy(df)
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0).fit(X, y)


def _mae(model, df: pd.DataFrame) -> float:
    X, y = _xy(df)
    return float((pd.Series(model.predict(X)) - y).abs().mean())


class SklearnNYCModel:
    touches_production = False

    def __init__(self, data: pd.DataFrame | None = None, path: Path = paths.NYC_HOURLY,
                 out_dir: Path = paths.CHALLENGERS_DIR, min_rows: int = 24 * 7):
        self.data, self.path, self.out_dir, self.min_rows = data, Path(path), Path(out_dir), min_rows

    def _load(self) -> pd.DataFrame:
        if self.data is not None:
            df = self.data
        elif self.path.exists():
            df = pd.read_parquet(self.path)
        else:
            raise BackendRefused(f"{self.path} not found (P1's scripts/01_build_data.py writes it): "
                                 "no data, no retraining")
        missing = {"timestamp", TARGET, *FEATURES[2:]} - set(df.columns)
        if missing:
            raise BackendRefused(f"NYC hourly table has no column(s) {sorted(missing)}")
        df = df.dropna(subset=[TARGET, *FEATURES[2:]]).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    def retrain_and_compare(self, order_id: str, now: datetime, window_days: int, holdout_days: int) -> Comparison:
        df = self._load()
        end = df["timestamp"].max()
        hold_start = end - timedelta(days=holdout_days)
        win_start = hold_start - timedelta(days=window_days)
        old, recent, hold = (df[df.timestamp < win_start], df[(df.timestamp >= win_start) & (df.timestamp < hold_start)],
                             df[df.timestamp >= hold_start])
        for name, part in (("old data (champion)", old), ("recent window (challenger)", recent), ("holdout", hold)):
            if len(part) < min(self.min_rows, 24):
                raise BackendRefused(f"not enough rows in the {name}: {len(part)}")
        champion, challenger = _fit(old), _fit(recent)
        cid = f"challenger-{order_id}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / f"{cid}.pkl", "wb") as f:
            pickle.dump({"model": challenger, "features": FEATURES, "target": TARGET,
                         "trained_on": [str(win_start), str(hold_start)]}, f)
        return Comparison(f"mae_{TARGET}", _mae(champion, hold), _mae(challenger, hold), cid, n_holdout=len(hold),
                          notes=f"champion on {len(old)} h before {win_start:%Y-%m-%d}, challenger on {len(recent)} h, "
                                f"scored on the last {len(hold)} h")

    def discard(self, challenger_id: str) -> None:
        (self.out_dir / f"{challenger_id}.pkl").unlink(missing_ok=True)
