"""Seasonal z-score: compare each hour with the SAME weekday-hour in training.

Monday 4 am is compared with other Mondays at 4 am, not with the global average.
Uses median and MAD (median absolute deviation) instead of mean and std, so a few
old anomalies in the training data do not distort what "normal" means.
"""
import numpy as np
import pandas as pd


class SeasonalZScore:
    name = "seasonal"
    needs_timestamps = True      # unlike the old detectors, it must know WHEN each value happened

    def __init__(self, threshold=3.0, tz="UTC", min_per_slot=5, by_weekday=True, min_threshold=2.0):
        self.threshold = threshold
        self.min_threshold = min_threshold
        self.tz = tz                  # time zone whose clock drives the daily rhythm
        self.min_per_slot = min_per_slot
        self.by_weekday = by_weekday

    def _slot(self, timestamps) -> np.ndarray:
        ts = pd.DatetimeIndex(timestamps)
        ts = ts.tz_localize("UTC") if ts.tz is None else ts
        ts = ts.tz_convert(self.tz)
        slot = ts.dayofweek * 24 + ts.hour if self.by_weekday else ts.hour
        return np.asarray(slot)

    def fit(self, values, timestamps):
        x = np.asarray(values, float)
        df = pd.DataFrame({"slot": self._slot(timestamps), "x": x})
        g = df.groupby("slot")["x"]
        self.med_ = g.median()
        dev = (df["x"] - df["slot"].map(self.med_)).abs()
        self.mad_ = dev.groupby(df["slot"]).median() * 1.4826
        self.n_ = g.size()
        self.g_med_ = float(np.median(x))
        self.g_mad_ = float(np.median(np.abs(x - self.g_med_)) * 1.4826) + 1e-9
        self.floor_ = 0.1 * self.g_mad_       # a slot may not be "too perfect"
        return self

    def baseline(self, timestamps):
        """What is NORMAL at these times: (typical value, spread). Used by score() and by the demo page."""
        slot = pd.Series(self._slot(timestamps))
        ok = (slot.map(self.n_).fillna(0).to_numpy() >= self.min_per_slot)
        med = np.where(ok, slot.map(self.med_).to_numpy(), self.g_med_)
        mad = np.where(ok, slot.map(self.mad_).to_numpy(), self.g_mad_)
        return med, np.maximum(mad, self.floor_)

    def score(self, values, timestamps) -> np.ndarray:
        med, mad = self.baseline(timestamps)
        return np.abs(np.asarray(values, float) - med) / mad

    def calibrate(self, values, timestamps, target_rate=0.01):
        """Pick the threshold so that about target_rate of normal hours alert.
        (Uses the training data itself for now; a validation period is better.)"""
        s = self.score(values, timestamps)
        # never below min_threshold: a feature that never varies must not get a limit of 0
        self.threshold = max(float(np.quantile(s, 1 - target_rate)), self.min_threshold)
        return self
