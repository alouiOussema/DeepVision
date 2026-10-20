"""Rolling baseline: compare the last few minutes with the hour just before them.

Used for BAD DEPLOYMENTS. A deploy changes the level of latency / error rate from one moment
to the next, so "normal" is simply what the metric looked like just before. No weekly rhythm
is needed, and slow changes of the day (traffic peaks) do not trigger it.

    now    = mean of the last `smooth` points
    before = median and std of the `window` points that came before those
    score  = |now - before_median| / (before_std / sqrt(smooth))      (a z-score of the mean)
The first `window + smooth` points get score 0 (not enough history yet).
"""
import numpy as np
import pandas as pd


class RollingBaseline:
    name = "rolling"
    needs_timestamps = True          # same calling convention as the seasonal detector

    def __init__(self, window=60, smooth=5, threshold=4.0, min_threshold=3.0, target_rate=0.001):
        self.window = window         # points used as the "before" picture
        self.smooth = smooth         # points averaged to describe "now"
        self.threshold = threshold
        self.min_threshold = min_threshold
        self.target_rate = target_rate   # share of normal MINUTES allowed to alert (much lower than per hour)

    def fit(self, values, timestamps=None):
        return self                  # nothing to learn: the baseline moves with the data

    def baseline(self, values):
        """(now, typical value before, spread of 'now' around it). Used by score() and by the demo page."""
        s = pd.Series(np.asarray(values, float))
        before = s.shift(self.smooth).rolling(self.window, min_periods=self.window)
        now = s.rolling(self.smooth, min_periods=self.smooth).mean()
        base = before.median()
        spread = np.maximum(before.std(), 0.02 * base.abs()) + 1e-9      # a flat metric must not give 0
        return now.to_numpy(), base.to_numpy(), (spread / np.sqrt(self.smooth)).to_numpy()

    def score(self, values, timestamps=None):
        now, base, spread = self.baseline(values)
        return np.nan_to_num(np.abs(now - base) / spread)

    def calibrate(self, values, timestamps=None, target_rate=0.01):
        """Pick the limit so that about target_rate of normal points alert."""
        s = self.score(values)[self.window + self.smooth:]
        if len(s):
            self.threshold = max(float(np.quantile(s, 1 - target_rate)), self.min_threshold)
        return self
