"""Plain z-score against ONE global mean and std. Kept only as the 'before' picture:
03_test.py compares it with the seasonal detector."""
import numpy as np


class ZScoreDetector:
    name = "zscore"
    needs_timestamps = False

    def __init__(self, threshold=3.0):
        self.threshold = threshold

    def fit(self, values):
        values = np.asarray(values, float)
        self.mean_ = float(values.mean())
        self.std_ = float(values.std()) + 1e-9
        return self

    def score(self, x):
        return abs(x - self.mean_) / self.std_
