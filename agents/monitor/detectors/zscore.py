# detectors/zscore.py
import numpy as np
from .base import Detector

class ZScoreDetector(Detector):
    name = "zscore"

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.mean_ = float(values.mean())
        self.std_  = float(values.std()) + 1e-9
        return self

    def score(self, x: float) -> float:
        return abs(x - self.mean_) / self.std_