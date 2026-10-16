import numpy as np
from sklearn.ensemble import IsolationForest
from .base import Detector


class IForestDetector(Detector):
    name = "iforest"

    def __init__(self, threshold: float = 1.5, random_state: int = 42):
        self.threshold = threshold
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.01,
            random_state=random_state,
        )
        self.mean_ = None
        self.std_  = None

    def fit(self, values):
        X = np.asarray(values, dtype=float).reshape(-1, 1)
        self.model.fit(X)
        raw = -self.model.decision_function(X)   # higher = more anomalous
        self.mean_ = float(raw.mean())
        self.std_  = float(raw.std()) + 1e-9
        return self

    def score(self, x: float) -> float:
        raw = float(-self.model.decision_function(np.array([[x]]))[0])
        return (raw - self.mean_) / self.std_