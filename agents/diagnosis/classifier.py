"""Step 2b: the trained ROOT-CAUSE CLASSIFIER. Feature vector -> one probability per cause.

The rule weights are written by hand; the classifier learns them from labelled incidents
(scenarios.py), so it can catch combinations the rules miss. Both are kept and blended
(agent.py), and evaluate.py measures each alone and together.

Model: gradient-boosted trees (same family as P1's flaky model), small and regularised because the
benchmark has a few hundred incidents.

Calibration, twice, because P4 multiplies our confidence by the action risk: a confidence of 0.8
must mean "right about 80 % of the time".
  - the classifier's own probabilities: isotonic, 3-fold (CalibratedClassifierCV)
  - the agent's FINAL confidence (rules and classifier blended): `calibrator`, an isotonic curve
    fitted on out-of-fold diagnoses of the train split (train.py), never on cases it was trained on
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from common.schemas import Category

from . import paths
from .evidence import FEATURES


class RootCauseClassifier:
    def __init__(self, model=None, classes: list[Category] | None = None, features: list[str] | None = None,
                 calibrator=None):
        self.model = model
        self.classes = classes or []
        self.features = features or list(FEATURES)
        self.calibrator = calibrator      # top probability -> observed accuracy (train.py, out-of-fold)

    def calibrate(self, p: float) -> float:
        return float(self.calibrator.predict([p])[0]) if self.calibrator is not None else p

    def fit_calibration(self, raw: list[float], correct: list[bool]) -> "RootCauseClassifier":
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, correct)
        return self

    @property
    def trained(self) -> bool:
        return self.model is not None

    def fit(self, X: np.ndarray, y: list[Category]) -> "RootCauseClassifier":
        labels = np.array([c.value for c in y])
        base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=200,
                                              l2_regularization=1.0, class_weight="balanced", random_state=0)
        smallest = min(np.unique(labels, return_counts=True)[1])
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3) if smallest >= 3 else base
        self.model.fit(X, labels)
        self.classes = [Category(c) for c in self.model.classes_]
        return self

    def predict_proba(self, vector: list[float]) -> dict[Category, float]:
        p = self.model.predict_proba(np.asarray([vector], float))[0]
        return {c: float(v) for c, v in zip(self.classes, p)}

    # ------------------------------------------------------------------ files
    def save(self, path: Path = paths.CLASSIFIER) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "classes": [c.value for c in self.classes],
                         "features": self.features, "calibrator": self.calibrator}, f)

    @classmethod
    def load(cls, path: Path = paths.CLASSIFIER) -> "RootCauseClassifier":
        """An untrained classifier if the file is missing, or was trained on other features."""
        if not Path(path).exists():
            return cls()
        with open(path, "rb") as f:
            m = pickle.load(f)
        if m["features"] != list(FEATURES):
            return cls()
        return cls(m["model"], [Category(c) for c in m["classes"]], m["features"], m.get("calibrator"))
