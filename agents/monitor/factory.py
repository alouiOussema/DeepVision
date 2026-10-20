"""Build, fit and score a detector by name. Training, the Monitor and the tests all use it."""
import numpy as np
import pandas as pd
from config import settings
from .detectors.psi import PSIDrift
from .detectors.rolling import RollingBaseline
from .detectors.seasonal import SeasonalZScore
from .detectors.zscore import ZScoreDetector


def slice_period(df, start, end):
    """Rows with start <= timestamp < end."""
    return df[(df.timestamp >= pd.Timestamp(start, tz="UTC")) &
              (df.timestamp <  pd.Timestamp(end,   tz="UTC"))]


def make_detector(name, tz="UTC", window_hours=None):
    if name == "seasonal":
        return SeasonalZScore(tz=tz)
    if name == "rolling":
        return RollingBaseline(window=settings.ROLLING_WINDOW, smooth=settings.ROLLING_SMOOTH,
                               target_rate=settings.ROLLING_TARGET_RATE)
    if name == "zscore":
        return ZScoreDetector(threshold=settings.ZSCORE_THRESHOLD)
    if name == "psi":
        return PSIDrift(threshold=settings.PSI_THRESHOLD, n_bins=settings.PSI_BINS,
                        window=window_hours or settings.PSI_WINDOW_HOURS)
    raise ValueError(f"unknown detector '{name}' (use seasonal, rolling, zscore or psi)")


def fit_detector(det, values, timestamps):
    values = np.asarray(values, float)
    if det.needs_timestamps:            # seasonal and psi also pick their own threshold
        det.fit(values, timestamps)
        det.calibrate(values, timestamps, getattr(det, "target_rate", settings.TARGET_ALERT_RATE))
    else:
        det.fit(values)
    return det


def score_series(det, values, timestamps):
    """One score per value."""
    values = np.asarray(values, float)
    if det.needs_timestamps:
        return det.score(values, timestamps)
    return np.array([det.score(float(v)) for v in values])
