# tests/test_zscore.py
import numpy as np
from agents.monitor.detectors.zscore import ZScoreDetector

def test_outlier_detected():
    train = np.random.normal(0, 1, 1000)
    det = ZScoreDetector(threshold=4.0).fit(train)
    assert det.is_anomaly(50.0) is True
    assert det.is_anomaly(0.1) is False