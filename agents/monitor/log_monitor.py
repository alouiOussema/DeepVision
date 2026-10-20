"""Log Monitor: replay HDFS blocks through the PCA detector and emit Alerts (source="logs").

Blocks have no timestamp in the event matrix, so the alert time is the moment of scoring.
details.top_events says WHICH events made the block look strange: useful evidence for the
Diagnosis agent.
"""
from datetime import datetime, timezone
import numpy as np
from config import settings
from .alert import Alert


class LogMonitor:
    def __init__(self, detector, event_names, dataset="hdfs", max_alerts=None):
        self.detector = detector
        self.event_names = list(event_names)
        self.dataset = dataset
        self.source = settings.DATASET_KIND[dataset]          # "logs"
        self.max_alerts = max_alerts                          # keep only the strangest N, if set
        self.checked = 0

    def run(self, block_ids, X):
        X = np.asarray(X, float)
        self.checked = len(X)
        ratio = self.detector.score(X) / self.detector.threshold    # 1.0 = just at the limit
        idx = np.flatnonzero(ratio > 1.0)
        if self.max_alerts is not None and len(idx) > self.max_alerts:
            idx = idx[np.argsort(-ratio[idx])[: self.max_alerts]]
            idx.sort()
        if len(idx) == 0:
            return
        resid = self.detector.residual_sq(X[idx])
        now = datetime.now(timezone.utc)
        for k, i in enumerate(idx):
            share = resid[k] / max(resid[k].sum(), 1e-12)
            top = np.argsort(-share)[:3]
            yield Alert(
                detector="block_pca", source=self.source, dataset=self.dataset, kind="anomaly",
                feature="block_events", score=float(ratio[i]), threshold=1.0, timestamp=now,
                value=float(ratio[i] * self.detector.threshold),
                details={"block_id": str(block_ids[i]),
                         "top_events": [{"event": self.event_names[j], "share": round(float(share[j]), 3)}
                                        for j in top]},
            )
