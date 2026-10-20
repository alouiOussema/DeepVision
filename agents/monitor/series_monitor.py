"""SeriesMonitor: score each feature, apply the vote, merge consecutive hours into ONE alert.

Two kinds of detectors live in a bundle:
  - anomaly detectors (seasonal, zscore, ...) VOTE together (min_votes) -> kind="anomaly"
  - drift detectors (psi) never vote with them; each one raises its own alerts -> kind="drift"

Alert.score = how many times above its limit the signal is (1.0 = just at the limit),
so Alert.threshold is always 1.0 and different detectors are comparable.
"""
import numpy as np
import pandas as pd
from config import settings
from .alert import Alert
from .factory import score_series


class SeriesMonitor:
    def __init__(self, bundle: dict, dataset: str, min_votes: int = 1,
                 max_gap_hours: int = 1, drift_gap_hours: int = 72):
        # bundle: {feature: {detector_name: fitted detector}}
        self.bundle = bundle
        self.dataset = dataset                          # nyc / travis / hdfs
        self.source = settings.DATASET_KIND[dataset]    # contract: ci / logs / metrics / model
        self.min_votes = min_votes
        self.max_gap = pd.Timedelta(hours=max_gap_hours)
        self.drift_gap = pd.Timedelta(hours=drift_gap_hours)   # drift flickers around its limit: merge
        self.checked = 0           # hour-features examined in the last run
        self.flagged_hours = 0     # hour-features flagged as ANOMALIES in the last run
        self.drift_hours = 0       # hour-features sitting inside a DRIFT episode

    def run(self, df):
        self.checked = self.flagged_hours = self.drift_hours = 0
        for feature, detectors in self.bundle.items():
            if feature not in df.columns:
                continue
            d = df[["timestamp", feature]].dropna().sort_values("timestamp")
            if d.empty:
                continue
            t = pd.DatetimeIndex(d["timestamp"])
            v = d[feature].to_numpy(float)
            self.checked += len(v)

            anomaly = {n: x for n, x in detectors.items() if getattr(x, "kind", "anomaly") == "anomaly"}
            drift = {n: x for n, x in detectors.items() if getattr(x, "kind", "anomaly") == "drift"}
            if anomaly:
                yield from self._vote(feature, t, v, anomaly, self.min_votes, "anomaly")
            for name, det in drift.items():
                yield from self._vote(feature, t, v, {name: det}, 1, "drift")

    def _vote(self, feature, t, v, detectors, min_votes, kind):
        if min_votes > len(detectors):
            raise ValueError(f"min_votes={min_votes} but only {len(detectors)} detector(s) "
                             f"for '{feature}': no alert could ever fire")
        names = list(detectors)
        ratio = np.vstack([score_series(detectors[n], v, t) / detectors[n].threshold
                           for n in names])                    # detectors x hours
        strength = np.sort(ratio, axis=0)[-min_votes]          # score that min_votes detectors reached
        flagged = strength > 1.0
        if not flagged.any():
            return
        if kind == "drift":
            self.drift_hours += int(flagged.sum())
        else:
            self.flagged_hours += int(flagged.sum())

        idx = np.flatnonzero(flagged)
        gap = self.drift_gap if kind == "drift" else self.max_gap
        starts_new_run = np.r_[True, (t[idx][1:] - t[idx][:-1]) > gap]
        run_id = np.cumsum(starts_new_run)

        for r in np.unique(run_id):
            m = idx[run_id == r]                               # hours of this run
            peak = m[strength[m].argmax()]
            voters = [n for i, n in enumerate(names) if ratio[i, peak] > 1.0]
            yield Alert(
                detector=f"vote({','.join(voters)})",
                source=self.source,
                dataset=self.dataset,
                kind=kind,
                feature=feature,
                score=float(strength[peak]),
                threshold=1.0,
                timestamp=t[m[0]],
                value=float(v[peak]),
                details={
                    "voters": voters,
                    "start": str(t[m[0]]),
                    "end": str(t[m[-1]]),
                    "n_points": int(len(m)),
                    "peak_raw_scores": {n: float(ratio[i, peak] * detectors[n].threshold)
                                        for i, n in enumerate(names)},
                },
            )
