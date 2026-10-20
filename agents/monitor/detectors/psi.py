"""PSI (Population Stability Index): has the DISTRIBUTION of a feature changed?

Idea: split the reference period (training) into 10 equally populated bins. Then look at the
last 7 days: what share of them falls in each bin? If the shares are about 10% each, nothing
changed. If they pile up in a few bins, the data drifted.

    PSI = sum over bins of (share_now - share_ref) * ln(share_now / share_ref)

Rule of thumb: < 0.10 stable, 0.10-0.25 some change, > 0.25 clear drift.
The window is 7 full days, so the daily rhythm is the same in the window and the reference
(otherwise every night would look like drift).
"""
import numpy as np


class PSIDrift:
    name = "psi"
    kind = "drift"                 # the Monitor treats drift separately from anomalies
    needs_timestamps = True

    def __init__(self, threshold=0.25, n_bins=10, window=168, min_fraction=0.8):
        self.threshold = threshold
        self.n_bins = n_bins
        self.window = window            # rows (hours) in the sliding window
        self.min_fraction = min_fraction
        self.eps = 1e-4

    def _bin(self, x):
        return np.searchsorted(self.inner_, x, side="right")      # 0 .. n_bins-1

    def fit(self, values, timestamps=None):
        ref = np.asarray(values, float)
        ref = ref[~np.isnan(ref)]
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, self.n_bins + 1)))
        self.inner_ = edges[1:-1]                                  # outer edges are open
        counts = np.bincount(self._bin(ref), minlength=len(self.inner_) + 1)
        self.ref_p_ = np.clip(counts / counts.sum(), self.eps, None)
        self.ref_p_ /= self.ref_p_.sum()
        return self

    def score(self, values, timestamps=None) -> np.ndarray:
        """PSI of the last `window` rows, for every row. The first window-1 rows get 0
        (not enough history yet)."""
        x = np.asarray(values, float)
        n, k, w = len(x), len(self.ref_p_), self.window
        out = np.zeros(n)
        if n < w:
            return out
        onehot = np.zeros((n + 1, k))
        ok = np.flatnonzero(~np.isnan(x))
        onehot[ok + 1, self._bin(x[ok])] = 1
        cum = onehot.cumsum(axis=0)
        counts = cum[w:] - cum[:-w]                                # window ending at row j-1
        total = counts.sum(axis=1)
        p = np.clip(counts / np.maximum(total, 1)[:, None], self.eps, None)
        p /= p.sum(axis=1, keepdims=True)
        psi = ((p - self.ref_p_) * np.log(p / self.ref_p_)).sum(axis=1)
        out[w - 1:] = np.where(total >= self.min_fraction * w, psi, 0.0)
        return out

    def calibrate(self, values, timestamps=None, target_rate=0.01):
        """Never go below the rule-of-thumb limit, but if normal weeks of the training year
        already differ from the whole year more than that, raise the limit to match."""
        s = self.score(values)[self.window - 1:]
        if len(s):
            self.threshold = max(self.threshold, float(np.quantile(s, 1 - target_rate)))
        return self
