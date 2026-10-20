"""PCA detector for HDFS blocks.

It learns, from NORMAL blocks only, which combinations of the 29 event counts usually go
together. A block whose events do not fit those patterns has a large reconstruction error.
score = that error; threshold = the error that only `false_alarm_rate` of normal blocks exceed.
"""
import numpy as np
from sklearn.decomposition import PCA


class BlockPCA:
    name = "block_pca"
    kind = "anomaly"

    def __init__(self, variance=0.95, false_alarm_rate=0.01, seed=0):
        self.variance = variance
        self.false_alarm_rate = false_alarm_rate
        self.seed = seed
        self.threshold = None

    @staticmethod
    def _prep(X):
        return np.log1p(np.asarray(X, float))

    def residual_sq(self, X):
        A = self._prep(X)
        return (A - self.pca_.inverse_transform(self.pca_.transform(A))) ** 2   # per event

    def score(self, X):
        return self.residual_sq(X).sum(axis=1)

    def fit(self, X_normal):
        self.pca_ = PCA(n_components=self.variance, random_state=self.seed).fit(self._prep(X_normal))
        s = self.score(X_normal)
        self.threshold = max(float(np.quantile(s, 1 - self.false_alarm_rate)), 1e-6)
        return self
