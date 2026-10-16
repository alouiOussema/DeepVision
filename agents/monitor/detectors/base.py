class Detector:
    name = "base"

    def fit(self, values):        # values: 1D numpy array
        raise NotImplementedError

    def score(self, x: float) -> float:
        raise NotImplementedError

    def is_anomaly(self, x: float) -> bool:
        return self.score(x) > self.threshold