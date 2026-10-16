# monitor/core.py
from .alert import Alert


class Monitor:
    def __init__(self, bundle: dict, source: str, min_votes: int = 2):
        # bundle: {feature: {detector_name: detector}}
        self.bundle = bundle
        self.source = source
        self.min_votes = min_votes

    def run(self, df):
        for _, row in df.iterrows():
            for feature, detectors in self.bundle.items():
                if feature not in df.columns:
                    continue
                value = row[feature]
                if value != value:      # NaN
                    continue

                fired = []
                for name, det in detectors.items():
                    if det.is_anomaly(value):
                        fired.append((name, det.score(value)))

                if len(fired) >= self.min_votes:
                    best_name, best_score = max(fired, key=lambda t: t[1])
                    yield Alert(
                        detector=f"vote({best_name})",
                        source=self.source,
                        feature=feature,
                        score=best_score,
                        threshold=detectors[best_name].threshold,
                        timestamp=row["timestamp"],
                        value=float(value),
                        details={"voters": [n for n, _ in fired]},
                    )