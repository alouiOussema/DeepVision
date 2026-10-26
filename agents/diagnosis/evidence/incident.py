"""What Triage already measured: kept as the first piece of evidence (no cause supported)."""
from __future__ import annotations

import math

from common.schemas import Incident


def collect(inc: Incident, f) -> None:
    for s in ("ci", "logs", "metrics", "model"):
        f.features[f"src_{s}"] = float(inc.source == s)
    f.features.update(
        kind_drift=float(inc.kind == "drift"),
        n_features=float(len(inc.features)),
        log_peak=math.log1p(max(inc.peak_ratio, 0.0)),
        log_duration_h=math.log1p(max(inc.duration_hours, 0.0)),
        max_voters=float(inc.max_voters),
        log_n_alerts=math.log1p(len(inc.alert_ids)),
    )
    f.add("triage_summary", f"Triage ({inc.severity.value}): {inc.summary}", "alert",
          features=inc.features, peak_ratio=inc.peak_ratio, duration_hours=inc.duration_hours)
