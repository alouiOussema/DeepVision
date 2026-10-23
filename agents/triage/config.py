"""All Triage limits in one place.

Starting values chosen from explore.py on P1's real alerts (23 Oct).
Tune them later with the sweep in evaluate.py and report why you kept them.
"""
from dataclasses import dataclass, field


@dataclass
class TriageConfig:
    # ---------------- A. series (nyc, deploy) ----------------
    # Two episodes of the same dataset are ONE incident if they overlap in time,
    # or if the second starts at most this many hours after the first ends.
    series_gap_hours: dict = field(default_factory=lambda: {"nyc": 1.0, "deploy": 0.25})

    # Severity from strength = score / threshold ("ratio")
    critical_ratio: float = 4.0
    high_ratio: float = 2.5
    medium_ratio: float = 1.5
    noise_ratio: float = 1.3          # below this, a lone short episode is noise (60% of series alerts!)

    # ---------------- B. logs (hdfs blocks) ----------------
    # Blocks with the same top-3 events failed the same way -> one incident per pattern.
    logs_incident_blocks: int = 10    # a pattern seen on at least this many blocks is a real problem
    logs_noise_ratio: float = 2.0     # a rare pattern this weak is noise; stronger -> incident

    # ---------------- C. ci (travis failed builds) ----------------
    # Failed builds of the same project on the same day -> one incident.
    # Careful: here a HIGH score means "probably flaky" (re-run), not "more serious".
    ci_flaky_ratio: float = 3.0       # ratio above this = likely flaky test
    ci_watch_builds: int = 2          # 2 real failures in a day: keep an eye on it
    ci_incident_builds: int = 3       # 3+ real failures in a day: the project is broken ...
    ci_usually_green: float = 0.2     # ... but only news if it usually passes (recent fail rate before today < 20%)