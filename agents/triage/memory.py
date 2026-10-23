"""The agent's memory: human feedback on past incidents, kept between runs.

When an operator closes an incident ("false alarm" or "real problem"), the verdict is saved
under a KEY that describes what kind of problem it was. Next time a similar incident appears,
the agent finds that feedback with its check_memory tool.

    logs   -> "logs:E2+E3+E4"          (same event pattern = same kind of failure)
    ci     -> "ci:zendesk/samson"       (same project)
    series -> "series:nyc:avg_fare"     (same dataset + feature)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import paths
from common.schemas import Incident

VERDICTS = {"false_alarm", "real"}


def key_for(inc: Incident) -> str:
    if inc.source == "logs":
        return "logs:" + "+".join(sorted(e["event"] for e in inc.alerts[0].details.get("top_events", [])))
    if inc.source == "ci":
        return "ci:" + str(inc.alerts[0].details.get("project", "?"))
    return f"series:{inc.dataset}:{'+'.join(inc.features)}"


class Memory:
    def __init__(self, path: str | Path = paths.MEMORY):
        self.path = Path(path)
        self.data: dict = json.loads(self.path.read_text()) if self.path.exists() else {}

    def lookup(self, key: str) -> dict | None:
        return self.data.get(key)

    def add_feedback(self, key: str, verdict: str, note: str = "") -> None:
        """Called when a human closes an incident. Counts repeated verdicts."""
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        old = self.data.get(key, {})
        counts = old.get("counts", {"false_alarm": 0, "real": 0})
        counts[verdict] += 1
        self.data[key] = {"verdict": max(counts, key=counts.get), "counts": counts, "note": note or old.get("note", ""),
                          "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def clear(self) -> None:
        self.data = {}