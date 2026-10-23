"""Load / save alerts, so the same code runs on mock data now and on P1's file later."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common.schemas import Alert


def load_alerts(path: str | Path) -> list[Alert]:
    """Read P1's alerts from .json (list of dicts), .jsonl or .csv."""
    path = Path(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".csv":
        rows = pd.read_csv(path).to_dict("records")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    alerts = [Alert.from_dict(r) for r in rows]
    return sorted(alerts, key=lambda a: (a.timestamp, a.source, a.feature))


def save_alerts(alerts: list[Alert], path: str | Path) -> None:
    Path(path).write_text(json.dumps([a.to_dict() for a in alerts], indent=2))


def to_frame(alerts: list[Alert]) -> pd.DataFrame:
    """One row per alert, with ratio and number of voters added."""
    return pd.DataFrame([{**{k: getattr(a, k) for k in
                             ("id", "detector", "source", "feature", "score", "threshold", "timestamp", "value")},
                          "ratio": a.ratio, "n_voters": len(a.voters),
                          "truth": a.details.get("_truth")} for a in alerts])