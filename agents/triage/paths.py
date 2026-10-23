"""Where Triage reads and writes files: the ONLY place a path is written.

The same code works in two places:
  - your own project : P1's files copied by hand into data/alerts.csv and data/triage_eval/
  - the team repo    : P1's scripts write them to logs/alerts.csv (04_run_monitor)
                       and logs/triage_eval/ (07_export_for_triage)
The first location that exists is used. Triage's OWN files always go to data/triage/.
"""
from pathlib import Path


def _first(*candidates: str) -> Path:
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[-1])


# --- inputs, produced by P1 (Monitor)
ALERTS = _first("logs/alerts.csv", "data/alerts.csv")
EVAL = _first("logs/triage_eval", "data/triage_eval")

# --- Triage's own files
OWN = Path("data/triage")
MEMORY = OWN / "memory.json"              # operator feedback used by the pipeline
APP_MEMORY = OWN / "app_memory.json"      # feedback given in the playground app
LLM_RESULTS = OWN / "llm_eval_results.json"