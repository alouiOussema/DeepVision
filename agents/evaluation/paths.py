"""File paths for the Evaluation layer: the ONLY place a path is written."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INJECTIONS_DIR = ROOT / "data" / "injections"

DATA_DIR = ROOT / "data" / "evaluation"
RUN_DIR = DATA_DIR / "latest"
DEMO_DIR = DATA_DIR / "demo"

RESULTS = "results.json"
TRACE = "trace.jsonl"
REPORTS = "reports"
AGENT_FILES = "agents"
