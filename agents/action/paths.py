"""File paths for the Action agent (P5): the ONLY place a path is written."""
from pathlib import Path

from agents.decision import paths as decision_paths

ROOT = Path(__file__).resolve().parents[2]

# --- what P5 reads and reports to (owned by Decision, P4)
ACTION_INBOX = decision_paths.ACTION_INBOX           # orders, with their status
DECISION_LOG = decision_paths.DECISION_LOG           # P5's steps are written there too (one trail for the deck)
DECISION_HISTORY = decision_paths.DECISION_HISTORY

# --- Action's own files (not in git)
DATA_DIR = ROOT / "data" / "action"
CLAIMS_DIR = DATA_DIR / "claims"                     # one file per order ever started: never execute twice
NOTIFICATIONS = DATA_DIR / "notifications.jsonl"     # what a human was told (simulated channel = this file)
DEPLOY_REQUESTS = DATA_DIR / "deploy_requests.jsonl" # retrained models waiting for a human to approve the deploy
CHALLENGERS_DIR = DATA_DIR / "challengers"           # retrained models (never copied over the live one)

# --- optional input of the real-ish retrain backend (P1's 01_build_data writes it; not in git)
NYC_HOURLY = ROOT / "data" / "hourly" / "nyc.parquet"

# --- demo files (python -m agents.action.demo): kept apart so a demo never touches real orders
DEMO_DIR = DATA_DIR / "demo"
DEMO_INBOX = DEMO_DIR / "action_inbox.json"
DEMO_LOG = DEMO_DIR / "decision_log.jsonl"
DEMO_HISTORY = DEMO_DIR / "decision_history.jsonl"
DEMO_CLAIMS = DEMO_DIR / "claims"
DEMO_NOTIFICATIONS = DEMO_DIR / "notifications.jsonl"
DEMO_DEPLOY_REQUESTS = DEMO_DIR / "deploy_requests.jsonl"
