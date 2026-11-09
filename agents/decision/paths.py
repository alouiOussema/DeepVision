"""File paths for the Decision agent (P4): the ONLY place a path is written."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- shared knowledge base (in git)
KB = ROOT / "knowledge_base"
RUNBOOKS = KB / "runbooks"
PAST_INCIDENTS = KB / "incidents"
ACTION_RISK_TABLE = KB / "action_risk_table.md"       # the policy (policy.py reads it)

# --- Decision's own files (not in git)
DATA_DIR = ROOT / "data" / "decision"
LOGS_DIR = ROOT / "logs"

DECISION_LOG = LOGS_DIR / "decision_log.jsonl"        # every step: data used, reason, action, result (slide 10)
ACTION_INBOX = DATA_DIR / "action_inbox.json"         # P4 -> P5: orders, with their status
DECISION_HISTORY = DATA_DIR / "decision_history.jsonl"    # actions granted (guards: rerun budget, cooldown, rate limit)
EVAL_RESULTS = DATA_DIR / "eval_results.json"

# --- demo files (python -m agents.decision.demo): kept apart so a demo never touches real orders
DEMO_DIR = DATA_DIR / "demo"
DEMO_INBOX = DEMO_DIR / "action_inbox.json"
DEMO_LOG = DEMO_DIR / "decision_log.jsonl"
DEMO_HISTORY = DEMO_DIR / "decision_history.jsonl"
