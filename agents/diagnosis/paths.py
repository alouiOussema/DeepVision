"""Where Diagnosis reads and writes files: the ONLY place a path is written."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- the knowledge base (in git): runbooks + past incidents + the action risk table
KB = ROOT / "knowledge_base"
RUNBOOKS = KB / "runbooks"
PAST_INCIDENTS = KB / "incidents"

# --- Diagnosis's own files (not in git)
OWN = ROOT / "data" / "diagnosis"
BENCHMARK = OWN / "benchmark.pkl"            # labelled incidents built by scenarios.py
CLASSIFIER = OWN / "classifier.pkl"          # root-cause classifier trained by train.py
EMBEDDINGS = OWN / "kb_embeddings.json"      # cached embeddings of the knowledge base
INBOX = OWN / "inbox.json"                   # diagnoses waiting for the approval page
REVIEWS = OWN / "reviews.jsonl"              # what humans said on the approval page
RESULTS = OWN / "eval_results.json"

# --- inputs from the other agents, when they exist (same convention as agents/triage/paths.py)
TRIAGE_EVAL = next((p for p in (ROOT / "logs" / "triage_eval", ROOT / "data" / "triage_eval") if p.exists()),
                   ROOT / "data" / "triage_eval")
HDFS_TEMPLATES = ROOT / "data" / "raw" / "hdfs" / "HDFS.log_templates.csv"
