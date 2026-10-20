from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------- raw inputs (never edited, never in git) ----------
NYC_RAW_DIR     = ROOT / "data" / "raw" / "nyc"        # monthly parquet files
TRAVIS_RAW_DIR  = ROOT / "data" / "raw" / "travis"     # TravisTorrent csv
HDFS_EVENT_FILE = ROOT / "data" / "raw" / "hdfs" / "Event_occurrence_matrix.csv"   # one row per block (the one we use)
# raw HDFS files, kept for later (rebuilding the matrix, naming the events). Not used by the code now.
HDFS_LOG_FILE      = ROOT / "data" / "raw" / "hdfs" / "HDFS.log"
HDFS_LABEL_FILE    = ROOT / "data" / "raw" / "hdfs" / "anomaly_label.csv"
HDFS_TEMPLATE_FILE = ROOT / "data" / "raw" / "hdfs" / "HDFS.log_templates.csv"

# ---------- tables built from the raw inputs (01_build_data.py) ----------
HOURLY_DIR      = ROOT / "data" / "hourly"
NYC_HOURLY_FILE = HOURLY_DIR / "nyc.parquet"           # one row per hour
# bad deployments: one row per MINUTE of pipeline metrics + the list of deploys (simulation, or real later)
DEPLOY_DIR          = ROOT / "data" / "deploy"
DEPLOY_METRICS_FILE = DEPLOY_DIR / "metrics.parquet"
DEPLOY_EVENTS_FILE  = DEPLOY_DIR / "deploys.parquet"
RCAEVAL_DIR         = ROOT / "data" / "raw" / "rcaeval"    # optional: public benchmark, see ingest/deploy.py

BUILDS_FILE     = ROOT / "data" / "builds.parquet"     # one row per CI build, with flaky label
SERIES_FILES    = {"nyc": NYC_HOURLY_FILE, "deploy": DEPLOY_METRICS_FILE}   # time series the Monitor watches

# ---------- outputs ----------
MODELS_DIR  = ROOT / "models"
LOGS_DIR    = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"
ALERTS_CSV  = LOGS_DIR / "alerts.csv"

for d in (HOURLY_DIR, DEPLOY_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- hourly series: what the Monitor watches ----------
FEATURES = {"nyc": ["trip_count", "avg_fare", "avg_distance"],
            "deploy": ["latency_ms", "error_rate"]}
SPLITS = {"nyc": {"train": ("2025-01-01", "2026-01-01"),
                  "test":  ("2026-01-01", "2026-09-01")},
          "deploy": {"train": ("2026-01-01", "2026-01-15"),     # 14 days with harmless deploys only
                     "test":  ("2026-01-15", "2026-01-29")}}    # 14 days with good AND bad deploys
MIN_TRAIN_ROWS = {"nyc": 100, "deploy": 1000}

# time zone used to read the daily rhythm. NYC timestamps are already New York wall-clock time
# (only labelled UTC), so they must NOT be converted: keep "UTC" = "use the clock as stored".
TIMEZONES = {"nyc": "UTC", "deploy": "UTC"}

# contract source of each dataset: ci / logs / metrics / model
DATASET_KIND = {"nyc": "model", "hdfs": "logs", "travis": "ci", "deploy": "metrics"}

# alerts closer together than this many HOURS are merged into one (hourly data: 1 h; minute data: 15 min)
MERGE_GAP_HOURS = {"nyc": 1, "deploy": 0.25}

# ---------- anomaly detection ----------
ZSCORE_THRESHOLD  = 2.0      # the old global baseline (kept to show the "before")
TARGET_ALERT_RATE = 0.01     # aim for ~1% of hours
DETECTORS         = {"nyc": ["seasonal"], "deploy": ["rolling"]}   # options: seasonal, rolling, zscore
MIN_VOTES         = 1          # with one detector per dataset, one vote is enough (raise it if you add more)
COMPARE_DETECTORS = ["zscore", "seasonal"]                  # what 03_test.py nyc compares

# ---------- data drift (PSI) ----------
# model inputs to watch, and the PSI window (in DAYS) chosen for each from `03_test.py psi`
DRIFT = {"nyc": {"avg_fare": 7, "avg_distance": 14, "avg_passenger": 28}}
PSI_THRESHOLD = 0.25         # rule of thumb: > 0.25 = clear drift
PSI_BINS = 10
PSI_WINDOW_HOURS = 168       # default window (7 full days, so the daily rhythm cancels out)

# ---------- HDFS logs (per block) ----------
HDFS_FALSE_ALARM_RATE = 0.02   # share of NORMAL blocks allowed to alert (chosen from the trade-off table)

# ---------- flaky tests (per CI build) ----------
FLAKY_SPLIT_DATE = "2015-07-01"   # train on failures before, test on failures after

# ---------- what the replay and the node send ----------
LOG_MAX_ALERTS = None   # node: None = send every log alert, or a number to keep only the strangest N
CI_REPLAY_DAYS = 30     # 04_run_monitor: CI alerts for the failed builds of the last N days

# ---------- bad deployments (rolling baseline on minute-level metrics) ----------
ROLLING_WINDOW  = 60     # minutes of "before" picture
ROLLING_SMOOTH  = 5      # minutes averaged to describe "now"
ROLLING_TARGET_RATE = 0.001   # share of normal MINUTES allowed to alert (0.1% = about 1 a day per metric)
DEPLOY_CHECK_MIN = 30    # an alert within this many minutes after a deploy is credited to that deploy

# RCAEval cases are short (about 71 minutes): shorter memory, and a few fixed limits to compare
RCAEVAL_WINDOW = 24     # points of 5 seconds = 2 minutes of memory
RCAEVAL_SMOOTH = 3      # 15 seconds
RCAEVAL_LIMITS = (4, 6, 8)
