from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# raw inputs — point these at your actual files
NYC_RAW_DIR      = ROOT / "data" / "raw" / "nyc"
TRAVIS_RAW_DIR   = ROOT / "data" / "raw" / "travis"
HDFS_LOG_FILE    = ROOT / "data" / "raw" / "hdfs" / "HDFS.log"
HDFS_LABEL_FILE  = ROOT / "data" / "raw" / "hdfs"  / "anomaly_label.csv"
HDFS_TEMPLATE_FILE = ROOT / "data" / "raw" / "hdfs"  / "HDFS.log_templates.csv"

# outputs
HOURLY_DIR   = ROOT / "data" / "hourly"
UNIFIED_FILE = ROOT / "data" / "unified.parquet"
MODELS_DIR   = ROOT / "models"
LOGS_DIR     = ROOT / "logs"
REPORTS_DIR  = ROOT / "reports"
ALERTS_CSV   = LOGS_DIR / "alerts.csv"

for d in (HOURLY_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# detection
ZSCORE_THRESHOLD = 2.0        # start here, tune on validation
TARGET_ALERT_RATE = 0.01      # aim for ~1% of hours

# time split
TRAIN_END   = "2026-01-01"
VAL_END     = "2026-04-01"
# everything after VAL_END is test