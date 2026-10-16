import csv
from pathlib import Path
from config import settings

FIELDS = ["timestamp", "detector", "source", "feature",
          "score", "threshold", "value"]

def log_alert(alert, path: Path = settings.ALERTS_CSV):
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(alert.to_row())