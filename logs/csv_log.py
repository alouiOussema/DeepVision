import csv
from config import settings


def reset_log():
    settings.ALERTS_CSV.unlink(missing_ok=True)


def log_alert(alert):
    row = alert.to_row()
    path = settings.ALERTS_CSV
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)