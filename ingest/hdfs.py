import re
import pandas as pd
import numpy as np
from config import settings

# matches: 081109 203615 148 INFO dfs.DataNode$PacketResponder: ...
LINE_RE = re.compile(r"^(\d{6})\s+(\d{6})\s+\d+\s+(\w+)")

# matches: blk_-1608999687919862906  (with optional minus sign)
BLOCK_RE = re.compile(r"(blk_-?\d+)")


def parse_timestamp(date_s: str, time_s: str) -> pd.Timestamp:
    return pd.to_datetime(
        date_s + time_s, format="%y%m%d%H%M%S", utc=True
    )


def build_hourly() -> pd.DataFrame:
    """One row per hour with volume / error / warn counts."""
    rows = []
    with open(settings.HDFS_LOG_FILE, "r", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            rows.append({
                "timestamp": parse_timestamp(m.group(1), m.group(2)),
                "level": m.group(3),
            })

    df = pd.DataFrame(rows)
    df["hour"] = df["timestamp"].dt.floor("h")

    hourly = df.groupby("hour").agg(
        log_volume=("level", "size"),
        error_count=("level", lambda s: (s == "ERROR").sum()),
        warn_count=("level", lambda s: (s == "WARN").sum()),
    ).reset_index().rename(columns={"hour": "timestamp"})

    hourly["source"] = "hdfs"
    return hourly


def load_labels() -> pd.DataFrame:
    """Block-level labels. Returns block_id + is_anomaly (0/1)."""
    df = pd.read_csv(settings.HDFS_LABEL_FILE)
    df.columns = [c.strip() for c in df.columns]
    if "BlockId" in df.columns:
        df = df.rename(columns={"BlockId": "block_id", "Label": "label"})
    else:
        df.columns = ["block_id", "label"]

    df["is_anomaly"] = (
        df["label"].astype(str).str.strip().str.lower() == "anomaly"
    ).astype(int)
    return df[["block_id", "is_anomaly"]]


def build_labels_hourly() -> pd.DataFrame:
    """
    For each hour: count distinct blocks + anomalous blocks.
    Flag an hour as anomalous if its anomaly_ratio is
    statistically high (mean + 2*std across all hours).
    """
    
    cache = settings.HOURLY_DIR / "hdfs_block_pairs.parquet"

    if cache.exists():
        print(f"Loading cached block pairs from {cache.name}")
        pdf = pd.read_parquet(cache)
    else:
        print("Parsing HDFS.log for block pairs (this takes a while) ...")
        labels = load_labels()
        block_to_anom = dict(zip(labels["block_id"], labels["is_anomaly"]))

        seen = set()
        pairs = []
        with open(settings.HDFS_LOG_FILE, "r", errors="ignore") as f:
            for line in f:
                m = LINE_RE.match(line)
                if not m:
                    continue
                bm = BLOCK_RE.search(line)
                if not bm:
                    continue
                hour = parse_timestamp(m.group(1), m.group(2)).floor("h")
                block_id = bm.group(1)
                key = (hour, block_id)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({
                    "timestamp": hour,
                    "block_id": block_id,
                    "is_anom": block_to_anom.get(block_id, 0),
                })
        pdf = pd.DataFrame(pairs)
        pdf.to_parquet(cache, index=False)
        print(f"Cached block pairs to {cache.name}")

    hourly = pdf.groupby("timestamp").agg(
        n_blocks=("block_id", "nunique"),
        n_anomalous_blocks=("is_anom", "sum"),
    ).reset_index()

    hourly["anomaly_ratio"] = (
        hourly["n_anomalous_blocks"] / hourly["n_blocks"]
    )

    N_ANOMALOUS_HOURS = 5
    threshold = float(
        hourly["anomaly_ratio"].nlargest(N_ANOMALOUS_HOURS).min()
    )
    hourly["is_anomalous_hour"] = (
        hourly["anomaly_ratio"] >= threshold
    ).astype(int)

    print(f"Top-{N_ANOMALOUS_HOURS} threshold: {threshold:.4f}")
    print(f"Flagged hours: "
          f"{int(hourly['is_anomalous_hour'].sum())} / {len(hourly)}")

    return hourly


if __name__ == "__main__":
    print("Building hourly features ...")
    hourly = build_hourly()

    print("Building hour-level labels ...")
    labels = build_labels_hourly()

    path     = settings.HOURLY_DIR / "hdfs.parquet"
    lab_path = settings.HOURLY_DIR / "hdfs_labels.parquet"

    hourly.to_parquet(path, index=False)
    labels.to_parquet(lab_path, index=False)

    print(f"\nWrote {len(hourly)} hourly rows -> {path}")
    print(f"Wrote {len(labels)} label rows  -> {lab_path}")
    print()
    print("Label distribution per hour:")
    print(labels["is_anomalous_hour"].value_counts())