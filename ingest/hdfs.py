"""HDFS event matrix: one row per BLOCK with a real label (Success / Fail)."""
import numpy as np
import pandas as pd
from config import settings


def load_event_matrix():
    """Returns block_ids, X (blocks x 29 event counts), y (1 = failing block), event_names."""
    df = pd.read_csv(settings.HDFS_EVENT_FILE)
    event_cols = [c for c in df.columns if c.startswith("E") and c[1:].isdigit()]
    label = df["Label"].astype(str).str.strip().str.lower()
    y = label.isin(["fail", "anomaly"]).to_numpy().astype(int)
    print(f"labels in file: {df['Label'].value_counts().to_dict()}")
    return df["BlockId"].to_numpy(), df[event_cols].to_numpy(float), y, event_cols


def split_blocks(n, seed=0):
    """Same random half for every script: True = used for training."""
    return np.random.default_rng(seed).random(n) < 0.5
