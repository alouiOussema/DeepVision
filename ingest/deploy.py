"""Bad deployments: minute-level pipeline metrics + the list of deploys.

No real deployment data exists yet (P5 will provide it), so there are two sources:

  simulate()      a made-up pipeline with daily traffic rhythm, harmless deploys and BAD deploys.
                  The truth is known, so recall and false alarms can be measured.
  load_rcaeval()  one case of the public RCAEval benchmark (a fault injected at a known time).
                  Download: https://github.com/phamquiluan/RCAEval  -> put the case folders in
                  data/raw/rcaeval/. Each case folder holds metrics + inject_time.txt.

Both give the same shape as every other source: a table with `timestamp` plus signal columns.
Real data from P5 only has to be turned into that shape (timestamp, latency_ms, error_rate).
"""
import numpy as np
import pandas as pd
from config import settings


def simulate(start="2026-01-01", days=28, seed=0):
    """Returns (metrics, deploys).
    metrics: timestamp (1 row per minute), latency_ms, error_rate
    deploys: timestamp, kind, bad (True/False), size (how much worse, in % of normal)
    The first 14 days only have harmless deploys (training), the next days also have bad ones."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=days * 1440, freq="min", tz="UTC")
    n = len(t)
    day = 2 * np.pi * (t.hour * 60 + t.minute).to_numpy() / 1440
    latency = 120 * (1 + 0.15 * np.sin(day - 2))                  # slower at peak hours
    errors = 0.004 * (1 + 0.30 * np.sin(day - 2))
    noise = np.zeros(n)                                           # noise that lasts a few minutes
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 0.03)
    latency = latency * (1 + noise)
    errors = np.abs(errors * (1 + 4 * noise) + rng.normal(0, 0.0004, n))

    deploys = []
    train_end = 14 * 1440
    pos = 180
    while pos < n - 240:
        pos += int(rng.integers(180, 540))                         # a deploy every 3-9 hours
        if pos >= n - 240:
            break
        bad = pos >= train_end and rng.random() < 0.4
        if bad:
            kind = str(rng.choice(["slow", "errors", "leak"]))
            size = int(rng.choice([20, 40, 80]))
            dur = 120                                              # a person rolls it back after 2 hours
            w = slice(pos, pos + dur)
            ramp = np.linspace(0, 1, dur) if kind == "leak" else np.ones(dur)
            if kind in ("slow", "leak"):
                latency[w] *= 1 + size / 100 * ramp
            if kind in ("errors", "leak"):
                errors[w] *= 1 + 3 * size / 100 * ramp
        else:
            kind, size = "harmless", 0
            latency[pos:pos + 3] *= 1.12                           # short restart blip: must NOT alert
        deploys.append({"timestamp": t[pos], "kind": kind, "bad": bad, "size": size})
    metrics = pd.DataFrame({"timestamp": t, "latency_ms": latency, "error_rate": errors})
    return metrics, pd.DataFrame(deploys)


def load_rcaeval(case_dir, step="5s"):
    """One RCAEval case -> (metrics, inject_time).
    Keeps only the latency and error columns, averaged to one point per `step` (the files are 1 second data).
    ASSUMPTIONS (the file layout comes from the project README, check with your copy):
      metrics.parquet or metrics.json with a `time` column (unix seconds or milliseconds),
      inject_time.txt with one unix timestamp."""
    from pathlib import Path
    d = Path(case_dir)
    f = d / "metrics.parquet"
    df = pd.read_parquet(f) if f.exists() else pd.read_json(d / "metrics.json")
    unit = "ms" if df["time"].iloc[0] > 1e11 else "s"
    df["timestamp"] = pd.to_datetime(df["time"], unit=unit, utc=True)
    cols = [c for c in df.columns if any(k in c.lower() for k in ("latency", "error"))]
    out = (df.set_index("timestamp")[cols].astype(float)
             .resample(step).mean().interpolate(limit=5).reset_index())
    inject = pd.to_datetime(int(float((d / "inject_time.txt").read_text().strip())), unit="s", utc=True)
    return out, inject
