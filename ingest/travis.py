import pandas as pd
from config import settings

TIME_CANDIDATES = [
    "gh_build_started_at",   # 2017 file
    "tr_started_at",         # 2016 file
    "build_started_at",
]
STATUS_CANDIDATES   = ["tr_status", "build_status"]
DURATION_CANDIDATES = ["tr_duration", "build_duration"]

VALID_STATUS = {"passed", "failed", "errored"}


def _pick(available, candidates):
    for c in candidates:
        if c in available:
            return c
    return None


def build_hourly() -> pd.DataFrame:
    files = sorted(settings.TRAVIS_RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV in {settings.TRAVIS_RAW_DIR}")

    parts = []
    for f in files:
        header = pd.read_csv(f, nrows=0).columns.tolist()

        time_col     = _pick(header, TIME_CANDIDATES)
        status_col   = _pick(header, STATUS_CANDIDATES)
        duration_col = _pick(header, DURATION_CANDIDATES)

        print(f"\n{f.name}:")
        print(f"  time     = {time_col}")
        print(f"  status   = {status_col}")
        print(f"  duration = {duration_col}")

        if time_col is None or status_col is None:
            print("  → SKIP (missing required column)")
            continue

        usecols = [c for c in [time_col, status_col, duration_col] if c]
        df = pd.read_csv(f, low_memory=False, usecols=usecols)

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.dropna(subset=[time_col, status_col])
        df = df[df[status_col].isin(VALID_STATUS)]

        if duration_col:
            df = df[(df[duration_col] > 0) & (df[duration_col] < 86400)]

        if len(df) == 0:
            print("  → 0 rows after cleaning — skip")
            continue

        df["hour"] = df[time_col].dt.floor("h")
        df["is_fail"] = (df[status_col] != "passed").astype(int)

        agg_spec = {
            "build_count":  ("is_fail", "size"),
            "failure_rate": ("is_fail", "mean"),
        }
        if duration_col:
            agg_spec["avg_duration"] = (duration_col, "mean")

        hourly = (df.groupby("hour")
                    .agg(**agg_spec)
                    .reset_index()
                    .rename(columns={"hour": "timestamp"}))
        hourly["source"] = "travis"
        parts.append(hourly)
        print(f"  → {len(hourly)} hours")

    if not parts:
        raise RuntimeError("No Travis file produced rows.")

    out = (pd.concat(parts, ignore_index=True)
             .sort_values("timestamp")
             .reset_index(drop=True))
    return out


if __name__ == "__main__":
    df = build_hourly()
    path = settings.HOURLY_DIR / "travis.parquet"
    df.to_parquet(path, index=False)
    print(f"\nWrote {len(df)} hourly rows → {path}")