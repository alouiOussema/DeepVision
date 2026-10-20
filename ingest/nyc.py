"""NYC taxi -> one row per hour.

Fixes compared with the old loader:
  * each monthly file is cut to ITS OWN month: stray trips stamped 2001, 2008, ... are dropped
  * hours are SUMMED across files, so an hour is never listed twice
  * trip_count counts every valid trip, including trips with an empty passenger_count
    (those are common in recent data and were silently removed before)
  * avg_passenger only averages trips where the count is known

NOTE: TLC timestamps are New York WALL-CLOCK time (no time zone). We label them UTC only to
keep pandas happy. Do not convert time zones for NYC (see settings.TIMEZONES).
"""
import re
import pandas as pd
from config import settings

COLUMNS = ["tpep_pickup_datetime", "fare_amount", "trip_distance", "passenger_count"]
MONTH_RE = re.compile(r"(\d{4})-(\d{2})")


def _month_bounds(path):
    m = MONTH_RE.search(path.stem)             # e.g. yellow_tripdata_2025-03.parquet
    if not m:
        return None
    start = pd.Timestamp(year=int(m[1]), month=int(m[2]), day=1, tz="UTC")
    return start, start + pd.DateOffset(months=1)


def build_hourly() -> pd.DataFrame:
    files = sorted(settings.NYC_RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {settings.NYC_RAW_DIR}")

    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=COLUMNS)
        n_rows = len(df)
        df["timestamp"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])

        bounds = _month_bounds(f)
        if bounds:
            df = df[(df.timestamp >= bounds[0]) & (df.timestamp < bounds[1])]
        n_month = len(df)

        # obvious garbage out. passenger_count is NOT required here.
        ok = (df.fare_amount.gt(0) & df.fare_amount.lt(500) &
              df.trip_distance.gt(0) & df.trip_distance.lt(100))
        df = df[ok]
        print(f"{f.name}: {n_rows:,} rows, {n_rows - n_month:,} outside the month, "
              f"{n_month - len(df):,} invalid -> {len(df):,} kept")

        pax = df["passenger_count"].where(df["passenger_count"] > 0)   # NaN if empty or 0
        df = df.assign(pax=pax, has_pax=pax.notna().astype(int),
                       hour=df["timestamp"].dt.floor("h"))
        parts.append(df.groupby("hour").agg(
            trip_count=("fare_amount", "size"),
            fare_sum=("fare_amount", "sum"),
            dist_sum=("trip_distance", "sum"),
            pax_sum=("pax", "sum"),
            pax_n=("has_pax", "sum"),
        ))

    out = pd.concat(parts).groupby(level=0).sum()          # safety: one row per hour
    out = out.reset_index().rename(columns={"hour": "timestamp"})
    out["avg_fare"] = out["fare_sum"] / out["trip_count"]
    out["avg_distance"] = out["dist_sum"] / out["trip_count"]
    out["avg_passenger"] = out["pax_sum"] / out["pax_n"].where(out["pax_n"] > 0)
    out = out[["timestamp", "trip_count", "avg_fare", "avg_distance", "avg_passenger"]]
    out["source"] = "nyc"
    return out.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    df = build_hourly()
    path = settings.HOURLY_DIR / "nyc.parquet"
    df.to_parquet(path, index=False)
    print(f"Wrote {len(df)} hourly rows -> {path}")
