import pandas as pd
from pathlib import Path
from config import settings

def build_hourly() -> pd.DataFrame:
    """Read all parquet files in NYC_RAW_DIR → one row per hour."""
    files = sorted(settings.NYC_RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {settings.NYC_RAW_DIR}")

    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=[
            "tpep_pickup_datetime", "fare_amount", "trip_distance" , "passenger_count"  # fare_amout = prix sur le compteur / 
        ])
        df = df.rename(columns={"tpep_pickup_datetime": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])

        # drop obvious garbage (your cleaning rules)
        df = df[
            (df.fare_amount > 0) &                    # remove negative and zero
            (df.fare_amount < 500) &                  # remove extreme outliers
            (df.trip_distance > 0) &                   # remove zero-distance
            (df.trip_distance < 100) &               # NYC rarely exceeds 100 miles
            (df.passenger_count > 0)               # remove zero-passenger records
            
        ]

        # floor to hour
        df["hour"] = df["timestamp"].dt.floor("h")

        hourly = df.groupby("hour").agg(
            trip_count=("fare_amount", "size"),
            avg_fare=("fare_amount", "mean"),
            avg_distance=("trip_distance", "mean"),
            avg_passenger=("passenger_count","mean")
        ).reset_index().rename(columns={"hour": "timestamp"})

        parts.append(hourly)

    out = pd.concat(parts).sort_values("timestamp").reset_index(drop=True)
    out["source"] = "nyc"
    return out

if __name__ == "__main__":
    df = build_hourly()
    path = settings.HOURLY_DIR / "nyc.parquet"
    df.to_parquet(path, index=False)
    print(f"Wrote {len(df)} hourly rows → {path}")