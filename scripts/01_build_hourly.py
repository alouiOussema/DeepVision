# scripts/01_build_hourly.py
import pandas as pd
from config import settings
from ingest import nyc, hdfs, travis

SOURCES = [
    ("nyc",    lambda: nyc.build_hourly()),
    ("hdfs",   lambda: hdfs.build_hourly()),
    ("travis", lambda: travis.build_hourly()),
]


def main():
    frames = []

    for name, builder in SOURCES:
        cache = settings.HOURLY_DIR / f"{name}.parquet"

        if cache.exists():
            print(f"{name}: cached — loading {cache.name}")
            frames.append(pd.read_parquet(cache))
            continue

        print(f"{name}: building ...")
        df = builder()
        df.to_parquet(cache, index=False)
        print(f"  → {len(df)} rows cached to {cache.name}")
        frames.append(df)

    unified = (pd.concat(frames, ignore_index=True)
                 .sort_values(["source", "timestamp"])
                 .reset_index(drop=True))
    unified.to_parquet(settings.UNIFIED_FILE, index=False)
    print(f"\nUnified → {settings.UNIFIED_FILE} ({len(unified)} rows)")


if __name__ == "__main__":
    main()