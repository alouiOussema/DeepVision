"""Step 1: build the tables the Monitor learns from.

    python -m scripts.01_build_data           # builds only what is missing
    python -m scripts.01_build_data --force   # rebuilds everything (do this after changing a loader)

  NYC taxi files        ->  data/hourly/nyc.parquet   one row per HOUR
  TravisTorrent csv     ->  data/builds.parquet       one row per CI BUILD (with flaky label)
  Deploy metrics        ->  data/deploy/*.parquet     one row per MINUTE + the list of deploys (simulation)
  HDFS event matrix     ->  nothing to build, only checked (one row per BLOCK)
"""
import sys
from config import settings
from ingest import deploy, nyc, travis
from ingest.hdfs import load_event_matrix

FORCE = "--force" in sys.argv


def main():
    if settings.NYC_HOURLY_FILE.exists() and not FORCE:
        print(f"nyc hourly: already built ({settings.NYC_HOURLY_FILE.name}), use --force to rebuild")
    else:
        df = nyc.build_hourly()
        df.to_parquet(settings.NYC_HOURLY_FILE, index=False)
        print(f"nyc hourly: {len(df):,} hours -> {settings.NYC_HOURLY_FILE}\n")

    if settings.BUILDS_FILE.exists() and not FORCE:
        print(f"ci builds: already built ({settings.BUILDS_FILE.name}), use --force to rebuild")
    else:
        b = travis.build_builds()
        b.to_parquet(settings.BUILDS_FILE, index=False)
        n_fail = int(b["label"].isin(["flaky_fail", "real_fail"]).sum())
        n_flaky = int((b["label"] == "flaky_fail").sum())
        print(f"ci builds: {len(b):,} builds, {n_fail:,} failed, {n_flaky:,} flaky "
              f"({n_flaky / max(n_fail, 1):.1%}) -> {settings.BUILDS_FILE}\n")

    if settings.DEPLOY_METRICS_FILE.exists() and not FORCE:
        print(f"deploy metrics: already built ({settings.DEPLOY_METRICS_FILE.name}), use --force to rebuild")
    else:
        metrics, deploys = deploy.simulate()
        metrics.to_parquet(settings.DEPLOY_METRICS_FILE, index=False)
        deploys.to_parquet(settings.DEPLOY_EVENTS_FILE, index=False)
        print(f"deploy metrics (simulated): {len(metrics):,} minutes, {len(deploys)} deploys, "
              f"{int(deploys['bad'].sum())} bad -> {settings.DEPLOY_DIR}\n")

    ids, X, y, _ = load_event_matrix()
    print(f"hdfs blocks: {len(y):,} blocks, {int(y.sum()):,} failing (read directly from the event matrix)")


if __name__ == "__main__":
    main()
