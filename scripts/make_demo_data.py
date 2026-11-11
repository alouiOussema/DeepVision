"""Builds the small files in demo_data/: a sample of each data source and the list of injected faults.

    python -m scripts.make_demo_data

NYC Taxi needs the monthly TLC files in data/raw/nyc/ (see demo_data/README.md for the links).
"""
from datetime import timedelta

import pandas as pd

from agents.evaluation.config import EvalConfig
from agents.evaluation.injector import deploy_episode, episodes, synthetic_cases
from config import settings
from ingest import nyc

OUT = settings.ROOT / "demo_data"
SCENARIO = {"ci": "1 · Flaky test", "deploy": "2 · Bad deploy", "model": "3 · Data drift", "logs": "4 · Unknown failure"}


def save(df: pd.DataFrame, name: str) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.3f")
    print(f"{len(df):6,} rows  {path.relative_to(settings.ROOT)}")


def nyc_taxi() -> None:
    files = sorted(settings.NYC_RAW_DIR.glob("yellow_tripdata_*.parquet"))
    if not files:
        print("no NYC files in data/raw/nyc/: skipped (see demo_data/README.md)")
        return
    cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance", "PULocationID",
            "DOLocationID", "fare_amount", "tip_amount", "total_amount"]
    sample = pd.read_parquet(files[-1], columns=cols).sample(15, random_state=1).sort_values("tpep_pickup_datetime")
    save(sample, "nyc_taxi/raw_trips_sample.csv")

    hourly = nyc.build_hourly().drop(columns=["source"])
    save(hourly, "nyc_taxi/hourly.csv")

    hourly["month"] = hourly["timestamp"].dt.strftime("%Y-%m")
    rows = []
    for f in files:
        month = f.stem.rsplit("_", 1)[-1]
        h = hourly[hourly["month"] == month]
        rows.append({"month": month, "file": f.name, "raw_trips": len(pd.read_parquet(f, columns=["fare_amount"])),
                     "trips_kept": int(h["trip_count"].sum()), "hours": len(h),
                     "avg_fare": (h["avg_fare"] * h["trip_count"]).sum() / h["trip_count"].sum(),
                     "avg_distance": h["avg_distance"].mean()})
    save(pd.DataFrame(rows), "nyc_taxi/monthly_summary.csv")


def ci_and_logs() -> None:
    cases = synthetic_cases()
    ci = [{"timestamp": a.timestamp, "project": a.details["project"], "build_id": a.details["build_id"],
           "commit": a.details["commit"], "flaky_probability": a.details["flaky_probability"],
           "recent_fail_rate": a.details["evidence"]["recent_fail_rate"], "true_cause": c.truth.value}
          for c in cases["ci"][:30] for a in c.incident.alerts]
    save(pd.DataFrame(ci), "ci_builds/builds_example.csv")
    logs = [{"block_id": a.details["block_id"], "events": "+".join(e["event"] for e in a.details["top_events"]),
             "anomaly_score": a.score, "true_cause": c.truth.value, "detail": c.truth_sub}
            for c in cases["logs"][:12] for a in c.incident.alerts[:3]]
    save(pd.DataFrame(logs), "hdfs_logs/blocks_example.csv")


def service() -> None:
    ep = deploy_episode(4)
    inj = next(i for i in ep.injections if i.truth_suspect == "release-064")
    m = ep.context.metrics["deploy"]
    lo, hi = inj.fault_start - timedelta(hours=2), inj.fault_end + timedelta(hours=1)
    save(m[(m["timestamp"] >= lo) & (m["timestamp"] <= hi)][["timestamp", "latency_ms", "error_rate"]],
         "service/metrics_around_release-064.csv")
    save(pd.DataFrame([{"timestamp": c.timestamp, "kind": c.kind, "id": c.id} for c in ep.context.changes]),
         "service/releases_and_config_pushes.csv")


def injected() -> None:
    rows = []
    for ep in episodes(EvalConfig()):
        for i in ep.injections:
            rows.append({"id": i.id, "scenario": SCENARIO[i.family], "cause": i.truth.value, "detail": i.truth_sub,
                         "release_or_config": i.truth_suspect, "injected_at": i.fault_start,
                         "alerts_raised": len(i.alert_ids), "real_fault": i.is_fault, "graded": i.scored})
    save(pd.DataFrame(rows), "injected/faults.csv")


if __name__ == "__main__":
    nyc_taxi()
    ci_and_logs()
    service()
    injected()
