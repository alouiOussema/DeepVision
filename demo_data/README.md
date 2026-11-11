# Demo data

DeepVision is the operations layer: it watches an MLOps pipeline and fixes it. This folder shows **what that pipeline
is made of**, and keeps a small sample of every source so the demo runs on any laptop, offline.

There are three layers, and they are not the same thing:

| Layer | What it is | Where it comes from |
|---|---|---|
| **1. Public data we brought in** | NYC taxi trips, CI build history, storage logs | Downloaded from public datasets (links below). Not created by us |
| **2. The pipeline it stands for** | A taxi-fare model with its CI, its web service and its storage | Built by the team on top of layer 1, plus a simulated web service |
| **3. Failures we injected** | Bad releases, flaky tests, drift, storage failures, each with its right answer | Added by us to test the agents. **Not in the original data** |

The full datasets are several gigabytes, so they are not in git. The files here are samples and summaries.
Rebuild them with `python -m scripts.make_demo_data`.

## The project we operate: a taxi-fare model

The ML model predicts the **average taxi fare of the coming hour** in New York, from the hour, the weekday, the
average trip distance and the number of trips (`agents/action/retrain_sklearn.py`). Around it: a CI that builds and
tests the code, a web service that serves it, and storage that holds the data.

## Sources

### NYC Taxi: real data
- **Source:** NYC Taxi & Limousine Commission, Trip Record Data (yellow taxis), public:
  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Files used:** `yellow_tripdata_2025-01`, `2025-02`, `2025-11`, `2025-12`, `2026-01`, `2026-02` (parquet, about
  60 MB each), downloaded from `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_<YYYY-MM>.parquet`
  into `data/raw/nyc/`.
- **What we did:** 22,663,974 raw trips → invalid trips and trips outside their month removed → 21,184,556 trips
  → **one row per hour** (trip count, average fare, average distance, average passengers), with the team's loader
  `ingest/nyc.py`. That hourly table is what the Monitor watches.
- **A real drift, not injected:** the average fare per trip went from **$17.95 (Jan 2025) to $21.02 (Jan 2026),
  +17 %** (February: +21 %).

| File | Content |
|---|---|
| `nyc_taxi/raw_trips_sample.csv` | 15 raw trips, as published by the TLC |
| `nyc_taxi/hourly.csv` | the hourly table built from them (4,296 hours) |
| `nyc_taxi/monthly_summary.csv` | raw vs kept trips, average fare and distance per month |

### CI builds: TravisTorrent format
- **Source:** TravisTorrent, build history of open-source projects on Travis CI: https://travistorrent.testroots.org
- **What the team did:** jobs merged into builds; a failed build is *flaky* when the same code also passed
  (`ingest/travis.py`). The full file is several GB and stays on the Monitor author's machine.
- **Here:** `ci_builds/builds_example.csv` holds **generated examples in the same format**, used by the evaluation,
  with the true cause of each failure.

### Storage logs: Loghub HDFS format
- **Source:** Loghub, HDFS logs of a Hadoop cluster, labelled per block: https://github.com/logpai/loghub
- **What the team did:** each block becomes a count of 29 log event types (`ingest/hdfs.py`).
- **Here:** `hdfs_logs/blocks_example.csv` holds **generated examples in the same format**, with the true cause.

### Web service metrics: simulated
- **Source:** none: no real deployment data exists, so the Monitor's author wrote a simulator (`ingest/deploy.py`):
  a service with daily traffic, latency, error rate, and releases, some of them bad.
- **Here:** `service/metrics_around_release-064.csv` (one bad release, minute by minute) and
  `service/releases_and_config_pushes.csv`.

## What we injected

`injected/faults.csv` lists every failure the evaluation injects: 392 entries, of which 334 are real faults and 58
are harmless controls. 154 real faults are graded; the others were used to train the Diagnosis classifier, so they
run but are not scored.

| Scenario | Injected causes |
|---|---|
| 1 · Flaky test | flaky tests, real code regressions, CI machines down |
| 2 · Bad deploy | bad releases (slower, errors, memory leak), bad config pushes, server overload |
| 3 · Data drift | slow drift of a model input, broken data (missing hours, bursts) |
| 4 · Unknown failure | storage failures (datanode network, replication) |
