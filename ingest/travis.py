"""TravisTorrent -> one row per BUILD, with a flaky label and causal features.

TravisTorrent has one row per JOB. We first remove jobs listed twice (the 2016 and 2017 files
overlap), then collapse jobs into builds.

FLAKY LABEL (the same idea as "same code, different result"):
  - builds of the SAME project, the SAME trigger commit (and the same merge target for pull
    requests) are re-runs of identical code
  - a FAILED build is "flaky_fail" if a build of that same code PASSED, otherwise "real_fail"
  - "errored" builds (setup/infra problems) are kept apart, "pass" is a passing build
Limit: a flaky failure that nobody re-ran looks like "real_fail" (label noise), so measured
recall on flaky failures will be pessimistic.

Features are CAUSAL: they only use what was known BEFORE the build (previous builds of the same
project). Nothing that depends on later re-runs is used as a feature.
"""
import numpy as np
import pandas as pd
from config import settings

CANDIDATES = {
    "build_id":     ["tr_build_id"],
    "job_id":       ["tr_job_id"],
    "project":      ["gh_project_name"],
    "commit":       ["git_trigger_commit"],
    "merged":       ["tr_virtual_merged_into"],
    "status":       ["tr_status", "build_status"],
    "started":      ["gh_build_started_at", "tr_started_at", "build_started_at"],
    "duration":     ["tr_duration", "build_duration"],
    "is_pr":        ["gh_is_pr"],
    "tests_run":    ["tr_log_num_tests_run"],
    "tests_failed": ["tr_log_num_tests_failed"],
}
REQUIRED = ["build_id", "project", "commit", "status", "started"]


def _pick(header, names):
    return next((n for n in names if n in header), None)


def read_jobs() -> pd.DataFrame:
    files = sorted(settings.TRAVIS_RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV in {settings.TRAVIS_RAW_DIR}")
    frames = []
    for f in files:
        header = pd.read_csv(f, nrows=0).columns.tolist()
        cols = {k: _pick(header, v) for k, v in CANDIDATES.items()}
        missing = [k for k in REQUIRED if cols[k] is None]
        if missing:
            print(f"{f.name}: SKIPPED, missing columns for {missing}")
            continue
        use = {k: c for k, c in cols.items() if c}
        df = pd.read_csv(f, usecols=list(use.values()), low_memory=False)
        frames.append(df.rename(columns={c: k for k, c in use.items()}))
        print(f"{f.name}: {len(df):,} job rows  (columns found: {sorted(use)})")
    jobs = pd.concat(frames, ignore_index=True)
    n = len(jobs)
    jobs = jobs.drop_duplicates(subset="job_id") if "job_id" in jobs else jobs.drop_duplicates()
    print(f"removed {n - len(jobs):,} jobs that appear in several files -> {len(jobs):,} jobs")
    return jobs


def jobs_to_builds(jobs: pd.DataFrame) -> pd.DataFrame:
    jobs = jobs.copy()
    jobs["status"] = jobs["status"].astype(str).str.strip().str.lower()
    jobs = jobs[jobs["status"].isin({"passed", "failed", "errored"})]
    jobs["started"] = pd.to_datetime(jobs["started"], errors="coerce", utc=True)
    jobs = jobs.dropna(subset=["started", "build_id", "project", "commit"])
    for c in ("duration", "tests_run", "tests_failed"):
        if c in jobs:
            jobs[c] = pd.to_numeric(jobs[c], errors="coerce")
    if "is_pr" in jobs:
        jobs["is_pr"] = jobs["is_pr"].astype(str).str.strip().str.lower().isin(["true", "1"])

    g = jobs.groupby("build_id", sort=False)
    b = g.agg(project=("project", "first"), commit=("commit", "first"),
              status=("status", "first"), started=("started", "min"),
              n_jobs=("status", "size"))
    for c, how in (("merged", "first"), ("is_pr", "first"), ("duration", "max")):
        if c in jobs:
            b[c] = g[c].agg(how)
    for c in ("tests_run", "tests_failed"):
        if c in jobs:
            b[c] = g[c].sum(min_count=1)          # stays empty when no job has log info
    b = b.reset_index()

    # ---- label: builds of the same code (project + commit + merge target) ----
    if "merged" in b:
        same_code = b["merged"].fillna("").astype(str)
    else:
        same_code = b["is_pr"].astype(int) if "is_pr" in b else 0
    b["code_id"] = b.assign(_m=same_code).groupby(["project", "commit", "_m"], sort=False).ngroup()
    n_pass_same_code = (b["status"] == "passed").groupby(b["code_id"]).transform("sum")
    b["label"] = np.select(
        [b["status"] == "passed", b["status"] == "errored", n_pass_same_code > 0],
        ["pass", "errored", "flaky_fail"], default="real_fail")

    # ---- causal features: only previous builds of the same project ----
    b = b.sort_values(["project", "started"]).reset_index(drop=True)
    b["is_fail"] = (b["status"] != "passed").astype(int)
    gp = b.groupby("project", sort=False)
    b["prev_status"] = gp["status"].shift(1)
    b["fail_rate_20"] = gp["is_fail"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    b["n_prev_builds"] = gp.cumcount()
    b["mins_since_prev"] = gp["started"].diff().dt.total_seconds() / 60
    return b.drop(columns=["is_fail", "code_id"])


def build_builds() -> pd.DataFrame:
    return jobs_to_builds(read_jobs())
