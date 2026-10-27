"""Labelled incidents to train and measure Diagnosis: the "injected failures with true category" of the plan.

Three sources, from most to least real (every Case says where it comes from):

  1. deploy  P1's deploy simulator (ingest/deploy.py) -> P1's rolling detector -> P2's Triage rules.
             The SAME chain as production, with the truth known: slow / errors / leak / harmless deploys.
             We add what P1's simulator does not have, so Diagnosis must tell causes APART:
               config pushes that break the error rate            -> config_change
               saturation with no change at all (latency ramps)   -> infra
             Truth = the fault whose time window contains the incident start; no fault = unknown.
  2. real    P1's labelled exports, when present (data/triage_eval or logs/triage_eval):
               ci_labels.csv    TravisTorrent builds: flaky_fail -> flaky_test, real_fail -> code_regression
               hdfs_labels.csv  Loghub HDFS blocks: failing -> infra, normal -> unknown (detector false alarm)
  3. synth   CI, log and model-input incidents in P1's exact Alert format, for the causes no public
             dataset labels (CI infrastructure outage, replication failure, data outage). Distributions
             overlap on purpose and include hard cases, so the score is not flattering.
             They will be replaced by P5's failure injections (data/injections/) when those exist.

Split: deploy runs by seed (seeds 0-3 train, 4-5 test), the rest 60/40 at random, fixed seed.
"""
from __future__ import annotations

import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from common.schemas import Alert, Category as C, Incident

from . import paths
from .context import ContextStore

DEPLOY_SUB = {"slow": "latency_regression", "errors": "error_regression", "leak": "resource_leak"}
TRAIN_SEEDS, TEST_SEEDS = (0, 1, 2, 3), (4, 5)


@dataclass
class Case:
    incident: Incident
    ctx: ContextStore
    truth: C
    truth_sub: str = ""
    truth_suspect: str = ""      # the change that must be undone ("" if none)
    family: str = ""             # deploy / ci / logs / model
    origin: str = ""             # "simulation" | "real" | "synthetic"
    split: str = "train"


# ====================================================================== 1. deploy (real chain)
def _inject(metrics: pd.DataFrame, deploys: pd.DataFrame, rng, n_config=3, n_infra=3):
    """Add config pushes and infra saturation to P1's simulated metrics, away from every deploy."""
    m = metrics.copy()
    t = m["timestamp"]
    start_test = 14 * 1440 + 120
    busy = [int(((d - t.iloc[0]).total_seconds()) // 60) for d in deploys["timestamp"]]
    faults, pushes = [], []
    for kind in ["config"] * n_config + ["infra"] * n_infra:
        for _ in range(200):
            pos = int(rng.integers(start_test, len(m) - 300))
            if all(abs(pos - b) > 200 for b in busy):
                break
        busy.append(pos)
        dur = int(rng.integers(60, 120))
        w = slice(pos, pos + dur)
        if kind == "config":            # wrong endpoint / secret: errors jump at once, latency does not move
            m.loc[m.index[w], "error_rate"] *= 1 + rng.uniform(1.5, 4.0)
            pushes.append({"timestamp": t.iloc[pos]})
            faults.append((t.iloc[pos], dur, C.CONFIG_CHANGE, "bad_config", f"config-{len(pushes):03d}"))
        else:                           # saturation: latency climbs, errors follow, nothing was changed
            ramp = np.minimum(np.arange(dur) / 40, 1.0)
            m.loc[m.index[w], "latency_ms"] *= 1 + rng.uniform(0.25, 0.6) * ramp
            m.loc[m.index[w], "error_rate"] *= 1 + rng.uniform(0.5, 1.5) * ramp
            faults.append((t.iloc[pos], dur, C.INFRA, "capacity_saturation", ""))
    return m, pd.DataFrame(pushes, columns=["timestamp"]), faults


def simulate_deploy(seed: int):
    """P1's deploy simulator + our config / infra faults -> P1's rolling detector.
    Returns (P1's Alert objects, ContextStore with the change log and the metrics, faults with their truth).
    Also used by pipeline.py to run the whole graph without downloading anything."""
    from agents.monitor.factory import fit_detector, make_detector, slice_period
    from agents.monitor.series_monitor import SeriesMonitor
    from config import settings
    from ingest.deploy import simulate

    rng = np.random.default_rng(1000 + seed)
    metrics, deploys = simulate(seed=seed)
    metrics, pushes, faults = _inject(metrics, deploys, rng)
    for k, d in enumerate(deploys.itertuples(), 1):
        if d.bad:   # a person rolls a bad deploy back after 2 hours (ingest/deploy.py)
            faults.append((d.timestamp, 120, C.BAD_DEPLOY, DEPLOY_SUB[d.kind], f"release-{k:03d}"))

    sp = settings.SPLITS["deploy"]
    train, test = slice_period(metrics, *sp["train"]), slice_period(metrics, *sp["test"])
    bundle = {f: {"rolling": fit_detector(make_detector("rolling"), train[f].values, train["timestamp"])}
              for f in settings.FEATURES["deploy"]}
    monitor = SeriesMonitor(bundle, "deploy", min_votes=settings.MIN_VOTES,
                            max_gap_hours=settings.MERGE_GAP_HOURS["deploy"])
    ctx = ContextStore.from_deploy_table(metrics, deploys, config_pushes=pushes)
    return list(monitor.run(test)), ctx, faults


def deploy_cases(seed: int, split: str) -> list[Case]:
    from agents.triage.agent import RuleTriage

    p1_alerts, ctx, faults = simulate_deploy(seed)
    alerts = [Alert.from_dict({**a.to_row(), "details": a.details}) for a in p1_alerts]
    incidents = RuleTriage(id_prefix=f"DEP{seed}").process(alerts)
    ctx = ctx.with_peers(incidents)

    cases = []
    for inc in incidents:
        hit = [f for f in faults if f[0] - timedelta(minutes=5) <= inc.start <= f[0] + timedelta(minutes=f[1] + 20)]
        if hit:
            t, dur, cat, sub, suspect = min(hit, key=lambda f: inc.start - f[0])
            if inc.start >= t + timedelta(minutes=dur - 5):     # alert on the way BACK (rollback, fault over)
                cat, sub, suspect = C.UNKNOWN, "recovery", ""
        else:
            cat, sub, suspect = C.UNKNOWN, "no_fault", ""
        cases.append(Case(inc, ctx, cat, sub, suspect, "deploy", "simulation", split))
    return cases


# ====================================================================== 2. real labelled exports
def _split(key: str) -> str:
    """Same answer on every run (Python's hash() of a string is not)."""
    return "test" if zlib.crc32(key.encode()) % 5 < 2 else "train"


def real_cases() -> list[Case]:
    """P1's alerts + labels, when the files exist. Returns [] otherwise."""
    from agents.triage import paths as triage_paths
    from agents.triage.agent import RuleTriage
    from agents.triage.io import load_alerts

    ci_f, hdfs_f = paths.TRIAGE_EVAL / "ci_labels.csv", paths.TRIAGE_EVAL / "hdfs_labels.csv"
    if not triage_paths.ALERTS.exists() or not (ci_f.exists() or hdfs_f.exists()):
        return []
    alerts = load_alerts(triage_paths.ALERTS)
    cases = []
    if ci_f.exists():
        lab = pd.read_csv(ci_f).set_index("build_id")["label"].to_dict()
        incidents = RuleTriage(id_prefix="RCI").process([a for a in alerts if a.family == "ci"])
        ctx = ContextStore().with_peers(incidents)
        for inc in incidents:
            ys = [lab.get(a.details.get("build_id")) for a in inc.alerts]
            flaky = sum(y == "flaky_fail" for y in ys)
            real = sum(y == "real_fail" for y in ys)
            if flaky + real:
                cases.append(Case(inc, ctx, C.FLAKY_TEST if flaky > real else C.CODE_REGRESSION, "", "", "ci",
                                  "real", _split(inc.id)))
    if hdfs_f.exists():
        lab = pd.read_csv(hdfs_f).set_index("block_id")["failing"].to_dict()
        incidents = RuleTriage(id_prefix="RLOG").process([a for a in alerts if a.family == "logs"])
        ctx = ContextStore().with_peers(incidents)
        for inc in incidents:
            failing = np.mean([lab.get(a.details["block_id"], 0) for a in inc.alerts])
            cases.append(Case(inc, ctx, C.INFRA if failing >= 0.5 else C.UNKNOWN, "", "", "logs", "real",
                              _split(inc.id)))
    return cases


# ====================================================================== 3. synthetic families
def _alert(**kw) -> Alert:
    base = dict(threshold=1.0, kind="anomaly", details={})
    return Alert.from_dict({**base, **kw})


def ci_cases(rng, n_flaky=70, n_regression=70, n_outages=14) -> list[Case]:
    from agents.triage.agent import RuleTriage

    projects = [f"org{k}/repo{k}" for k in range(60)]
    day0 = datetime(2016, 1, 4, tzinfo=timezone.utc)
    alerts, truth, day = [], {}, 0

    def builds(project, n, p_lo, p_hi, flaky_rate, recent, same_commit, when, prev_failed=0.2):
        commit = f"{rng.integers(16**8):08x}"
        out = []
        for b in range(n):
            if not same_commit:
                commit = f"{rng.integers(16**8):08x}"
            p = float(rng.uniform(p_lo, p_hi))
            out.append(_alert(detector="flaky_model", source="ci", feature="build_failed", dataset="travis",
                              score=p / 0.15, value=p, timestamp=(when + timedelta(minutes=20 * b)).isoformat(),
                              details={"flaky_probability": round(p, 4), "project": project,
                                       "build_id": int(rng.integers(10**8)), "commit": commit,
                                       "evidence": {"project_flaky_rate": round(float(rng.uniform(*flaky_rate)), 3),
                                                    "recent_fail_rate": round(float(rng.uniform(*recent)), 3),
                                                    "previous_build_failed": float(b > 0 or rng.random() < prev_failed)}}))
        return out

    for _ in range(n_flaky):
        day += 1
        proj, when = projects[rng.integers(len(projects))], day0 + timedelta(days=day, hours=int(rng.integers(8, 20)))
        alerts += builds(proj, int(rng.integers(1, 3)), 0.25, 0.95, (0.2, 0.7), (0.05, 0.45), False, when)
        truth[(proj, when.date())] = (C.FLAKY_TEST, "")
    for _ in range(n_regression):
        day += 1
        proj, when = projects[rng.integers(len(projects))], day0 + timedelta(days=day, hours=int(rng.integers(8, 20)))
        alerts += builds(proj, int(rng.integers(1, 5)), 0.02, 0.5, (0.05, 0.4), (0.0, 0.35), rng.random() < 0.6, when)
        truth[(proj, when.date())] = (C.CODE_REGRESSION, "")
    for _ in range(n_outages):          # shared runners down: many projects red within the same hour
        day += 1
        when = day0 + timedelta(days=day, hours=int(rng.integers(8, 20)))
        for proj in rng.choice(projects, int(rng.integers(3, 8)), replace=False):
            t = when + timedelta(minutes=int(rng.integers(0, 40)))
            alerts += builds(str(proj), int(rng.integers(1, 3)), 0.1, 0.7, (0.1, 0.5), (0.0, 0.3), False, t)
            truth[(str(proj), t.date())] = (C.INFRA, "ci_runners")

    incidents = RuleTriage(id_prefix="SCI").process(alerts)
    ctx = ContextStore().with_peers(incidents)
    return [Case(i, ctx, *truth[(i.alerts[0].details["project"], i.start.date())], "", "ci", "synthetic")
            for i in incidents]


LOG_POOLS = {   # event pools per truth; disjoint so Triage's grouping by pattern cannot mix two truths
    (C.INFRA, "datanode_network"): ["E7", "E10", "E12", "E14", "E17", "E8"],
    (C.INFRA, "replication"): ["E29", "E25", "E24", "E18"],
    (C.UNKNOWN, "known_false_alarm"): ["E2", "E3", "E4"],
    (C.UNKNOWN, "no_fault"): ["E1", "E13", "E15", "E19", "E22", "E23", "E26", "E27", "E28"],
}


def log_cases(rng, n=(40, 20, 6, 30)) -> list[Case]:
    from agents.triage.agent import RuleTriage

    alerts, truth = [], {}
    for (key, pool), count in zip(LOG_POOLS.items(), n):
        for _ in range(count):
            events = sorted(rng.choice(pool, 3, replace=False)) if len(pool) > 3 else sorted(pool)
            pattern = "+".join(sorted(events))
            if pattern in truth:
                continue
            truth[pattern] = key
            for b in range(int(rng.integers(1, 25))):
                share = rng.dirichlet(np.ones(3))
                score = float(rng.uniform(1.2, 9.0))
                alerts.append(_alert(detector="block_pca", source="logs", feature="block_events", dataset="hdfs",
                                     score=score, value=score, timestamp="2026-10-20T11:00:00+00:00",
                                     details={"block_id": f"blk_{rng.integers(10**12)}",
                                              "top_events": [{"event": e, "share": round(float(s), 3)}
                                                             for e, s in zip(events, share)]}))
    incidents = RuleTriage(id_prefix="SLOG").process(alerts)
    ctx = ContextStore().with_peers(incidents)
    pat = lambda i: "+".join(sorted(e["event"] for e in i.alerts[0].details["top_events"]))
    return [Case(i, ctx, *truth[pat(i)], "", "logs", "synthetic") for i in incidents]


def model_cases(rng, n_drift=30, n_quality=25, n_outage=10, n_single=25) -> list[Case]:
    from agents.triage.agent import RuleTriage

    feats = ["trip_count", "avg_fare", "avg_distance"]
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    alerts, truth, day = [], {}, 0

    def series(feature, start, hours, ratio, kind="anomaly", psi=None):
        end = start + timedelta(hours=hours - 1)
        details = {"voters": ["psi" if kind == "drift" else "seasonal"], "start": str(start), "end": str(end),
                   "n_points": hours, "peak_raw_scores": {"psi": psi} if psi else {"seasonal": ratio * 3}}
        return _alert(detector=f"vote({details['voters'][0]})", source="model", feature=feature, dataset="nyc",
                      kind=kind, score=ratio, value=psi or ratio, timestamp=start.isoformat(), details=details)

    for _ in range(n_drift):
        day += 30
        f, psi = str(rng.choice(["avg_fare", "avg_distance", "avg_passenger"])), float(rng.uniform(0.26, 0.9))
        alerts.append(series(f, t0 + timedelta(days=day), int(rng.integers(72, 400)), psi / 0.25, "drift", psi))
        truth[(day, "drift")] = (C.DATA_DRIFT, f)
    for _ in range(n_quality):          # several inputs jump in the same hours
        day += 30
        start = t0 + timedelta(days=day, hours=int(rng.integers(24)))
        for f in rng.choice(feats, int(rng.integers(2, 4)), replace=False):
            alerts.append(series(str(f), start, int(rng.integers(1, 5)), float(rng.uniform(1.3, 6))))
        truth[(day, "anomaly")] = (C.DATA_QUALITY, "")
    for _ in range(n_outage):           # HARD: ingestion lost hours, only trip_count collapses (one feature)
        day += 30
        alerts.append(series("trip_count", t0 + timedelta(days=day), int(rng.integers(2, 6)), float(rng.uniform(5, 12))))
        truth[(day, "anomaly")] = (C.DATA_QUALITY, "missing_data")
    for _ in range(n_single):           # one input, short, weak: a real-world oddity, no fault
        day += 30
        alerts.append(series(str(rng.choice(feats)), t0 + timedelta(days=day), int(rng.integers(1, 3)),
                             float(rng.uniform(1.0, 2.5))))
        truth[(day, "anomaly")] = (C.UNKNOWN, "no_fault")

    incidents = RuleTriage(id_prefix="SMOD").process(alerts)
    ctx = ContextStore().with_peers(incidents)
    key = lambda i: ((i.start - t0).days // 30 * 30, i.kind)
    return [Case(i, ctx, *truth[key(i)], "", "model", "synthetic") for i in incidents]


# ====================================================================== the benchmark
def build(verbose: bool = True) -> list[Case]:
    cases = []
    for seeds, split in ((TRAIN_SEEDS, "train"), (TEST_SEEDS, "test")):
        for s in seeds:
            got = deploy_cases(s, split)
            cases += got
            if verbose:
                print(f"  deploy seed {s} ({split}): {len(got)} incidents")
    rng = np.random.default_rng(7)
    synth = ci_cases(rng) + log_cases(rng) + model_cases(rng)
    for c in synth:
        c.split = "test" if rng.random() < 0.4 else "train"
    real = real_cases()
    if verbose:
        print(f"  synthetic: {len(synth)} incidents | real exports: {len(real)} incidents")
    return cases + synth + real


def load_or_build(rebuild: bool = False) -> list[Case]:
    if paths.BENCHMARK.exists() and not rebuild:
        with open(paths.BENCHMARK, "rb") as f:
            return pickle.load(f)
    cases = build()
    paths.BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.BENCHMARK, "wb") as f:
        pickle.dump(cases, f)
    return cases
