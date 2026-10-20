"""Step 3: prove the detectors work. Each test measures one detector against known truth.

    python -m scripts.03_test nyc      # planted anomalies: global z-score vs seasonal (recall, false alerts)
    python -m scripts.03_test psi      # drift: PSI window sizes against injected drift
    python -m scripts.03_test hdfs     # log detectors on real block labels, and the limit trade-off
    python -m scripts.03_test flaky    # flaky model vs simple baselines   (optional: a split date)
    python -m scripts.03_test deploy   # bad deploys: simulation, plus RCAEval cases if downloaded
    python -m scripts.03_test          # all five
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from config import settings
from ingest import deploy as deploy_data
from ingest.hdfs import load_event_matrix, split_blocks
from agents.monitor import ci_monitor
from agents.monitor.detectors.block_pca import BlockPCA
from agents.monitor.testing import evaluate, plant
from agents.monitor.factory import make_detector, fit_detector, score_series, slice_period

pd.set_option("display.width", 160)


def nyc_split():
    df = pd.read_parquet(settings.NYC_HOURLY_FILE)
    sp = settings.SPLITS["nyc"]
    return slice_period(df, *sp["train"]), slice_period(df, *sp["test"])


# ------------------------------------------------------------------ nyc: planted anomalies
def test_nyc():
    train, test = nyc_split()
    tz = settings.TIMEZONES["nyc"]
    for feature in settings.FEATURES["nyc"]:
        tr = train[["timestamp", feature]].dropna()
        te = test[["timestamp", feature]].dropna()
        series, labels = plant(te, feature, tr[feature].values, tr["timestamp"])
        tables, summaries = {}, {}
        for name in settings.COMPARE_DETECTORS:
            try:
                det = fit_detector(make_detector(name, tz), tr[feature].values, tr["timestamp"])
            except ImportError as e:
                print(f"  ({name} skipped: {e})")
                continue
            scores = score_series(det, series[feature].values, series["timestamp"])
            tables[name], summaries[name] = evaluate(series["timestamp"], series["planted"],
                                                     scores > det.threshold, labels)
        view = next(iter(tables.values()))[["kind", "size", "events"]].copy()
        for name, t in tables.items():
            view[name] = t["recall"].values
        print(f"\n=== nyc / {feature}  ({len(labels)} planted events) ===")
        print("share of planted events caught (recall):")
        print(view.round(2).to_string(index=False))
        board = []
        for name, sm in summaries.items():
            t = tables[name]
            board.append({"detector": name, "false_alerts_per_week": round(sm["false_alerts_per_week"], 1),
                          "alert_rate_clean": f"{sm['alert_rate_on_clean_hours']:.1%}",
                          "recall_overall": f"{sm['recall_overall']:.0%}",
                          "recall_spikes_drops": f"{t[t.kind.isin(['spike', 'drop'])]['recall'].mean():.0%}"})
        print("\nscoreboard (spikes and drops last 1 hour, so luck plays no role there):")
        print(pd.DataFrame(board).to_string(index=False))


# ------------------------------------------------------------------ psi: injected drift
def test_psi():
    WINDOWS_DAYS = (7, 14, 28)
    CASES = [("step", 1.05), ("step", 1.10), ("step", 1.25), ("ramp", 1.10), ("ramp", 1.25)]
    STARTS, RAMP_HOURS = (0.20, 0.45, 0.70), 24 * 30

    def drift(v, start, factor, kind):
        v = v.copy()
        if kind == "step":
            v[start:] *= factor
        else:
            v *= 1 + (factor - 1) * np.clip((np.arange(len(v)) - start) / RAMP_HOURS, 0, 1)
        return v

    def episodes(flag):
        return int(np.sum(np.diff(np.r_[0, flag.astype(int)]) == 1))

    train, test = nyc_split()
    for feature in settings.DRIFT["nyc"]:
        tr = train[["timestamp", feature]].dropna()
        te = test[["timestamp", feature]].dropna().reset_index(drop=True)
        rows = []
        for days in WINDOWS_DAYS:
            det = make_detector("psi")
            det.window = days * 24
            fit_detector(det, tr[feature].values, tr["timestamp"])
            quiet = det.score(te[feature].values) > det.threshold
            row = {"window": f"{days} d", "limit": round(det.threshold, 2), "quiet_%": f"{quiet.mean():.1%}",
                   "false_alarms": f"{episodes(quiet)} in {len(te) // 24} d"}
            for kind, factor in CASES:
                delays = []
                for frac in STARTS:
                    start = int(frac * len(te))
                    new = (det.score(drift(te[feature].values, start, factor, kind)) > det.threshold) & ~quiet
                    hit = np.flatnonzero(new[start:])
                    if len(hit):
                        delays.append(hit[0] / 24)
                row[f"{kind} +{factor - 1:.0%}"] = (f"{len(delays)}/{len(STARTS)} {np.median(delays):.0f}d"
                                                    if delays else f"0/{len(STARTS)}")
            rows.append(row)
        print(f"\n=== nyc / {feature} ===   (cell: caught x of 3, median delay in days)")
        print(pd.DataFrame(rows).to_string(index=False))


# ------------------------------------------------------------------ hdfs: block detectors
def test_hdfs():
    def at_limit(s_tr, s_te, yte, rate):
        pred = (s_te > np.quantile(s_tr, 1 - rate)).astype(int)
        p, r, f, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        return {"false_alarm_rate": f"{rate:.1%}", "precision": round(p, 3), "recall": round(r, 3),
                "F1": round(f, 3), "false_alarms": int(((pred == 1) & (yte == 0)).sum())}

    def total_events(Xtr, Xte):                       # ONE number per block
        a, b = Xtr.sum(1), Xte.sum(1)
        med, mad = np.median(a), np.median(np.abs(a - np.median(a))) * 1.4826 + 1e-9
        return np.abs(a - med) / mad, np.abs(b - med) / mad

    def iforest_29(Xtr, Xte):                         # IsolationForest on all 29 counts
        A, B = np.log1p(Xtr), np.log1p(Xte)
        m = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(A)
        return -m.decision_function(A), -m.decision_function(B)

    def pca_29(Xtr, Xte):                             # PCA on all 29 counts (used by the Monitor)
        d = BlockPCA(false_alarm_rate=settings.HDFS_FALSE_ALARM_RATE).fit(Xtr)
        return d.score(Xtr), d.score(Xte)

    ids, X, y, _ = load_event_matrix()
    is_train = split_blocks(len(y))
    Xtr, Xte, yte = X[is_train & (y == 0)], X[~is_train], y[~is_train]
    print(f"blocks: {len(y):,}  failing: {int(y.sum()):,} ({y.mean():.1%});  trained on {len(Xtr):,} normal blocks, "
          f"tested on {len(yte):,} unseen ({int(yte.sum()):,} failing)\n")
    rows, scores = [], {}
    for name, fn in {"total_events": total_events, "iforest_29": iforest_29, "pca_29": pca_29}.items():
        scores[name] = fn(Xtr, Xte)
        rows.append({"method": name, "AUC": round(roc_auc_score(yte, scores[name][1]), 3),
                     **at_limit(*scores[name], yte, 0.01)})
    print("all methods, limit = 1% false alarms on normal blocks:")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\npca_29: the trade-off when the limit moves")
    print(pd.DataFrame([at_limit(*scores["pca_29"], yte, r) for r in (0.005, 0.01, 0.02, 0.05, 0.10)]).to_string(index=False))


# ------------------------------------------------------------------ flaky: model vs baselines
def test_flaky(split_date=None):
    b, X, y, train_mask, test_mask, prior = ci_monitor.prepare(pd.read_parquet(settings.BUILDS_FILE),
                                                          split_date or settings.FLAKY_SPLIT_DATE)
    Xtr, ytr, Xte, yte = X[train_mask], y[train_mask], X[test_mask], y[test_mask]
    base = float(yte.mean())
    print(f"failed builds  train: {len(ytr):,} ({int(ytr.sum()):,} flaky = {ytr.mean():.1%})   "
          f"test: {len(yte):,} ({int(yte.sum()):,} flaky = {base:.1%})")
    print(f"random guessing is right {base:.1%} of the time (AUC 0.5)\n")
    model = ci_monitor.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]

    def metrics(score):
        return {"AUC": round(roc_auc_score(yte, score), 3), "avg_precision": round(average_precision_score(yte, score), 3)}

    calm = 1 - Xte["fail_rate_20"].fillna(Xte["fail_rate_20"].median())
    print(pd.DataFrame([{"scorer": "model (all features)", **metrics(p)},
                        {"scorer": "project history only", **metrics(Xte["proj_flaky_rate"].fillna(prior))},
                        {"scorer": "calm project (1 - recent fail rate)", **metrics(calm)}]).to_string(index=False))
    order, n, top = np.argsort(-p), len(p), []
    for share in (0.01, 0.02, 0.05, 0.10, 0.20):
        k = max(int(n * share), 1)
        hit = int(yte.to_numpy()[order[:k]].sum())
        top.append({"flag_top": f"{share:.0%} of failures", "builds_flagged": k, "precision": round(hit / k, 3),
                    "lift_vs_random": round(hit / k / base, 1), "flaky_found": f"{hit / max(int(yte.sum()), 1):.0%}"})
    print("\nif we call the most suspicious failures 'flaky':")
    print(pd.DataFrame(top).to_string(index=False))
    sample = np.random.default_rng(0).choice(n, size=min(n, 20000), replace=False)
    imp = permutation_importance(model, Xte.iloc[sample], yte.iloc[sample], scoring="average_precision",
                                 n_repeats=3, random_state=0)
    print("\nwhat the model relies on (drop in average precision when shuffled):")
    print(pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False).head(8).round(4).to_string())


# ------------------------------------------------------------------ deploy: bad deploys vs harmless deploys
def test_deploy():
    df = pd.read_parquet(settings.DEPLOY_METRICS_FILE)
    ev = pd.read_parquet(settings.DEPLOY_EVENTS_FILE)
    sp = settings.SPLITS["deploy"]
    train, test = slice_period(df, *sp["train"]), slice_period(df, *sp["test"]).reset_index(drop=True)
    ev = ev[(ev.timestamp >= pd.Timestamp(sp["test"][0], tz="UTC")) & (ev.timestamp < pd.Timestamp(sp["test"][1], tz="UTC"))]
    t = pd.DatetimeIndex(test["timestamp"])
    check = pd.Timedelta(minutes=settings.DEPLOY_CHECK_MIN)
    in_window = np.zeros(len(t), bool)                       # minutes just after a deploy
    for e in ev.itertuples():
        in_window |= np.asarray((t >= e.timestamp) & (t < e.timestamp + check))
    print(f"test period: {len(ev)} deploys, {int(ev.bad.sum())} bad, {int((~ev.bad).sum())} harmless")
    print(f"an alert within {settings.DEPLOY_CHECK_MIN} min after a deploy counts for that deploy\n")

    rows, per_event = [], []
    for name in ("zscore", "rolling"):
        flagged = {}
        for feature in settings.FEATURES["deploy"]:
            det = fit_detector(make_detector(name), train[feature].values, train["timestamp"])
            flagged[feature] = score_series(det, test[feature].values, test["timestamp"]) > det.threshold
        any_flag = np.any(list(flagged.values()), axis=0)
        caught, delays, false_alarm = [], [], []
        for e in ev.itertuples():
            m = np.asarray((t >= e.timestamp) & (t < e.timestamp + check))
            hit = any_flag[m]
            per_event.append({"detector": name, "kind": e.kind, "size_%": e.size, "caught": bool(hit.any())})
            if e.bad:
                caught.append(bool(hit.any()))
                if hit.any():
                    delays.append(int(hit.argmax()))
            else:
                false_alarm.append(bool(hit.any()))
        outside = any_flag[~in_window]
        rows.append({"detector": name, "bad_deploys_caught": f"{np.mean(caught):.0%} ({sum(caught)}/{len(caught)})",
                     "median_delay_min": float(np.median(delays)) if delays else None,
                     "harmless_flagged": f"{np.mean(false_alarm):.0%} ({sum(false_alarm)}/{len(false_alarm)})",
                     "flagged_minutes_elsewhere": f"{outside.mean():.2%}"})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n('elsewhere' = minutes far from any deploy: alerts there are false alarms)")
    pe = pd.DataFrame(per_event)
    print("\nshare caught, by kind of deploy (leak = slow ramp over 2 h, the hardest):")
    print(pe.pivot_table(index=["kind", "size_%"], columns="detector", values="caught", aggfunc="mean").round(2).to_string())

    cases = sorted(p for p in settings.RCAEVAL_DIR.glob("*") if p.is_dir() and not p.name.startswith(".")) if settings.RCAEVAL_DIR.exists() else []
    if not cases:
        print(f"\nRCAEval: no case folders in {settings.RCAEVAL_DIR} (optional, see ingest/deploy.py)")
        return
    # RCAEval cases are SHORT (about 71 one-minute rows, fault injected at minute ~35), so the rolling
    # baseline uses a shorter memory here. The limit cannot be calibrated on 35 minutes: we try a few fixed ones.
    W, SM, LIMITS = settings.RCAEVAL_WINDOW, settings.RCAEVAL_SMOOTH, settings.RCAEVAL_LIMITS
    results = []                                   # one row per case and limit
    for c in cases:
        try:
            m, inject = deploy_data.load_rcaeval(c)
        except Exception as e:
            print(f"  {c.name}: skipped ({type(e).__name__}: {e})")
            continue
        ts = pd.DatetimeIndex(m["timestamp"])
        before_inj, after = np.asarray(ts < inject), np.asarray((ts >= inject) & (ts < inject + check))
        if before_inj.sum() < W + SM + 10 or not after.any():
            print(f"  {c.name}: skipped (only {int(before_inj.sum())} points before the injection)")
            continue
        det = make_detector("rolling"); det.window, det.smooth = W, SM
        parts = c.name.split("_")                       # re1ob_<service>_<fault>_<n>
        service, fault = (parts[1], parts[2]) if len(parts) >= 4 else ("", "?")
        cols = [k for k in m.columns if k != "timestamp" and k.startswith(service + "_")] or \
               [k for k in m.columns if k != "timestamp"]       # the injected service's own metrics
        worst = np.max([det.score(m[col].values) for col in cols], axis=0)
        for lim in LIMITS:
            results.append({"limit": lim, "fault": fault, "caught": bool(worst[after].max() > lim),
                            "early_flag": bool((worst[before_inj] > lim).any())})
            results[-1]["fresh"] = results[-1]["caught"] and not results[-1]["early_flag"]
    if not results:
        print("\nRCAEval: no usable case")
        return
    r = pd.DataFrame(results)
    n = r[r.limit == LIMITS[0]].shape[0]
    print(f"\nRCAEval: {n} cases, rolling baseline with a {W * 5 // 60}-minute memory (5 s points), fault noticed within "
          f"{settings.DEPLOY_CHECK_MIN} min of the injection")
    print("fresh = caught AND quiet before the injection (the honest number)")
    print(r.groupby("limit").agg(caught=("caught", "mean"), flagged_before_injection=("early_flag", "mean"), fresh=("fresh", "mean")).round(2).to_string())
    print("\nfresh catches, by injected fault type:")
    print(r.pivot_table(index="fault", columns="limit", values="fresh", aggfunc="mean").round(2).to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    tests = {"nyc": test_nyc, "psi": test_psi, "hdfs": test_hdfs, "deploy": test_deploy,
             "flaky": lambda: test_flaky(sys.argv[2] if len(sys.argv) > 2 else None)}
    if what != "all" and what not in tests:
        sys.exit(f"unknown test '{what}'. Choose from {list(tests)} or 'all'")
    for name, fn in tests.items():
        if what in ("all", name):
            print(f"\n################ {name} ################")
            fn()