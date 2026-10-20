"""Step 6 (optional): build a small HTML page that REPLAYS real scenarios through the trained detectors.

    python -m scripts.06_scenario_page            ->  reports/scenario_player.html   (double-click to open)

Nothing is faked: the page shows what the detectors really computed on your data and the exact
Alert (the JSON Triage would receive) for each scenario. Needs the models from 02_train.
"""
import json
import pickle
import numpy as np
import pandas as pd
from config import settings
from ingest.hdfs import load_event_matrix, split_blocks
from agents.monitor.ci_monitor import CIMonitor
from agents.monitor.factory import slice_period
from agents.monitor.log_monitor import LogMonitor
from agents.monitor.series_monitor import SeriesMonitor

TEMPLATE = settings.ROOT / "scripts" / "scenario_page_template.html"
OUT = settings.REPORTS_DIR / "scenario_player.html"


def load(name):
    with open(settings.MODELS_DIR / name, "rb") as f:
        return pickle.load(f)


def clean(x):
    """numbers/dates -> plain JSON (NaN becomes null)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [clean(v) for v in x]
    if isinstance(x, (pd.Timestamp,)):
        return str(x)[:16]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def alert_json(a):
    """The Alert as P4's node sends it."""
    return clean({**a.to_contract(), "timestamp": a.timestamp.isoformat(), "kind": a.kind,
                  "dataset": a.dataset, "details": a.details})



# ------------------------------------------------------------------ time-series scenarios
def series_alerts(dataset, bundle, df):
    return list(SeriesMonitor(bundle, dataset=dataset, min_votes=settings.MIN_VOTES,
                                max_gap_hours=settings.MERGE_GAP_HOURS[dataset]).run(df))


def nyc_anomaly(models):
    bundle = models["nyc"]
    df = slice_period(pd.read_parquet(settings.SERIES_FILES["nyc"]), *settings.SPLITS["nyc"]["test"])
    alerts = [a for a in series_alerts("nyc", bundle, df) if a.kind == "anomaly"]
    examples = []
    for feature in settings.FEATURES["nyc"]:
        det = bundle[feature]["seasonal"]
        mine = sorted([a for a in alerts if a.feature == feature], key=lambda a: -a.score)[:2]
        d = df[["timestamp", feature]].dropna().reset_index(drop=True)
        t = pd.DatetimeIndex(d["timestamp"])
        med, mad = det.baseline(t)
        score = det.score(d[feature].values, t)
        for a in mine:
            c = int(np.searchsorted(t, a.timestamp))
            i0, i1 = max(c - 72, 0), min(c + 36, len(d))
            s = slice(i0, i1)
            examples.append({
                "label": f"{feature} · {str(a.timestamp)[:10]} (score {a.score:.1f}×)",
                "x": [str(v)[:16] for v in t[s]], "signal": d[feature].values[s], "signal_name": feature,
                "lo": med[s] - det.threshold * mad[s], "hi": med[s] + det.threshold * mad[s],
                "ratio": score[s] / det.threshold,
                "alerts": [{"i": c - i0, "alert": alert_json(a)}],
                "verdict": f"The value left the normal band for its weekday-hour. One alert, merged from {a.details['n_points']} hour(s)."})
    return {"id": "nyc", "tab": "Metric anomaly", "kind": "series", "unit": "hour",
            "title": "Metric anomaly: a taxi feature leaves its normal band",
            "blurb": ["Grey band = what is normal for THIS weekday-hour (seasonal median ± limit).",
                      "Bottom curve = score divided by the limit. When it crosses 1.0 the Monitor builds an Alert."],
            "examples": examples}


def drift(models):
    bundle = models["nyc"]
    df = slice_period(pd.read_parquet(settings.SERIES_FILES["nyc"]), *settings.SPLITS["nyc"]["test"])
    alerts = [a for a in series_alerts("nyc", bundle, df) if a.kind == "drift"]
    examples = []
    for feature, days in settings.DRIFT["nyc"].items():
        det = bundle[feature]["psi"]
        d = df[["timestamp", feature]].dropna().reset_index(drop=True)
        t = pd.DatetimeIndex(d["timestamp"])
        v = d[feature].values
        ratio = det.score(v) / det.threshold
        step = np.arange(0, len(d), 6)                     # one point every 6 hours keeps the page light
        w = det.window
        bins = []
        for j in step:
            win = v[max(j - w + 1, 0): j + 1]
            win = win[~np.isnan(win)]
            bins.append(np.bincount(det._bin(win), minlength=len(det.ref_p_)) / max(len(win), 1) if len(win) else np.zeros(len(det.ref_p_)))
        mine = [a for a in alerts if a.feature == feature]
        examples.append({
            "label": f"{feature} · PSI window {days} days · {len(mine)} drift alert(s)",
            "x": [str(v_)[:16] for v_ in t[step]], "signal": pd.Series(v).rolling(24, min_periods=1).mean().values[step],
            "signal_name": f"{feature} (daily mean)", "lo": None, "hi": None, "ratio": ratio[step],
            "alerts": [{"i": int(np.searchsorted(t[step], a.timestamp)), "alert": alert_json(a)} for a in mine],
            "bins": {"ref": det.ref_p_, "at": bins, "title": f"Share of the last {days} days in each reference bin"},
            "verdict": "PSI watches the SHAPE of the distribution: the bars at the right pile up in a few bins when the data drifts."})
    return {"id": "drift", "tab": "Data drift", "kind": "series", "unit": "6 hours",
            "title": "Data drift: the model inputs no longer look like the training year",
            "blurb": ["The training year is cut into 10 equally populated bins (reference = 10% each).",
                      "Each step compares the last window with that reference; the bars show where the values now fall."],
            "examples": examples}


def deploys(models):
    bundle = models["deploy"]
    df = pd.read_parquet(settings.SERIES_FILES["deploy"])
    ev = pd.read_parquet(settings.DEPLOY_EVENTS_FILE)
    sp = settings.SPLITS["deploy"]["test"]
    test = slice_period(df, *sp).reset_index(drop=True)
    ev = ev[(ev.timestamp >= pd.Timestamp(sp[0], tz="UTC")) & (ev.timestamp < pd.Timestamp(sp[1], tz="UTC"))]
    alerts = series_alerts("deploy", bundle, test)
    t = pd.DatetimeIndex(test["timestamp"])
    check = pd.Timedelta(minutes=settings.DEPLOY_CHECK_MIN)

    def alert_after(e):
        mine = [a for a in alerts if e.timestamp <= a.timestamp < e.timestamp + check]
        return min(mine, key=lambda a: a.timestamp) if mine else None

    rows = [(e, alert_after(e)) for e in ev.itertuples()]
    pick, used = [], set()

    def take(pred, label):
        for e, a in rows:
            if pred(e, a) and e.Index not in used:
                used.add(e.Index); pick.append((label, e, a)); return

    take(lambda e, a: e.kind == "slow" and e.size == 80, "BAD deploy: latency +80%")
    take(lambda e, a: e.kind == "errors" and e.size == 40, "BAD deploy: errors ×2.2")
    take(lambda e, a: e.kind == "leak" and e.size == 80, "BAD deploy: slow leak +80%")
    take(lambda e, a: e.kind == "leak" and e.size == 20 and a is None, "BAD deploy: slow leak +20% (missed)")
    take(lambda e, a: e.kind == "harmless" and a is None, "HARMLESS deploy: restart blip")
    take(lambda e, a: e.kind == "harmless" and a is not None, "HARMLESS deploy: false alarm")
    examples = []
    for label, e, a in pick:
        c = int(np.searchsorted(t, e.timestamp))
        i0, i1 = max(c - 90, 0), min(c + 180, len(test))
        s = slice(i0, i1)
        feature = "error_rate" if e.kind == "errors" else "latency_ms"
        det = bundle[feature]["rolling"]
        now, base, spread = det.baseline(test[feature].values)
        ratio = np.zeros(i1 - i0)
        for f in settings.FEATURES["deploy"]:                      # worst of the two metrics
            d_ = bundle[f]["rolling"]
            ratio = np.maximum(ratio, d_.score(test[f].values)[s] / d_.threshold)
        alert_i = int(np.searchsorted(t[s], a.timestamp)) if a else None
        delay = f"{int((a.timestamp - e.timestamp) / pd.Timedelta('1min'))} min" if a else None
        if e.bad:
            verdict = (f"Caught {delay} after the deploy." if a else
                       "Missed: a slow 2-hour ramp changes too little per minute, so the rolling baseline follows it. This is the known weakness."
                       if e.kind == "leak" else "Missed: the change was too small compared with the normal noise of this metric.")
        else:
            verdict = "Correctly ignored: the short restart blip is smoothed away." if a is None else \
                "False alarm on a harmless deploy: the price of catching bad ones early."
        examples.append({
            "label": label, "x": [str(v)[11:16] for v in t[s]], "signal": test[feature].values[s], "signal_name": feature,
            "lo": (base - det.threshold * spread)[s], "hi": (base + det.threshold * spread)[s], "ratio": ratio,
            "markers": [{"i": c - i0, "label": "deploy"}], "window": {"i0": c - i0, "i1": min(c - i0 + settings.DEPLOY_CHECK_MIN, i1 - i0 - 1), "label": "30 min to react"},
            "alerts": [{"i": alert_i, "alert": alert_json(a)}] if a else [], "verdict": verdict})
    return {"id": "deploy", "tab": "Bad deployment", "kind": "series", "unit": "minute",
            "title": "Bad deployment: metrics after a release vs the hour before it",
            "blurb": ["Band = typical value of the last hour ± the limit. A bad release pushes the metric out of it.",
                      "Harmless deploys also wiggle the metric: the detector must stay quiet for them (simulated data)."],
            "examples": examples}


# ------------------------------------------------------------------ CI + logs scenarios
def flaky_scenario():
    builds = pd.read_parquet(settings.BUILDS_FILE)
    since = builds["started"].max() - pd.Timedelta(days=settings.CI_REPLAY_DAYS)
    alerts = sorted(CIMonitor(load("flaky_model.pkl")).run(builds, since), key=lambda a: -a.score)
    if not alerts:
        return None
    pick = [("most likely FLAKY", alerts[0]), ("likely flaky", alerts[min(len(alerts) // 20, len(alerts) - 1)]),
            ("typical failure", alerts[len(alerts) // 2]), ("most likely REAL failure", alerts[-1])]
    labels = builds.set_index("build_id")["label"]
    ex = []
    for name, a in pick:
        d = a.details
        ex.append({"label": f"{name} · {d['project']}", "score": a.score, "prob": d["flaky_probability"], "project": d["project"],
                   "build_id": d["build_id"], "evidence": d["evidence"], "tests_failed": d["tests_failed"],
                   "truth": str(labels.get(d["build_id"], "?")), "alert": alert_json(a)})
    return {"id": "ci", "tab": "Flaky test", "kind": "flaky", "title": "Flaky test: a failed build gets a flaky probability",
            "blurb": ["Every FAILED build becomes an Alert. The score is how many times more likely than a typical failure it is to be flaky.",
                      "Score ≥ 1: probably flaky (a re-run may pass). Score < 1: probably a real failure. The decision is Triage's, not ours."],
            "examples": ex}


def logs_scenario():
    m = load("hdfs_block_pca.pkl")
    det, events = m["detector"], m["events"]
    ids, X, y, _ = load_event_matrix()
    test = np.flatnonzero(~split_blocks(len(y)))
    score = det.score(X[test]) / det.threshold
    fail, ok = test[y[test] == 1], test[y[test] == 0]
    sf, so = score[y[test] == 1], score[y[test] == 0]
    chosen = [("FAILING block, strongly flagged", fail[np.argmax(sf)]),
              ("FAILING block, just above the limit", fail[np.argsort(np.abs(sf - 1.3))[0]]),
              ("NORMAL block, quiet", ok[np.argmin(so)]),
              ("NORMAL block, false alarm", ok[np.argmax(so)])]
    ex = []
    for name, i in chosen:
        A = det._prep(X[[i]])
        rec = det.pca_.inverse_transform(det.pca_.transform(A))
        alert = next(iter(LogMonitor(det, events).run(ids[[i]], X[[i]])), None)
        ex.append({"label": f"{name} · {ids[i]}", "block": str(ids[i]), "truth": "Fail" if y[i] else "Success",
                   "counts": A[0], "recon": rec[0], "events": events, "ratio": float(det.score(X[[i]])[0] / det.threshold),
                   "alert": alert_json(alert) if alert else None})
    return {"id": "logs", "tab": "Unknown error (logs)", "kind": "logs", "title": "Unknown error: a log block that does not fit the normal pattern",
            "blurb": ["PCA learned which event counts usually go together in NORMAL blocks and rebuilds each block from that.",
                      "Bars: what the block did (log scale) vs what normal would look like. A big gap = unknown error. The biggest gaps become top_events."],
            "examples": ex}


def main():
    models = load("monitor_state.pkl")
    scenarios = [nyc_anomaly(models), drift(models), deploys(models), flaky_scenario(), logs_scenario()]
    data = {"scenarios": [s for s in scenarios if s], "built_from": str(settings.ROOT),
            "check_min": settings.DEPLOY_CHECK_MIN}
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(clean(data)))
    OUT.write_text(html, encoding="utf-8")
    n = sum(len(s["examples"]) for s in data["scenarios"])
    print(f"{len(data['scenarios'])} scenarios, {n} examples -> {OUT}\nOpen it in your browser (double-click).")


if __name__ == "__main__":
    main()
