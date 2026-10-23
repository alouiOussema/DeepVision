"""Measure Triage against ground truth exported by P1 (data/triage_eval/).

Baseline = no triage: every alert pages a human.

  A. nyc planted anomalies  -> do real problems still reach Diagnosis after Triage?
       caught_by_monitor : planted events that got at least one alert
       paged             : ... and that alert ended up in an INCIDENT
       triage_recall     : paged / caught_by_monitor   (what Triage loses; must stay high)
       pages             : incidents sent to Diagnosis
       pages_on_planted  : incidents that contain a planted event (the rest are real-data
                           anomalies, e.g. New Year: not wrong, just not labelled)
  B. hdfs block labels      -> are the HDFS incidents real failures?
  C. ci build labels        -> are the builds sent to "auto re-run" really flaky?
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import paths
from common.schemas import Alert, Decision

from .agent import RuleTriage
from .config import TriageConfig
from .io import load_alerts

EVAL = paths.EVAL


def _overlaps(a: Alert, start, end) -> bool:
    return a.start <= end and a.end >= start


# ------------------------------------------------------------------ A. nyc planted
def eval_planted(feature: str, cfg: TriageConfig | None = None, folder: Path = EVAL, post=None) -> dict:
    """post: optional function incidents -> incidents run after the rules (e.g. the LLM layer)."""
    alerts = load_alerts(folder / f"planted_alerts_nyc_{feature}.csv")
    labels = pd.read_csv(folder / f"planted_labels_nyc_{feature}.csv", parse_dates=["start", "end"])
    triage = RuleTriage(cfg)
    incidents = triage.process(alerts)
    if post:
        incidents = post(incidents)
    decision = {a.id: i.decision for i in incidents for a in i.alerts}

    rows = []
    for ev in labels.itertuples():
        hits = [a for a in alerts if a.kind == "anomaly" and _overlaps(a, ev.start, ev.end)]
        decs = {decision[a.id] for a in hits}
        best = (Decision.INCIDENT if Decision.INCIDENT in decs else
                Decision.WATCH if Decision.WATCH in decs else
                Decision.NOISE if decs else None)
        rows.append({"kind": ev.kind, "size": ev.size, "caught": bool(hits),
                     "paged": best == Decision.INCIDENT, "watch": best == Decision.WATCH,
                     "noise": best == Decision.NOISE})
    ev = pd.DataFrame(rows)

    sent = [i for i in incidents if i.decision == Decision.INCIDENT]
    on_planted = sum(any(_overlaps(a, r.start, r.end) for a in i.alerts for r in labels.itertuples())
                     for i in sent)
    caught = int(ev["caught"].sum())
    return {
        "feature": feature, "alerts": len(alerts), "planted_events": len(ev),
        "caught_by_monitor": caught, "paged": int(ev["paged"].sum()),
        "watch": int(ev["watch"].sum()), "noise": int(ev["noise"].sum()),
        "triage_recall": round(ev["paged"].sum() / caught, 3) if caught else None,
        "pages": len(sent), "pages_on_planted": on_planted,
        "page_reduction_%": round(100 * (1 - len(sent) / len(alerts)), 1),
        "by_kind": ev[ev["caught"]].groupby("kind")[["paged", "watch", "noise"]].mean().round(2),
    }


# ------------------------------------------------------------------ B. hdfs
def eval_hdfs(alerts: list[Alert], cfg: TriageConfig | None = None, folder: Path = EVAL, post=None) -> dict:
    lab = pd.read_csv(folder / "hdfs_labels.csv").set_index("block_id")["failing"].to_dict()
    logs = [a for a in alerts if a.family == "logs"]
    incidents = RuleTriage(cfg).process(logs)
    if post:
        incidents = post(incidents)
    rows = []
    for i in incidents:
        y = [lab.get(a.details["block_id"]) for a in i.alerts]
        rows.append({"pattern": i.alerts[0].details and "+".join(sorted(e["event"] for e in
                     i.alerts[0].details["top_events"])), "decision": i.decision.value,
                     "blocks": len(y), "failing": sum(v == 1 for v in y)})
    df = pd.DataFrame(rows)
    df["precision"] = (df["failing"] / df["blocks"]).round(3)
    sent = df[df["decision"] == "incident"]
    return {
        "alerted_blocks": len(logs), "failing_among_alerted": int(df["failing"].sum()),
        "monitor_precision": round(df["failing"].sum() / df["blocks"].sum(), 3),
        "incidents": len(sent),
        "failing_blocks_paged": int(sent["failing"].sum()),
        "recall_of_failing": round(sent["failing"].sum() / max(df["failing"].sum(), 1), 3),
        "precision_paged": round(sent["failing"].sum() / max(sent["blocks"].sum(), 1), 3),
        "per_pattern": df.sort_values("blocks", ascending=False),
    }


# ------------------------------------------------------------------ C. ci
def eval_ci(alerts: list[Alert], cfg: TriageConfig | None = None, folder: Path = EVAL) -> dict:
    cfg = cfg or TriageConfig()
    lab = pd.read_csv(folder / "ci_labels.csv").set_index("build_id")["label"].to_dict()
    ci = [a for a in alerts if a.family == "ci"]
    flaky = {a.id for a in ci if lab.get(a.details.get("build_id")) == "flaky_fail"}
    rerun = {a.id for a in ci if a.ratio >= cfg.ci_flaky_ratio}
    return {
        "failed_builds": len(ci), "really_flaky": len(flaky),
        "sent_to_rerun": len(rerun), "rerun_really_flaky": len(rerun & flaky),
        "rerun_precision": round(len(rerun & flaky) / max(len(rerun), 1), 3),
        "flaky_recall": round(len(rerun & flaky) / max(len(flaky), 1), 3),
        "flaky_base_rate": round(len(flaky) / max(len(ci), 1), 3),
    }


def sweep_planted(features=("trip_count", "avg_fare", "avg_distance"),
                  noise_ratios=(1.1, 1.2, 1.3, 1.5), medium_ratios=(1.3, 1.5, 2.0)):
    """Try several limits on the planted data: keep triage_recall high, pages low."""
    out = []
    for nr in noise_ratios:
        for mr in medium_ratios:
            rs = [eval_planted(f, TriageConfig(noise_ratio=nr, medium_ratio=mr)) for f in features]
            caught = sum(r["caught_by_monitor"] for r in rs)
            out.append({"noise_ratio": nr, "medium_ratio": mr,
                        "triage_recall": round(sum(r["paged"] for r in rs) / caught, 3),
                        "pages": sum(r["pages"] for r in rs),
                        "pages_on_planted": sum(r["pages_on_planted"] for r in rs)})
    return out