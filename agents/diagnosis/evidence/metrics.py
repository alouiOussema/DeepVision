"""What did the raw metrics do around the incident? Gives the SUB-CAUSE of a metric incident.

The Monitor only says "latency left its band". Reading the minutes around it tells more:

    anchor  = the suspect change (or the incident start if nothing changed)
    before  = median of the `before_min` minutes before the anchor
    after   = the `after_min` minutes after it, cut in three parts
    change  = biggest relative change of the first or last third vs before
    ramp    = 1 - change(first third) / change(last third)
              ~0 -> a STEP (jumped at once)     ~1 -> a RAMP (keeps climbing: leak, saturation)

    latency step only        -> latency regression (slow release)
    errors step only         -> error regression / bad config
    both, slowly climbing    -> resource leak (after a release) or capacity saturation (no release)
    nothing moved any more   -> a short blip (harmless restart): nothing to blame
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from common.schemas import Category as C, Incident

from ..config import DiagnosisConfig
from ..context import ContextStore

SHORT = {"latency_ms": "lat", "error_rate": "err"}
ONSET_SMOOTH = 10     # minutes averaged to find the onset (the error rate is too noisy minute by minute)
RECOVERY_MIN = 30     # minutes compared on each side of the incident start to spot a recovery


def shape(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """(relative change, ramp score) of `after` compared with `before`."""
    base = float(np.median(before))
    if base <= 0 or len(after) < 3:
        return 0.0, 0.0
    first, _, last = (float(np.mean(p)) / base - 1 for p in np.array_split(after, 3))
    change = first if abs(first) > abs(last) else last
    ramp = float(np.clip(1 - first / last, 0, 1)) if last > 0 and last >= first else 0.0
    return change, ramp


def onset(ctx: ContextStore, inc: Incident, first_change: datetime, cfg: DiagnosisConfig) -> datetime | None:
    """When did the metrics start to degrade?

        level   = median of the hour before the first suspect change
        crossed = first minute the 10-minute mean is above level x (1 + moved limit)
        onset   = the first RAW minute above the limit inside that 10-minute window

    Measured on the train deploys (96 with a known guilty change): this simple estimator blames the
    right release 90 times; a change-point estimator (max of mean-after minus mean-before) only 69,
    because noisy error-rate minutes pull it too early. Simple wins (see README, "what we tried").
    None if no metric data, or if nothing moved enough to tell."""
    base = ctx.metric_window(inc.dataset, first_change - timedelta(minutes=cfg.before_min), first_change)
    look = ctx.metric_window(inc.dataset, first_change - timedelta(minutes=30), inc.end + timedelta(minutes=5))
    if base.empty or look.empty:
        return None
    found = []
    for col in SHORT:
        if col not in look:
            continue
        limit = float(np.median(base[col])) * (1 + cfg.moved_change.get(col, cfg.moved_default))
        raw = look[col].to_numpy(float)
        smooth = pd.Series(raw).rolling(ONSET_SMOOTH, min_periods=ONSET_SMOOTH).mean().to_numpy()
        above = np.flatnonzero(smooth > limit)
        if len(above):
            k = above[0]
            found.append(look["timestamp"].iloc[k - ONSET_SMOOTH + 1 + int(np.argmax(raw[k - ONSET_SMOOTH + 1: k + 1] > limit))])
    return min(found).to_pydatetime() if found else None


def recovery(ctx: ContextStore, inc: Incident, cfg: DiagnosisConfig) -> dict[str, float] | None:
    """Relative change across the incident start, if every metric that moved went DOWN (back to normal)."""
    before = ctx.metric_window(inc.dataset, inc.start - timedelta(minutes=RECOVERY_MIN), inc.start - timedelta(minutes=2))
    after = ctx.metric_window(inc.dataset, inc.start, inc.start + timedelta(minutes=RECOVERY_MIN))
    if before.empty or after.empty:
        return None
    rel = {c: float(after[c].mean() / max(before[c].mean(), 1e-12) - 1) for c in SHORT if c in before}
    limit = {c: cfg.moved_change.get(c, cfg.moved_default) for c in rel}
    if any(r < -limit[c] for c, r in rel.items()) and not any(r > limit[c] for c, r in rel.items()):
        return rel
    return None


def collect(inc: Incident, ctx: ContextStore, cfg: DiagnosisConfig, f) -> None:
    back = recovery(ctx, inc, cfg)
    if back is not None:      # the detector also fires when a degradation ENDS (rollback, fault over)
        f.features["recovery"] = 1.0
        f.sub_causes[C.UNKNOWN] = "recovery"
        f.add("metric_recovery", "At the incident start the metrics went DOWN ("
                                 + ", ".join(f"{c} {r:+.0%}" for c, r in back.items())
                                 + "): this is the end of an earlier degradation (rollback or fault over), "
                                   "not a new problem.", "metrics",
              {C.UNKNOWN: 2.5, C.BAD_DEPLOY: -1.5, C.CONFIG_CHANGE: -1.5, C.INFRA: -1.0}, **back)
        return
    if f.suspect is not None:                 # the change the changes collector blamed
        anchor = f.suspect.timestamp
    elif f.onset is not None:                 # degradation without a change: start a bit before it
        anchor = f.onset - timedelta(minutes=cfg.detect_lag_min)
    else:
        anchor = inc.start - timedelta(minutes=cfg.detect_lag_min)
    before = ctx.metric_window(inc.dataset, anchor - timedelta(minutes=cfg.before_min), anchor)
    after = ctx.metric_window(inc.dataset, anchor, anchor + timedelta(minutes=cfg.after_min))
    if before.empty or after.empty:
        f.add("metrics_missing", f"No raw metrics for {inc.dataset} around the incident: shape not checked.", "metrics")
        return

    moved, ramps, parts = [], [], []
    for col, short in SHORT.items():
        if col not in before:
            continue
        change, ramp = shape(before[col].to_numpy(float), after[col].to_numpy(float))
        is_moved = change > cfg.moved_change.get(col, cfg.moved_default)
        f.features.update({f"{short}_change": change, f"{short}_ramp": ramp, f"{short}_moved": float(is_moved)})
        parts.append(f"{col} {change:+.0%}" + (f" ({'ramp' if ramp > cfg.ramp_limit else 'step'})" if is_moved else ""))
        if is_moved:
            moved.append(col)
            ramps.append(ramp)
    where = f"after {f.suspect.id}" if f.suspect is not None else "around the incident start"
    numbers = ", ".join(parts)
    has_deploy = f.suspect is not None and f.suspect.kind == "deploy"

    if not moved:
        f.add("metric_blip", f"{where.capitalize()}, metrics stayed near their level of the hour before "
                             f"({numbers}): a short blip, not a lasting degradation.", "metrics",
              {C.BAD_DEPLOY: -1.2, C.CONFIG_CHANGE: -1.2, C.INFRA: -0.5, C.UNKNOWN: 1.2})
        return

    if len(moved) == 2 and np.mean(ramps) > cfg.ramp_limit:
        f.sub_causes.update({C.BAD_DEPLOY: "resource_leak", C.INFRA: "capacity_saturation"})
        f.add("metric_ramp", f"{where.capitalize()}, latency and errors kept climbing together ({numbers}): "
                             "the typical shape of a resource leak or saturation.", "metrics",
              {C.BAD_DEPLOY: 0.8 if has_deploy else 0.0, C.INFRA: 0.6, C.CONFIG_CHANGE: -0.6})
    elif moved == ["latency_ms"]:
        f.sub_causes.update({C.BAD_DEPLOY: "latency_regression", C.INFRA: "slow_dependency"})
        f.add("metric_step", f"{where.capitalize()}, only latency jumped ({numbers}): requests got slower, "
                             "they did not fail.", "metrics",
              {C.BAD_DEPLOY: 0.6 if has_deploy else 0.0, C.CONFIG_CHANGE: -0.3})
    elif moved == ["error_rate"]:
        f.sub_causes.update({C.BAD_DEPLOY: "error_regression", C.CONFIG_CHANGE: "bad_config",
                             C.INFRA: "dependency_errors"})
        f.add("metric_step", f"{where.capitalize()}, only the error rate jumped ({numbers}): requests fail "
                             "fast, which is typical of a wrong setting or a broken code path.", "metrics",
              {C.CONFIG_CHANGE: 0.6, C.BAD_DEPLOY: 0.3})
    else:
        f.sub_causes.update({C.BAD_DEPLOY: "latency_and_error_regression", C.INFRA: "capacity_saturation"})
        f.add("metric_step", f"{where.capitalize()}, latency and errors jumped together ({numbers}).", "metrics",
              {C.BAD_DEPLOY: 0.6 if has_deploy else 0.0, C.INFRA: 0.3})
