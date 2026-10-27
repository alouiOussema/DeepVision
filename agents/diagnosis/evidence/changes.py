"""Did something CHANGE just before the incident, and WHICH change? The first question of any on-call.

    deploy shortly before   -> bad_deploy      (stronger if the effect came right after)
    config push before      -> config_change
    nothing changed         -> not our release: infra or unknown

Which change to blame (the SUSPECT) matters as much as the cause: P5 rolls back exactly that one.
Blaming the latest change is wrong when the degradation began earlier (a harmless release landing
on top of a leak would be rolled back for nothing). So when raw metrics exist, we first find the
ONSET, the minute the metrics started to degrade (metrics.onset), and blame the last change before it:

    release A ........ onset ........ release B ........ alert
                ^ suspect = A               B came after the problem started: not guilty
"""
from __future__ import annotations

from datetime import timedelta

from common.schemas import Category as C, Incident

from ..config import DiagnosisConfig
from ..context import Change, ContextStore
from .metrics import onset as find_onset

TOLERANCE = timedelta(minutes=5)      # the onset estimate is about this precise (noisy error rate)


def _weights(c: Change, quick: bool, full: bool) -> dict:
    scale = 1.0 if full else 0.3
    if c.kind == "deploy":
        return {C.BAD_DEPLOY: (1.8 if quick else 1.2) * scale, C.INFRA: -0.8 * scale, C.UNKNOWN: -0.5 * scale}
    return {C.CONFIG_CHANGE: (2.0 if quick else 1.3) * scale, C.INFRA: -0.8 * scale, C.UNKNOWN: -0.5 * scale}


def _name(c: Change) -> str:
    return f"{'Release' if c.kind == 'deploy' else 'Config push'} {c.id}"


def collect(inc: Incident, ctx: ContextStore, cfg: DiagnosisConfig, f) -> None:
    if not ctx.changes:
        f.add("change_log", "No change log available: deploys and config pushes cannot be checked.", "change_log")
        return
    recent = ctx.changes_before(inc.start, cfg.change_window_min)
    if not recent:
        f.features["no_change"] = 1.0
        f.add("no_change", f"No deploy or config push in the {cfg.change_window_min:g} min before the incident "
                           f"started ({inc.start:%m-%d %H:%M}).", "change_log",
              {C.BAD_DEPLOY: -2.5, C.CONFIG_CHANGE: -2.5, C.INFRA: 1.0, C.UNKNOWN: 0.3})
        return

    f.onset = find_onset(ctx, inc, recent[-1].timestamp, cfg)
    if f.onset is not None:
        guilty = [c for c in recent if c.timestamp <= f.onset + TOLERANCE]
        if not guilty:        # it was already going wrong before the first change
            f.features["no_change"] = 1.0
            f.add("degraded_before_changes", f"Metrics were already degrading at {f.onset:%H:%M}, before "
                                             f"{_name(recent[-1])} ({recent[-1].timestamp:%H:%M}): no change in "
                                             "the window explains it.", "metrics",
                  {C.BAD_DEPLOY: -1.5, C.CONFIG_CHANGE: -1.5, C.INFRA: 1.0}, onset=str(f.onset))
            return
        f.suspect = guilty[0]
    else:
        f.suspect = recent[0]

    s = f.suspect
    mins = (inc.start - s.timestamp).total_seconds() / 60
    quick = mins <= cfg.quick_effect_min
    if s.kind == "deploy":
        f.features.update(has_deploy=1.0, mins_since_deploy=mins)
    else:
        f.features.update(has_config=1.0, mins_since_config=mins)
    when = f"{mins:.0f} min {'right after' if quick else 'after'}"
    onset = f"; metrics started to degrade at {f.onset:%H:%M}" if f.onset is not None else ""
    f.add(f"{s.kind}_before", f"{_name(s)} went out at {s.timestamp:%m-%d %H:%M}; the incident started {when}"
                              f"{onset}.", "change_log", _weights(s, quick, True),
          change_id=s.id, minutes=round(mins, 1))

    later = [c for c in recent if c.timestamp > s.timestamp]
    if later:     # changes after the suspect: mentioned, cleared
        f.add("later_changes", f"{', '.join(_name(c) for c in later)} came after the degradation started: "
                               f"not the cause, do not roll back {'them' if len(later) > 1 else 'it'}.",
              "change_log", {}, cleared=[c.id for c in later])
    for c in recent:
        if c.timestamp < s.timestamp and c.kind != s.kind:   # an older change of the other kind: a weak suspect
            q = (inc.start - c.timestamp).total_seconds() / 60 <= cfg.quick_effect_min
            f.add(f"{c.kind}_before", f"{_name(c)} also went out before ({c.timestamp:%m-%d %H:%M}), "
                                      "but it is not the most likely trigger.", "change_log",
                  _weights(c, q, False), change_id=c.id)
            break
