"""Steps 2 and 3 of Triage: measure each group, then give it a severity and a decision.

    measure(group)  -> Measures   (numbers that describe the group)
    judge(group)    -> (severity, decision, reasons)

One rule set per family, because the score does not mean the same thing:
  series : high ratio = far from normal              -> more serious
  logs   : many blocks with the same pattern         -> more serious
  ci     : high ratio = probably FLAKY (re-run it)   -> NOT more serious

These functions are also the TOOLS the LLM agent will call later, so they stay
small, deterministic and explained.
"""
from __future__ import annotations

from dataclasses import dataclass

from common.schemas import Decision, Severity

from .config import TriageConfig
from .grouping import Group


# ------------------------------------------------------------------ measures
@dataclass
class Measures:
    n_alerts: int
    peak_ratio: float         # how strong
    width: int                # how many different features
    duration_hours: float     # how long (first start -> last end)
    max_voters: int           # how many detectors agreed at best
    drift: bool               # every alert is a slow drift, not a sudden anomaly
    n_real: int = 0           # ci only: failures that look real (ratio < ci_flaky_ratio)
    n_flaky: int = 0          # ci only: failures that look flaky
    fail_rate_before: float | None = None   # ci only: project's recent fail rate before the first failure

    def text(self) -> str:
        return (f"{self.n_alerts} alert(s), strength {self.peak_ratio:.2f}, {self.width} feature(s), "
                f"{self.duration_hours:g} h, up to {self.max_voters} detector(s) agree")


def measure(g: Group, cfg: TriageConfig) -> Measures:
    a = g.alerts
    start = min(x.start for x in a)
    end = max(x.end for x in a)
    step = min(x.duration_hours / int(x.details.get("n_points", 1)) for x in a)   # 1 h or 1 min
    real = sorted([x for x in a if x.ratio < cfg.ci_flaky_ratio], key=lambda x: x.timestamp) \
        if g.family == "ci" else []
    before = real[0].details.get("evidence", {}).get("recent_fail_rate") if real else None
    return Measures(
        n_alerts=len(a),
        peak_ratio=max(x.ratio for x in a),
        width=len({x.feature for x in a}),
        duration_hours=round((end - start).total_seconds() / 3600 + step, 2),
        max_voters=max(len(x.voters) for x in a),
        drift=all(x.kind == "drift" for x in a),
        n_real=len(real),
        n_flaky=len(a) - len(real) if g.family == "ci" else 0,
        fail_rate_before=before,
    )


# ------------------------------------------------------------------ A. series
def _judge_series(m: Measures, cfg: TriageConfig):
    if m.drift:   # the data slowly changed: a retraining question, not an emergency
        sev = Severity.MEDIUM if m.peak_ratio >= cfg.medium_ratio else Severity.LOW
        return sev, Decision.WATCH, f"drift (strength {m.peak_ratio:.2f}): data changed slowly, check before retraining"

    strong = m.peak_ratio >= cfg.medium_ratio        # width/duration only escalate real signals
    if m.peak_ratio >= cfg.critical_ratio:
        sev, why = Severity.CRITICAL, f"very strong anomaly (strength {m.peak_ratio:.1f})"
    elif strong and m.width >= 2 and m.duration_hours >= 3:
        sev, why = Severity.CRITICAL, f"{m.width} features abnormal for {m.duration_hours:g} h"
    elif m.peak_ratio >= cfg.high_ratio:
        sev, why = Severity.HIGH, f"strong anomaly (strength {m.peak_ratio:.1f})"
    elif strong and m.width >= 2:
        sev, why = Severity.HIGH, f"{m.width} features abnormal together"
    elif strong or m.max_voters >= 2 or m.width >= 2 or m.duration_hours >= 2:
        sev, why = Severity.MEDIUM, "clear but limited anomaly"
    else:
        sev, why = Severity.LOW, f"weak anomaly (strength {m.peak_ratio:.2f})"

    if sev in (Severity.HIGH, Severity.CRITICAL):
        return sev, Decision.INCIDENT, why
    if m.max_voters >= 2:
        return sev, Decision.INCIDENT, f"{why}; {m.max_voters} detectors agree"
    if strong and m.duration_hours >= 2:
        return sev, Decision.INCIDENT, f"{why}; clear signal lasting {m.duration_hours:g} h"
    if m.width >= 2 or m.duration_hours >= 2:
        return sev, Decision.WATCH, f"{why}; ambiguous: weak but {m.width} feature(s) / {m.duration_hours:g} h"
    if m.peak_ratio < cfg.noise_ratio:
        return sev, Decision.NOISE, f"{why}; below {cfg.noise_ratio}, 1 feature, short"
    return sev, Decision.WATCH, f"{why}; ambiguous: 1 feature, short"


# ------------------------------------------------------------------ B. logs
def _judge_logs(m: Measures, cfg: TriageConfig):
    n = m.n_alerts
    if n >= cfg.logs_incident_blocks and m.peak_ratio >= cfg.critical_ratio and n >= 100:
        sev = Severity.CRITICAL
    elif n >= 100:
        sev = Severity.HIGH
    elif n >= cfg.logs_incident_blocks:
        sev = Severity.MEDIUM
    else:
        sev = Severity.LOW
    why = f"{n} block(s) with this pattern, strongest {m.peak_ratio:.1f}"
    if n >= cfg.logs_incident_blocks:
        return sev, Decision.INCIDENT, f"{why}: recurring failure"
    if m.peak_ratio < cfg.logs_noise_ratio:
        return sev, Decision.NOISE, f"{why}: rare and weak"
    # measured on P1's labels: rare but strong patterns were real failures in 81 of 82 blocks
    return sev, Decision.INCIDENT, f"{why}: rare but clearly abnormal"


# ------------------------------------------------------------------ C. ci
def _judge_ci(m: Measures, cfg: TriageConfig):
    real, flaky, before = m.n_real, m.n_flaky, m.fail_rate_before
    if real >= cfg.ci_incident_builds:
        if before is not None and before < cfg.ci_usually_green:
            return Severity.HIGH, Decision.INCIDENT, (f"{real} real-looking failures the same day on a project "
                                                      f"that usually passes (fail rate before: {before:.0%}): newly broken")
        return Severity.MEDIUM, Decision.WATCH, (f"{real} real-looking failures, but the project often fails "
                                                 f"(fail rate before: {'?' if before is None else f'{before:.0%}'}): not new")
    if real >= cfg.ci_watch_builds:
        return Severity.MEDIUM, Decision.WATCH, f"{real} real-looking failures the same day: keep an eye on it"
    if flaky:
        return Severity.LOW, Decision.INCIDENT, (f"{flaky} build(s) likely flaky (ratio >= {cfg.ci_flaky_ratio}): "
                                                 "candidate for automatic re-run, no human needed")
    return Severity.LOW, Decision.NOISE, "1 normal failed build: the developer is already notified by CI"


# ------------------------------------------------------------------ entry point
def judge(g: Group, cfg: TriageConfig) -> tuple[Severity, Decision, Measures, str]:
    m = measure(g, cfg)
    fn = {"series": _judge_series, "logs": _judge_logs, "ci": _judge_ci}[g.family]
    sev, dec, why = fn(m, cfg)
    return sev, dec, m, why