"""Why did the build fail? Flaky test, real regression, or the CI machines themselves?

    P1's flaky probability high             -> flaky_test
    project history full of flaky failures  -> flaky_test
    same commit fails again on re-run       -> code_regression (a flaky test usually passes the 2nd time)
    project usually green, now red          -> code_regression
    many OTHER projects red in the same hour -> infra (shared runners, network, package mirror)
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from common.schemas import Category as C, Incident

from ..config import DiagnosisConfig
from ..context import ContextStore


def _mean(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None and not np.isnan(float(x))]
    return float(np.mean(xs)) if xs else None


def collect(inc: Incident, ctx: ContextStore, cfg: DiagnosisConfig, f) -> None:
    d = [a.details for a in inc.alerts]
    project = d[0].get("project", "?")
    probs = [x.get("flaky_probability") for x in d]
    p_mean, p_max = _mean(probs), max((p for p in probs if p is not None), default=None)
    proj_rate = _mean(x.get("evidence", {}).get("project_flaky_rate") for x in d)
    recent = _mean(x.get("evidence", {}).get("recent_fail_rate") for x in d[:1])   # before the first failure
    prev_failed = _mean(x.get("evidence", {}).get("previous_build_failed") for x in d)
    commits = {x.get("commit") for x in d}
    f.features.update({k: v for k, v in dict(flaky_mean=p_mean, flaky_max=p_max, project_flaky_rate=proj_rate,
                                              recent_fail_rate=recent, prev_failed_share=prev_failed,
                                              n_commits=float(len(commits))).items() if v is not None})

    if p_mean is not None:
        if p_mean >= cfg.flaky_high:
            w, word = {C.FLAKY_TEST: 1.6, C.CODE_REGRESSION: -1.0}, "likely flaky"
        elif p_mean <= cfg.flaky_low:
            w, word = {C.CODE_REGRESSION: 1.2, C.FLAKY_TEST: -1.2}, "likely a real failure"
        else:
            w, word = {}, "undecided"
        f.add("flaky_probability", f"P1's flaky model gives {p_mean:.2f} on average (max {p_max:.2f}) over "
                                   f"{len(d)} failed build(s) of {project}: {word}.", "alert", w,
              mean=round(p_mean, 3), max=round(p_max, 3))
    if proj_rate is not None and proj_rate >= 0.4:
        f.add("project_history", f"{proj_rate:.0%} of {project}'s earlier failures were flaky.", "alert",
              {C.FLAKY_TEST: 0.8}, project_flaky_rate=round(proj_rate, 3))
    if len(d) >= 2 and len(commits) == 1:
        f.add("same_commit_fails", f"The same commit {next(iter(commits))[:8]} failed {len(d)} times: a re-run "
                                   "did not fix it.", "alert", {C.CODE_REGRESSION: 1.0, C.FLAKY_TEST: -0.8})
    if recent is not None and recent < 0.2:
        f.add("usually_green", f"{project} failed only {recent:.0%} of its recent builds before this: "
                               "the failure is news.", "alert", {C.CODE_REGRESSION: 0.6})

    # peers: other projects failing in the same hour -> the machines, not the code
    near = [p for p in ctx.peers if p.source == "ci" and p.id != inc.id
            and abs((p.start - inc.start).total_seconds()) <= timedelta(hours=1).total_seconds()]
    others = sorted({p.alerts[0].details.get("project") for p in near} - {project})
    f.features["peer_projects"] = float(len(others))
    if len(others) >= cfg.infra_projects - 1:
        f.sub_causes[C.INFRA] = "ci_runners"
        names = ", ".join(others[:4]) + ("..." if len(others) > 4 else "")
        f.add("many_projects_fail", f"{len(others)} other project(s) failed within the same hour ({names}): a "
                                    f"shared CI problem is more likely than {len(others) + 1} code bugs at once.",
              "alert", {C.INFRA: 2.4, C.FLAKY_TEST: -0.5, C.CODE_REGRESSION: -0.8}, projects=others)
    elif others:
        f.add("few_projects_fail", f"{len(others)} other project(s) also failed within the hour: probably "
                                   "unrelated.", "alert", {C.INFRA: 0.3}, projects=others)
