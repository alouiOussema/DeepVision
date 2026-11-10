"""Grades every incident against the injected truth and computes the KPIs of the presentation.

MTTD only exists for the deploy family (the others skip the Monitor). A false action is a fix run on the wrong
cause or release, or one that failed or was reverted.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import median

from common.schemas import Action, ActionStatus, Category, Decision, Route

from .injector import Episode, Injection

FIX_FOR = {Category.FLAKY_TEST: Action.RERUN, Category.BAD_DEPLOY: Action.ROLLBACK,
           Category.CONFIG_CHANGE: Action.ROLLBACK, Category.DATA_DRIFT: Action.RETRAIN}
FIXES = set(FIX_FOR.values())
RAN = {ActionStatus.EXECUTED.value, ActionStatus.FAILED.value, ActionStatus.REVERTED.value}


def _min(a: datetime, b: datetime) -> float:
    return round((a - b).total_seconds() / 60, 1)


@dataclass
class Record:
    """One incident, from Triage to the Action agent's report, next to the truth."""
    key: str
    episode: str
    family: str
    incident_id: str
    start: datetime
    n_alerts: int
    sent: bool
    triage: str
    injection: str = ""
    scored: bool = True
    truth: str = ""
    truth_sub: str = ""
    truth_suspect: str = ""
    is_fault: bool = False
    category: str = ""
    sub_cause: str = ""
    suspect: str = ""
    confidence: float | None = None
    action: str = ""
    route: str = ""
    target: str = ""
    status: str = ""
    rule: str = ""
    outcome: str = ""
    finished_at: datetime | None = None
    notified: list = field(default_factory=list)
    diagnosis_right: bool | None = None
    action_right: bool | None = None
    wrong_fix: bool = False
    silenced: bool = False
    closed_duplicate: bool = False
    needed_human: bool = False
    fixed: bool = False
    fixed_alone: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["finished_at"] = self.finished_at.isoformat() if self.finished_at else None
        return d


def _expected_fix(inj: Injection | None) -> Action | None:
    return FIX_FOR.get(inj.truth) if inj and inj.is_fault else None


def records(ep: Episode, state: dict, orders: list, key=lambda iid: iid) -> list[Record]:
    """orders: the run's inbox (final statuses). key: the tracer's key function."""
    diags = {d.incident_id: d for d in state.get("diagnoses", [])}
    final = {o.incident_id: o for o in orders}
    runs = {r.incident_id: r for r in state.get("actions_done", [])}
    out = []
    for inc in state.get("final") or state.get("incidents", []):
        inj = ep.injection_for(inc.alert_ids)
        r = Record(key(inc.id), ep.name, ep.family, inc.id, inc.start, len(inc.alert_ids),
                   inc.decision is Decision.INCIDENT, inc.decision.value)
        if inj:
            r.injection, r.truth, r.truth_sub, r.truth_suspect, r.is_fault, r.scored = (
                inj.id, inj.truth.value, inj.truth_sub, inj.truth_suspect, inj.is_fault, inj.scored)
        d, o, run = diags.get(inc.id), final.get(inc.id), runs.get(inc.id)
        if d:
            r.category, r.sub_cause, r.suspect, r.confidence = d.category.value, d.sub_cause, d.suspect, d.confidence
            if inj:
                r.diagnosis_right = (d.category is inj.truth
                                     and (not inj.truth_suspect or d.suspect == inj.truth_suspect))
        if o:
            r.action, r.route, r.target, r.status, r.rule = (o.action.value, o.route.value, o.target, o.status.value,
                                                             o.rule)
            r.needed_human = o.route in (Route.APPROVAL, Route.ESCALATE)
            expected = _expected_fix(inj)
            right_target = o.action is not Action.ROLLBACK or (inj is not None and o.target == inj.truth_suspect)
            if expected:
                r.action_right = o.action is expected and right_target
            elif inj and inj.is_fault:
                r.action_right = o.action is Action.ESCALATE
            else:
                r.action_right = o.action in (Action.NONE, Action.ESCALATE)
            closed = bool(inj and inj.is_fault and o.action is Action.NONE)
            r.closed_duplicate = closed and o.rule == "guard:duplicate"
            r.silenced = closed and not r.closed_duplicate
        if run:
            r.outcome, r.finished_at, r.notified = run.outcome, run.finished_at, list(run.notified)
        if r.action in {a.value for a in FIXES} and r.status in RAN:
            r.wrong_fix = not r.action_right or r.status != ActionStatus.EXECUTED.value
            r.fixed = bool(r.action_right) and r.status == ActionStatus.EXECUTED.value
            r.fixed_alone = r.fixed and r.route == Route.ACT.value
        out.append(r)
    return out


def _pct(n: int, total: int) -> float | None:
    return round(100 * n / total, 1) if total else None


def _med(xs: list[float]) -> float | None:
    return round(median(xs), 1) if xs else None


def kpis(recs: list[Record], injections: list[Injection]) -> dict:
    """The deck's KPIs for one family (or all of them), on the graded incidents and injections only."""
    recs = [r for r in recs if r.scored]
    injections = [i for i in injections if i.scored]
    n_alerts = sum(r.n_alerts for r in recs)
    sent = [r for r in recs if r.sent]
    graded = [r for r in sent if r.diagnosis_right is not None]
    fixes_run = [r for r in recs if r.action in {a.value for a in FIXES} and r.status in RAN]
    wrong = [r for r in fixes_run if r.wrong_fix]
    faults = [i for i in injections if i.is_fault]
    by_inj: dict[str, list[Record]] = defaultdict(list)
    for r in recs:
        by_inj[r.injection].append(r)

    mttd, mttr, resolved, alone, missed_monitor, missed_triage = [], [], 0, 0, 0, 0
    starts = {r.key: r.start for r in recs}
    for inj in faults:
        mine = by_inj.get(inj.id, [])
        if not inj.alert_ids:
            missed_monitor += 1
            continue
        if not any(r.sent for r in mine):
            missed_triage += 1
        if inj.fault_start and mine:
            mttd.append(_min(min(r.start for r in mine), inj.fault_start))
        done = [r for r in mine if r.fixed and r.finished_at]
        if done:
            resolved += 1
            alone += any(r.fixed_alone for r in done)
            first = min(done, key=lambda r: r.finished_at)
            mttr.append(_min(first.finished_at, inj.fault_start or starts[first.key]))

    fixable = [i for i in faults if FIX_FOR.get(i.truth)]
    return {
        "alerts": n_alerts,
        "incidents": len(recs),
        "sent_to_diagnosis": len(sent),
        "faults_injected": len(faults),
        "fixable_faults": len(fixable),
        "mttd_min": _med(mttd),
        "missed_by_monitor": missed_monitor,
        "missed_by_triage": missed_triage,
        "noise_reduction_pct": _pct(n_alerts - len(sent), n_alerts),
        "diagnosis_accuracy_pct": _pct(sum(bool(r.diagnosis_right) for r in graded), len(graded)),
        "fixes_run": len(fixes_run),
        "false_action_pct": _pct(len(wrong), len(fixes_run)),
        "wrong_auto_actions": sum(r.route == Route.ACT.value and not r.action_right for r in fixes_run),
        "wrong_but_reported_executed": sum(not r.action_right and r.status == ActionStatus.EXECUTED.value
                                           for r in fixes_run),
        "silenced_faults": sum(r.silenced for r in recs),
        "closed_as_duplicate": sum(r.closed_duplicate for r in recs),
        "resolved_pct": _pct(resolved, len(fixable)),
        "auto_resolved_pct": _pct(alone, len(fixable)),
        "mttr_min": _med(mttr),
        "human_pages": sum(r.needed_human for r in recs),
        "unneeded_human_pages": sum(r.needed_human and not r.is_fault for r in recs),
    }


def summarize(per_episode: list[tuple[Episode, list[Record]]]) -> dict:
    """KPIs per family and overall."""
    fam: dict[str, dict] = defaultdict(lambda: {"recs": [], "inj": []})
    for ep, recs in per_episode:
        for k in (ep.family, "all"):
            fam[k]["recs"] += recs
            fam[k]["inj"] += ep.injections
    order = [f for f in ("deploy", "ci", "model", "logs") if f in fam] + ["all"]
    return {f: kpis(fam[f]["recs"], fam[f]["inj"]) for f in order if f in fam}


ROWS = [("alerts", "alerts in", ""), ("sent_to_diagnosis", "incidents sent to Diagnosis", ""),
        ("noise_reduction_pct", "noise reduction", " %"), ("faults_injected", "faults injected", ""),
        ("mttd_min", "MTTD (median)", " min"), ("missed_by_monitor", "missed by the Monitor", ""),
        ("missed_by_triage", "kept as watch/noise by Triage", ""),
        ("diagnosis_accuracy_pct", "diagnosis accuracy", " %"), ("fixes_run", "fixes run", ""),
        ("false_action_pct", "false action rate", " %"), ("wrong_auto_actions", "wrong automatic actions", ""),
        ("wrong_but_reported_executed", "wrong fixes P5 verified as OK", ""),
        ("silenced_faults", "silenced faults", ""),
        ("closed_as_duplicate", "faults closed as a duplicate", ""), ("resolved_pct", "fixable faults resolved", " %"),
        ("auto_resolved_pct", "... with no human", " %"), ("mttr_min", "MTTR (median)", " min"),
        ("human_pages", "incidents that reached a person", ""), ("unneeded_human_pages", "... with nothing broken", "")]


def table(summary: dict) -> str:
    fams = list(summary)
    f = lambda v, u: "-" if v is None else f"{v}{u}"
    lines = [f"{'':34s}" + "".join(f"{x:>10s}" for x in fams)]
    for k, label, unit in ROWS:
        lines.append(f"{label:34s}" + "".join(f"{f(summary[x][k], unit):>10s}" for x in fams))
    return "\n".join(lines)
