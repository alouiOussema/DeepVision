"""The Decision agent (P4): Diagnosis -> ActionDecision, by a policy written in code (no LLM).

Diagnosis names the cause and SUGGESTS an action. Decision multiplies the risk of that action by the
confidence of the diagnosis (the action risk table + a few team rules) and answers one question:
may the Action agent (P5) do it alone, does a human approve it first, or does a human take over?

The rule chain, first match wins. Every step adds one readable line to `reasons` (the decision trail):

    A  no-fault        recognised recovery / known false alarm, confident        -> NONE      (nobody paged)
    B  floor           confidence < 0.50                                         -> ESCALATE  (no action proposed)
    C  never alone     code_regression / infra / data_quality / unknown          -> ESCALATE
    D  preconditions   rollback without a suspect, rerun off CI, retrain off a model -> ESCALATE
    E  human           needs_human, "LLM argues for ...", or below the cause's bar -> APPROVAL
    F  risk gate       rerun -> ACT | rollback -> APPROVAL (team rule) | retrain -> ACT by day, deploy needs a human
    G  guards          rerun budget used -> ESCALATE (a real bug) | already open / just done -> CLOSED (duplicate)
    H  safety valves   kill switch, dry run, circuit breaker -> an ACT becomes APPROVAL

The table is read from knowledge_base/action_risk_table.md (policy.py): the bars are not duplicated here.
"""
from __future__ import annotations

from datetime import datetime

from common.schemas import (Action, ActionDecision, ActionStatus, Category, Diagnosis, Incident,
                            Route)

from . import policy as policy_mod
from .config import DecisionConfig
from .log import DecisionLog, action_key, utcnow
from .policy import PolicyRow

EXECUTABLE = {Action.RERUN, Action.ROLLBACK, Action.RETRAIN}      # actions that change something
NO_FAULT = {"recovery", "known_false_alarm"}                      # sub-causes Diagnosis may call "no fault"
LLM_WARNING = "LLM argues for"                                    # prefix of Diagnosis' disagreement warning
STATUS = {Route.ACT: ActionStatus.APPROVED, Route.ESCALATE: ActionStatus.APPROVED,   # escalating = notifying: safe
          Route.APPROVAL: ActionStatus.PENDING_APPROVAL, Route.NONE: ActionStatus.CLOSED}


def signature(inc: Incident | None, diag: Diagnosis) -> str:
    """What the incident is ABOUT, for the guards: the release to undo, else the CI project, else the signals."""
    if diag.suspect:
        return diag.suspect
    if inc is None:
        return diag.incident_id
    if inc.source == "ci":
        project = next((a.details.get("project") for a in inc.alerts if a.details.get("project")), None)
        if project:
            return f"ci:{project}"
    return f"{inc.source}:{inc.dataset}:{'+'.join(sorted(inc.features))}"


class _Verdict:
    """The decision being built: rules change it, and say why."""

    def __init__(self, action: Action, route: Route, rule: str):
        self.action, self.route, self.rule = action, route, rule
        self.reasons: list[str] = []

    def to(self, action: Action, route: Route, rule: str, reason: str) -> "_Verdict":
        self.action, self.route, self.rule = action, route, rule
        self.reasons.append(reason)
        return self


class DecisionAgent:
    def __init__(self, cfg: DecisionConfig | None = None, policy: dict[Category, PolicyRow] | None = None,
                 log: DecisionLog | None = None):
        """log: the decision log + history. With it, the guards remember (rerun budget, cooldown, circuit
        breaker) and run() records every decision. Without it, every decision stands alone (tests, demos)."""
        self.cfg = cfg or DecisionConfig()
        self.policy = policy or policy_mod.load()
        self.log = log

    # ------------------------------------------------------------------ one diagnosis
    def decide(self, diag: Diagnosis, incident: Incident | None = None, now: datetime | None = None,
               open_orders: dict[tuple[str, Action], str] | None = None,
               extra_grants: list[dict] | None = None) -> ActionDecision:
        """open_orders: (key, action) -> id of orders still waiting or not yet executed (duplicates are closed).
        extra_grants: history rows granted earlier in the same batch (the guards must see them)."""
        now = now or utcnow()
        row = self.policy[diag.category]
        key = signature(incident, diag)
        v = self._chain(diag, incident, row, diag.confidence, row.bar, now)
        self._guards(v, diag, row, key, now, open_orders or {}, extra_grants or [])
        self._valves(v, row, key, now, extra_grants or [])
        return self._build(diag, incident, row, v, key, now)

    def _chain(self, diag, inc, row: PolicyRow, conf: float, bar: float | None, now: datetime) -> _Verdict:
        cfg = self.cfg
        cause = diag.category.value
        proposed = row.action
        v = _Verdict(proposed, Route.ACT, f"table:{cause}")

        # A. a recognised no-fault: nothing to fix, nobody to page. Only when Diagnosis is sure.
        if diag.suggested_action == "none":
            if (diag.category is Category.UNKNOWN and diag.sub_cause in NO_FAULT and not diag.needs_human
                    and conf >= cfg.no_fault_min_confidence):
                return v.to(Action.NONE, Route.NONE, "no_fault",
                            f"Diagnosis recognised '{diag.sub_cause}' with confidence {conf:.2f} "
                            f">= {cfg.no_fault_min_confidence:.2f}: nothing to do, nobody is paged (logged)")
            return v.to(Action.ESCALATE, Route.ESCALATE, "no_fault:unsure",
                        "Diagnosis says 'no action', but it is not a confident recognised no-fault "
                        f"(sub-cause '{diag.sub_cause or '-'}', confidence {conf:.2f}): a human looks, we never silence a fault")

        if diag.suggested_action != proposed.value:
            v.reasons.append(f"Diagnosis suggested '{diag.suggested_action}'; the table says '{proposed.value}' "
                             f"for {cause}: the table wins")

        # B. the floor
        if conf < cfg.escalate_below:
            return v.to(Action.ESCALATE, Route.ESCALATE, f"floor:{cfg.escalate_below:.2f}",
                        f"confidence {conf:.2f} < {cfg.escalate_below:.2f}: no action is proposed, a human takes over")

        # C. causes that are never handled alone
        if row.never_alone:
            return v.to(Action.ESCALATE, Route.ESCALATE, f"table:{cause}:never_alone",
                        f"the table never handles {cause} alone (risk {row.risk}): a human takes over"
                        + (f" ({row.detail})" if row.detail else ""))

        # D. can the action even be done?
        problem = self._precondition(diag, inc, proposed)
        if problem:
            return v.to(Action.ESCALATE, Route.ESCALATE, "precondition", problem)

        # E. a human first?
        why = []
        if diag.needs_human:
            why.append("Diagnosis itself says a human must look (needs_human)")
        llm = [w for w in diag.warnings if w.startswith(LLM_WARNING)]
        if llm:
            why.append(f"the LLM disagrees with the classifier ({llm[0]}): both causes are shown to the reviewer")
        if bar is not None and conf < bar:
            why.append(f"confidence {conf:.2f} is under the table's bar {bar:.2f} for {proposed.value}")
        if why:
            first = ("human:needs_human" if diag.needs_human else "human:llm_disagrees" if llm else "table:below_bar")
            v.route, v.rule = Route.APPROVAL, first
            v.reasons += why
            return v

        # F. the risk gate: confident enough, so what does the action risk?
        if proposed is Action.ROLLBACK and not cfg.auto_rollback:
            v.route, v.rule = Route.APPROVAL, "gate:rollback"
            v.reasons.append(f"rollback needs a human (team rule, deck scenario 2) although {conf:.2f} >= {bar:.2f} "
                             "would be enough by the table; `auto_rollback` switches this off")
        elif proposed is Action.RETRAIN and row.daytime_only and not self._daytime(now):
            v.route, v.rule = Route.APPROVAL, "gate:night"
            v.reasons.append(f"retraining is day time only ({cfg.day_start_hour:02d}:00-{cfg.day_end_hour:02d}:00 UTC), "
                             f"it is {now:%H:%M}: it waits for a human")
        else:
            v.reasons.append(f"{proposed.value} is {row.risk}-risk{', reversible' if row.reversible else ''} and "
                             f"confidence {conf:.2f} >= {bar:.2f}: safe to do alone")
            if proposed is Action.RETRAIN and cfg.retrain_human_before_deploy:
                v.reasons.append("the new model is never deployed alone: a human approves the deploy")
        return v

    @staticmethod
    def _precondition(diag: Diagnosis, inc: Incident | None, proposed: Action) -> str:
        if proposed is Action.ROLLBACK and not diag.suspect:
            return "rollback needs a release or config to undo, and Diagnosis named none: a human takes over"
        if inc is not None:
            if proposed is Action.RERUN and inc.source != "ci":
                return f"rerun only makes sense for a CI job, this incident is '{inc.source}': a human takes over"
            if proposed is Action.RETRAIN and inc.source != "model":
                return f"retrain only makes sense for model inputs, this incident is '{inc.source}': a human takes over"
        return ""

    def _daytime(self, now: datetime) -> bool:
        return self.cfg.day_start_hour <= now.hour < self.cfg.day_end_hour

    # ------------------------------------------------------------------ G. guards that remember
    def _guards(self, v: _Verdict, diag, row: PolicyRow, key: str, now, open_orders, extra) -> None:
        cfg = self.cfg
        if v.action is Action.NONE:
            return
        grants = (self.log.grants(cfg.history_window_h, now) if self.log else []) + extra
        same = [g for g in grants if g["key"] == key]
        if v.action is Action.RERUN:
            used = sum(g["action"] == "rerun" for g in same)
            if used >= cfg.rerun_budget:
                v.to(Action.ESCALATE, Route.ESCALATE, "guard:rerun_budget",
                     f"a re-run was already granted on {key} ({used} of {cfg.rerun_budget} allowed): it failed again, "
                     "so it is a real bug, not a flaky test: a human takes over")
                return
        pending = open_orders.get((key, v.action))
        recent = [g for g in same if g["action"] == v.action.value
                  and (now - datetime.fromisoformat(g["at"])).total_seconds() / 60 < cfg.cooldown_min]
        if pending or recent:
            ref = pending or recent[-1]["decision_id"]
            v.to(Action.NONE, Route.NONE, "guard:duplicate",
                 f"{v.action.value} on {key} is already {'open' if pending else 'done'} ({ref}"
                 f"{'' if pending else f', within the {cfg.cooldown_min:.0f} min cooldown'}): not repeated")

    # ------------------------------------------------------------------ H. safety valves
    def _valves(self, v: _Verdict, row: PolicyRow, key: str, now, extra) -> None:
        cfg = self.cfg
        if v.route is Route.NONE and not cfg.enabled and v.rule == "no_fault":
            v.to(Action.ESCALATE, Route.ESCALATE, "valve:disabled", "the agent is switched off: even no-fault cases go to a human")
            return
        if v.route is not Route.ACT or v.action not in EXECUTABLE:
            return
        if not cfg.enabled:
            v.to(v.action, Route.APPROVAL, "valve:disabled", "the agent is switched off (kill switch): nothing runs alone")
        elif cfg.dry_run:
            v.to(v.action, Route.APPROVAL, "valve:dry_run", "dry run: decisions are made and logged, nothing runs alone")
        else:
            grants = (self.log.grants(1.0, now) if self.log else []) + extra
            auto = [g for g in grants if g["auto"] and g["action"] in {a.value for a in EXECUTABLE}
                    and (now - datetime.fromisoformat(g["at"])).total_seconds() < 3600]
            if len(auto) >= cfg.max_auto_per_hour:
                v.to(v.action, Route.APPROVAL, "valve:circuit_breaker",
                     f"{len(auto)} automatic actions in the last hour (limit {cfg.max_auto_per_hour}): a human checks this one")

    # ------------------------------------------------------------------ build the order for P5
    def _build(self, diag: Diagnosis, inc: Incident | None, row: PolicyRow, v: _Verdict, key: str, now) -> ActionDecision:
        cfg = self.cfg
        proposed = v.action is row.action
        constraints = {"max_retries": cfg.max_retries, "verify_after_min": cfg.verify_after_min,
                       "revert_if_worse": cfg.revert_if_worse and v.action in EXECUTABLE,
                       "human_before_deploy": v.action is Action.RETRAIN and cfg.retrain_human_before_deploy}
        if v.action is Action.RERUN:
            constraints["reruns"] = cfg.rerun_budget
        if v.action is Action.ESCALATE and row.detail:
            constraints["notify"] = row.detail                   # e.g. "author" for code_regression
        cause = diag.category.value + (f" ({diag.sub_cause})" if diag.sub_cause else "")
        what = {Action.NONE: "nothing to do", Action.ESCALATE: "a human takes over"}.get(
            v.action, f"{v.action.value}" + (f" {diag.suspect}" if v.action is Action.ROLLBACK and diag.suspect else ""))
        summary = (f"{cause}, confidence {diag.confidence:.2f} -> {what}. "
                   + (f"{inc.summary} " if inc is not None else "") + diag.explanation).strip()
        snapshot = {
            "action_key": key, "bar": row.bar, "suggested_action": diag.suggested_action, "needs_human": diag.needs_human,
            "sub_cause": diag.sub_cause, "suspect": diag.suspect, "method": diag.method,
            "explanation": diag.explanation, "warnings": list(diag.warnings), "runbooks": list(diag.runbooks),
            "incident_summary": inc.summary if inc is not None else "",
            "evidence": [{"id": e.id, "statement": e.statement, "source": e.source} for e in diag.evidence],
            "hypotheses": [{"category": h.category.value, "probability": h.probability, "sub_cause": h.sub_cause}
                           for h in diag.hypotheses]}
        return ActionDecision(
            incident_id=diag.incident_id, diagnosis_id=diag.id, category=diag.category, action=v.action,
            route=v.route, status=STATUS[v.route], confidence=diag.confidence,
            risk=row.risk if proposed else "none", target=diag.suspect if v.action is Action.ROLLBACK else "",
            rule=v.rule, reasons=v.reasons,
            summary=summary, snapshot=snapshot, constraints=constraints, created=now)

    # ------------------------------------------------------------------ a batch
    def run(self, diagnoses: list[Diagnosis], incidents: list[Incident] | None = None, now: datetime | None = None,
            open_orders: dict[tuple[str, Action], str] | None = None) -> list[ActionDecision]:
        """Decide every diagnosis. Earlier decisions of the batch count for the later ones (rerun budget,
        duplicates, circuit breaker). With a log, every decision is recorded and granted actions are remembered."""
        now = now or utcnow()
        by_id = {i.id: i for i in incidents or []}
        open_orders = dict(open_orders or {})
        extra: list[dict] = []
        out: list[ActionDecision] = []
        # most confident first: when two diagnoses compete for the same target, the best-supported one wins
        for diag in sorted(diagnoses, key=lambda d: -d.confidence):
            d = self.decide(diag, by_id.get(diag.incident_id), now, open_orders, extra)
            out.append(d)
            if d.route is not Route.NONE:
                open_orders[(action_key(d), d.action)] = d.id
            if self.log:                      # the history file already holds earlier grants of this batch
                self.log.event("decided", d, now)
                if d.status is ActionStatus.APPROVED:
                    self.log.grant(d, now)
            elif d.status is ActionStatus.APPROVED:     # no log: keep them in memory for the guards
                extra.append({"at": now.isoformat(timespec="seconds"), "decision_id": d.id, "incident_id": d.incident_id,
                              "key": action_key(d), "action": d.action.value, "auto": True})
        order = {d.incident_id: n for n, d in enumerate(diagnoses)}
        return sorted(out, key=lambda d: order[d.incident_id])
