"""Markdown incident report, written from an incident's timeline."""
from __future__ import annotations

from .score import Record
from .trace import TraceEvent


def _t(e: TraceEvent) -> str:
    return e.at.strftime("%Y-%m-%d %H:%M") if e.at else "-"


def _cell(s: str, n: int = 160) -> str:
    s = " ".join(str(s).split()).replace("|", "/")
    return s if len(s) <= n else s[: n - 1] + "…"


def verdict(r: Record) -> str:
    if not r.sent:
        return "missed: a real fault Triage kept as " + r.triage if r.is_fault else f"filtered as {r.triage} (right)"
    if r.silenced:
        return "WRONG: a real fault was closed as 'nothing to do'"
    if r.wrong_fix:
        return "WRONG: " + ("the wrong fix ran" if not r.action_right else f"the right fix {r.outcome}")
    if r.fixed:
        return "fixed and verified" + (" with no human" if r.fixed_alone else " (a human approved it)")
    if r.action == "escalate":
        return "handed to a person" + (" (right: nothing could be fixed alone)" if r.action_right else "")
    if r.action == "none":
        return "closed, nobody paged" + (" (right)" if r.action_right else "")
    return r.status or "-"


def incident_report(r: Record, events: list[TraceEvent]) -> str:
    first = events[0] if events else None
    diag = next((e for e in events if e.agent == "diagnosis"), None)
    act = next((e for e in events if e.agent == "action"), None)
    lines = [f"# Incident {r.incident_id} ({r.episode})", "",
             f"**Outcome:** {verdict(r)}", "",
             "| | |", "|---|---|",
             f"| Detected | {_t(first) if first else '-'} ({r.n_alerts} alert(s)) |",
             f"| Triage | {r.triage}{' — sent to Diagnosis' if r.sent else ''} |"]
    if r.category:
        lines.append(f"| Cause (Diagnosis) | {r.category}{' / ' + r.sub_cause if r.sub_cause else ''}"
                     f"{', release ' + r.suspect if r.suspect else ''}, confidence {r.confidence:.2f} |")
    if r.action:
        lines.append(f"| Decision | {r.route}: {r.action}{' ' + r.target if r.target else ''} [{r.rule}] |")
    if r.outcome:
        lines.append(f"| Action | {r.outcome}{'; told ' + ', '.join(r.notified) if r.notified else ''} |")
    if r.injection:
        lines.append(f"| Truth (injected) | {r.truth}{' / ' + r.truth_sub if r.truth_sub else ''}"
                     f"{', release ' + r.truth_suspect if r.truth_suspect else ''} ({r.injection}) |")
    lines += ["", "## Timeline", "", "| Time | Agent | Did | Result |", "|---|---|---|---|"]
    lines += [f"| {_t(e)} | {e.agent} | {_cell(e.action)} | {_cell(e.result)} |" for e in events]
    if diag:
        lines += ["", "## Why (Diagnosis)", "", _cell(diag.reason, 1200)]
    dec = next((e for e in events if e.agent == "decision"), None)
    if dec:
        lines += ["", "## Decision trail", ""] + [f"- {_cell(x, 400)}" for x in dec.reason.split(" -> ")]
    if act:
        lines += ["", "## What the Action agent did", ""] + [f"- {_cell(x, 400)}" for x in act.reason.split(" | ")]
    return "\n".join(lines) + "\n"
