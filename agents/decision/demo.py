"""The deck's four scenarios (+3 more), end to end through the Decision agent.

    python -m agents.decision.demo [--night] [--auto-rollback]
    python -m streamlit run ui/decision_app.py          # then tick "Demo data" and approve / reject

Uses its own files (data/decision/demo/), reset at every run, so it never touches real orders.
"""
from __future__ import annotations

import argparse
import textwrap
from datetime import datetime, timedelta, timezone

from common.schemas import ActionDecision

from . import paths
from .agent import DecisionAgent
from .config import DecisionConfig
from .inbox import ActionInbox
from .log import DecisionLog
from .scenarios import T0, deck_scenarios

BAR = "-" * 100
ICON = {"act": "ACT alone     ", "approval": "ASK A HUMAN   ", "escalate": "ESCALATE      ", "none": "NOTHING TO DO "}


def demo_log() -> DecisionLog:
    return DecisionLog(paths.DEMO_LOG, paths.DEMO_HISTORY)


def demo_inbox() -> ActionInbox:
    return ActionInbox(paths.DEMO_INBOX, demo_log())


def reset() -> None:
    for p in (paths.DEMO_INBOX, paths.DEMO_LOG, paths.DEMO_HISTORY):
        p.unlink(missing_ok=True)


def card(name: str, d: ActionDecision) -> str:
    what = d.action.value + (f" {d.target}" if d.target else "")
    return "\n".join([BAR, f"{name}", f"  {ICON[d.route.value]} -> {what:<22} confidence {d.confidence:.2f}  risk {d.risk or '-'}  "
                      f"[{d.rule}]  status: {d.status.value}",
                      *[textwrap.fill(r, 96, initial_indent="    - ", subsequent_indent="      ") for r in d.reasons]])


def run_demo(night: bool = False, auto_rollback: bool = False, now: datetime | None = None) -> list[ActionDecision]:
    """Fill the demo inbox with the scenarios' decisions; returns them (scenario order)."""
    reset()
    now = now or (T0.replace(hour=2) if night else T0)
    agent = DecisionAgent(DecisionConfig(auto_rollback=auto_rollback), log=demo_log())
    cases = deck_scenarios()
    # scenario 6 comes after scenario 1 by 20 minutes: that is what the re-run budget needs to see
    out: list[ActionDecision] = []
    for k, (name, inc, diag) in enumerate(cases):
        t = now + timedelta(minutes=20 * k)
        out.append(agent.run([diag], [inc], t)[0])
    demo_inbox().add(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", action="store_true", help="read the clock as 02:00 (retraining is day time only)")
    ap.add_argument("--auto-rollback", action="store_true", help="follow the table's 0.80 rule instead of the human gate")
    args = ap.parse_args()
    decisions = run_demo(args.night, args.auto_rollback)
    print(f"Decision agent - the deck's scenarios (clock {'02:00' if args.night else '10:00'} UTC, "
          f"auto_rollback={args.auto_rollback})")
    for (name, _, _), d in zip(deck_scenarios(), decisions):
        print(card(name, d))
    inbox = demo_inbox()
    print(f"{BAR}\n{len(decisions)} decisions | P5 may execute now: {len(inbox.ready())} | "
          f"waiting for a human: {len(inbox.pending())}")
    print(f"decision log: {paths.DEMO_LOG}\napproval page: python -m streamlit run ui/decision_app.py   (tick 'Demo data')")


if __name__ == "__main__":
    main()
