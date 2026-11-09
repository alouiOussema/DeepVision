"""The deck's four scenarios end to end, success AND failure branches: Diagnosis -> Decision (P4) -> [human] -> Action (P5).

    python -m agents.action.demo                  # simulation, virtual clock (15 min of waiting take 0 s)
    python -m agents.action.demo --real-retrain   # scenario 3 retrains with scikit-learn on data/hourly/nyc.parquet

Uses its own files (data/action/demo/), reset at every run, so it never touches real orders.
"""
from __future__ import annotations

import argparse
import shutil
import textwrap
from datetime import timedelta

from agents.decision.agent import DecisionAgent
from agents.decision.inbox import ActionInbox
from agents.decision.log import DecisionLog
from common.schemas import ActionStatus

from . import kpis, paths
from .agent import ActionAgent, ActionRun
from .backends import FileNotifier
from .clock import SimClock
from .config import ActionConfig
from .executors import job_refs_from_incidents
from .scenarios import T0, Case, deck_cases

BAR = "-" * 100
OUTCOME = {"executed": "DONE + VERIFIED", "failed": "FAILED (reported)", "reverted": "UNDONE (reported)",
           "skipped": "NOT RUN"}


def demo_log() -> DecisionLog:
    return DecisionLog(paths.DEMO_LOG, paths.DEMO_HISTORY)


def demo_inbox() -> ActionInbox:
    return ActionInbox(paths.DEMO_INBOX, demo_log())


def reset() -> None:
    for p in (paths.DEMO_INBOX, paths.DEMO_LOG, paths.DEMO_HISTORY, paths.DEMO_NOTIFICATIONS, paths.DEMO_DEPLOY_REQUESTS):
        p.unlink(missing_ok=True)
    shutil.rmtree(paths.DEMO_CLAIMS, ignore_errors=True)


def play(case: Case, start, real_retrain: bool = False) -> tuple:
    """One case: Decision decides, a human resolves if asked, the Action agent runs whatever is approved."""
    inbox = demo_inbox()
    order = DecisionAgent(log=inbox.log).run([case.diagnosis], [case.incident], start)[0]
    inbox.add([order])
    human = ""
    if order.status is ActionStatus.PENDING_APPROVAL and case.approver:
        verdict = "reject" if case.reject else "approve"
        order = inbox.resolve(order.id, verdict, case.approver, "read the explanation", now=start + timedelta(minutes=4))
        human = f"{case.approver} {'rejected' if case.reject else 'approved'} it after 4 min"
    world = case.world()
    if real_retrain and order.action.value == "retrain":
        from .retrain_sklearn import SklearnNYCModel
        world.model = SklearnNYCModel(out_dir=paths.DEMO_DIR / "challengers")
    world.notifier = FileNotifier(paths.DEMO_NOTIFICATIONS)
    agent = ActionAgent(ActionConfig(), world, inbox, SimClock(start + timedelta(minutes=5)),
                        claims_dir=paths.DEMO_CLAIMS, deploy_requests=paths.DEMO_DEPLOY_REQUESTS)
    run = agent.run([order], job_refs_from_incidents([case.incident]))[0]
    return order, human, run


def card(case: Case, order, human: str, run: ActionRun) -> str:
    what = order.action.value + (f" {order.target}" if order.target else "")
    lines = [BAR, case.name, f"  expected : {case.story}",
             f"  Decision : {order.route.value} -> {what}  [{order.rule}]" + (f"   human: {human}" if human else ""),
             f"  Action   : {OUTCOME[run.outcome]}" + (f"  attempts {run.attempts}" if run.attempts else "")
             + (f"  MTTR {run.mttr_min:g} min" if run.mttr_min is not None else "")
             + (f"  told: {', '.join(run.notified)}" if run.notified else "")]
    lines += [textwrap.fill(s, 96, initial_indent="    - ", subsequent_indent="      ") for s in run.steps]
    ok = run.outcome == case.expected
    lines.append(f"  {'OK' if ok else 'UNEXPECTED'}: outcome {run.outcome} (expected {case.expected})")
    return "\n".join(lines)


def run_demo(real_retrain: bool = False) -> list[tuple]:
    reset()
    out = []
    for k, case in enumerate(deck_cases()):
        out.append((case, *play(case, T0 + timedelta(minutes=30 * k), real_retrain)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-retrain", action="store_true", help="scenario 3: scikit-learn on data/hourly/nyc.parquet")
    args = ap.parse_args()
    results = run_demo(args.real_retrain)
    print("Action agent (P5) - the deck's scenarios, success and failure branches  [SIMULATION, virtual clock]")
    for case, order, human, run in results:
        print(card(case, order, human, run))
    inbox = demo_inbox()
    print(BAR)
    print(kpis.report(kpis.compute(inbox.all(), inbox.log.events())))
    msgs = FileNotifier(paths.DEMO_NOTIFICATIONS).messages()
    print(f"{BAR}\n{len(msgs)} notifications: " + ", ".join(f"{m['to']} ({m.get('incident', m.get('order'))})" for m in msgs))
    bad = [c.name for c, _, _, r in results if r.outcome != c.expected]
    print(f"{len(results) - len(bad)}/{len(results)} scenarios behaved as the deck says" + (f"; UNEXPECTED: {bad}" if bad else ""))
    print(f"files: {paths.DEMO_DIR}")


if __name__ == "__main__":
    main()
