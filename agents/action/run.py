"""Run the Action agent on its own: execute everything the action inbox says is approved (simulation).

    python -m agents.action.run            # the real inbox (data/decision/action_inbox.json)
    python -m agents.action.run --demo     # the Decision demo's inbox (python -m agents.decision.demo + approve in the UI)

Typical use: a human approved a rollback on ui/decision_app.py after the pipeline ran; this picks it up.
Real execution is not available from here: this repo has no real CI / deploy / model backend (see README).
"""
from __future__ import annotations

import argparse

from agents.decision import demo as decision_demo
from agents.decision.inbox import ActionInbox
from common.schemas import Action

from . import kpis, paths
from .agent import ActionAgent
from .backends import Backends, SimDeploy


def run_ready(inbox: ActionInbox, orders: list | None = None, **kw) -> list:
    """Run the approved orders, each rollback against its OWN simulated service.

    There is no change log here, so each release to roll back is assumed live and faulty (printed). One world per
    order: the approved incidents are separate stories; a single shared service would add every release's damage
    together, and rolling back one of six "faulty" releases would then always look like a failure.
    """
    orders = inbox.ready() if orders is None else orders
    runs = []
    for o in orders:
        if o.action is Action.ROLLBACK and o.target:
            print(f"  simulation assumption: {o.target} is live and faulty (so the rollback can fix it)")
            backends = Backends(deploy=SimDeploy([o.target], {o.target: (0.05, 60.0)}))
        else:
            backends = None
        runs += ActionAgent(backends=backends, inbox=inbox, **kw).run([o])
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use the Decision demo's inbox (data/decision/demo/)")
    args = ap.parse_args()
    inbox = decision_demo.demo_inbox() if args.demo else ActionInbox()
    kw = dict(claims_dir=paths.DEMO_CLAIMS, notifications=paths.DEMO_NOTIFICATIONS,
              deploy_requests=paths.DEMO_DEPLOY_REQUESTS) if args.demo else {}
    ready = inbox.ready()
    print(f"inbox {inbox.path}: {len(ready)} approved order(s) to run [simulation]")
    runs = run_ready(inbox, ready, **kw)
    for r in runs:
        print(f"  {r.incident_id}  {r.action:9s} -> {r.outcome:9s} attempts {r.attempts}  told: "
              f"{', '.join(r.notified) or '-'}\n      {r.detail}")
    print(kpis.report(kpis.compute(inbox.all(), inbox.log.events())))
    print(f"notifications: {kw.get('notifications', paths.NOTIFICATIONS)}")


if __name__ == "__main__":
    main()
