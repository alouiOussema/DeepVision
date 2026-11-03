"""Measure the Decision policy on the Diagnosis benchmark (TEST split), against two simpler policies.

    python -m agents.decision.evaluate [--rebuild]

Needs the Diagnosis benchmark and classifier (python -m agents.diagnosis.train); builds them if missing.
Every case has a known true cause (and the true release to undo), so we can count REAL mistakes:

    wrong_auto_actions   an action P5 would run alone (rerun / rollback / retrain) on the wrong cause or the wrong release
    silenced_faults      "nothing to do" decided alone on a real fault
    act_alone_pct        share of incidents P5 handles with no human at all  (the autonomy we gain)
    human_load_pct       share that reaches a person (approval page + escalation)
    human_on_fixable     share of FIXABLE incidents (flaky / bad deploy / config / drift) that still reached a person
    right_auto_pct       share where P5 acts alone AND is right

Policies
    suggested   act on whatever Diagnosis suggests unless it says needs_human     (no Decision agent)
    deck        slide 11 as written: safe and >= 0.8 -> act, risky -> human, < 0.5 -> escalate
    ours        the table + the team rules (human gate on rollback)
    ours+auto   the same with auto_rollback = True: what the human gate costs and what it buys

The stateful guards (rerun budget, cooldown, circuit breaker) need a history and are not simulated here.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from common.schemas import Action, Category, Diagnosis

from . import paths
from .agent import EXECUTABLE, DecisionAgent
from .config import DecisionConfig

FIXABLE = {Category.FLAKY_TEST, Category.BAD_DEPLOY, Category.CONFIG_CHANGE, Category.DATA_DRIFT}
NOON = datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ the policies: Diagnosis -> (action, alone?)
def suggested_policy(d: Diagnosis, inc) -> tuple[Action, str]:
    action = Action(d.suggested_action) if d.suggested_action in {a.value for a in Action} else Action.ESCALATE
    if d.needs_human:
        return action, "human"
    return action, "alone" if action in EXECUTABLE or action is Action.NONE else "escalate"


def deck_policy(d: Diagnosis, inc) -> tuple[Action, str]:
    """Slide 11: safe & >= 0.8 -> act; risky -> human; < 0.5 -> escalate."""
    action = Action(d.suggested_action) if d.suggested_action in {a.value for a in Action} else Action.ESCALATE
    if d.confidence < 0.5 or action in (Action.ESCALATE, Action.NONE):
        return Action.ESCALATE, "escalate"
    if action is Action.ROLLBACK:
        return action, "human"
    return action, "alone" if d.confidence >= 0.8 and not d.needs_human else "human"


def ours(cfg: DecisionConfig):
    agent = DecisionAgent(cfg)

    def policy(d: Diagnosis, inc) -> tuple[Action, str]:
        a = agent.decide(d, inc, NOON)
        return a.action, {"act": "alone", "approval": "human", "escalate": "escalate", "none": "alone"}[a.route.value]
    return policy


# ------------------------------------------------------------------ scoring (pure: no model needed)
def score(rows: list[dict]) -> dict:
    """rows: one per incident: truth, truth_suspect, suspect, action (Action), how ('alone'|'human'|'escalate')."""
    n = len(rows)
    alone = [r for r in rows if r["how"] == "alone"]
    acting = [r for r in alone if r["action"] in EXECUTABLE]

    def right(r) -> bool:
        if r["action"] is Action.NONE:
            return r["truth"] is Category.UNKNOWN
        wanted = {Action.RERUN: Category.FLAKY_TEST, Action.RETRAIN: Category.DATA_DRIFT}
        if r["action"] is Action.ROLLBACK:
            return r["category"] is r["truth"] and r["suspect"] == r["truth_suspect"]
        return r["category"] is r["truth"] and wanted.get(r["action"]) is r["truth"]

    fixable = [r for r in rows if r["truth"] in FIXABLE]
    pct = lambda k, t: round(100 * k / t, 1) if t else None
    return {
        "n": n,
        "act_alone_pct": pct(len(acting), n),
        "human_load_pct": pct(sum(r["how"] != "alone" for r in rows), n),
        "right_auto_pct": pct(sum(right(r) for r in acting), n),
        "wrong_auto_actions": sum(not right(r) for r in acting),
        "silenced_faults": sum(r["action"] is Action.NONE and r["how"] == "alone" and r["truth"] is not Category.UNKNOWN
                               for r in rows),
        "human_on_fixable": pct(sum(r["how"] != "alone" for r in fixable), len(fixable)),
    }


def evaluate(cases, diags, policies: dict) -> dict:
    out = {}
    for name, pol in policies.items():
        rows = []
        for c, d in zip(cases, diags):
            action, how = pol(d, c.incident)
            rows.append({"truth": c.truth, "truth_suspect": c.truth_suspect, "category": d.category, "suspect": d.suspect,
                         "action": action, "how": how})
        out[name] = score(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild the Diagnosis benchmark")
    args = ap.parse_args()

    from agents.diagnosis.agent import DiagnosisAgent
    from agents.diagnosis.classifier import RootCauseClassifier
    from agents.diagnosis.config import DiagnosisConfig
    from agents.diagnosis.knowledge import KnowledgeBase
    from agents.diagnosis.scenarios import load_or_build
    from agents.diagnosis.train import train

    cases = load_or_build(args.rebuild)
    test = [c for c in cases if c.split == "test"]
    kb, cfg = KnowledgeBase(embed_model=None), DiagnosisConfig()
    clf = RootCauseClassifier.load()
    if not clf.trained:
        print("training the Diagnosis classifier (about 2 min)...")
        clf = train(cases, cfg, kb)
        clf.save()
    agent = DiagnosisAgent(cfg, kb, clf)
    diags = [agent.diagnose(c.incident, c.ctx) for c in test]
    policies = {"suggested": suggested_policy, "deck": deck_policy, "ours": ours(DecisionConfig()),
                "ours+auto_rollback": ours(DecisionConfig(auto_rollback=True))}
    results = {"all test incidents": evaluate(test, diags, policies)}
    sent = [k for k, c in enumerate(test) if c.incident.decision.value == "incident"]
    results["sent by Triage"] = evaluate([test[k] for k in sent], [diags[k] for k in sent], policies)

    cols = ["n", "act_alone_pct", "right_auto_pct", "wrong_auto_actions", "silenced_faults", "human_load_pct", "human_on_fixable"]
    for title, res in results.items():
        print(f"\n{title}\n{'policy':20s}" + "".join(f"{c:>20s}" for c in cols))
        for name, r in res.items():
            print(f"{name:20s}" + "".join(f"{str(r[c]):>20s}" for c in cols))
    paths.EVAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    paths.EVAL_RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {paths.EVAL_RESULTS}")


if __name__ == "__main__":
    main()
