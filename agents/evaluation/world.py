"""Simulated CI, deployments and model registry built from the truth.

Only the release we broke does damage, so rolling back the wrong one fails the check. One world per order.
"""
from __future__ import annotations

from agents.action.backends import Backends, SimCI, SimDeploy, SimModel
from common.schemas import Category, Incident

from .config import EvalConfig
from .injector import Injection


def backends_for(inj: Injection | None, incident: Incident | None, context, cfg: EvalConfig | None = None) -> Backends:
    """inj None = the incident matches no injected fault: a healthy service, a passing CI, no drift."""
    cfg = cfg or EvalConfig()
    truth = inj.truth if inj else Category.UNKNOWN

    ci = SimCI(default="passed" if truth is Category.FLAKY_TEST else "failed")

    t = incident.start if incident else None
    live = [c.id for c in getattr(context, "changes", []) if t is None or c.timestamp <= t]
    effects, ambient = {}, (0.0, 0.0)
    if inj and truth in (Category.BAD_DEPLOY, Category.CONFIG_CHANGE) and inj.truth_suspect:
        effects[inj.truth_suspect] = cfg.harm
        if inj.truth_suspect not in live:
            live.append(inj.truth_suspect)
    if truth is Category.INFRA:
        ambient = cfg.infra_ambient
    deploy = SimDeploy(live, effects, ambient=ambient)

    challenger = cfg.better_challenger_mae if truth is Category.DATA_DRIFT else cfg.worse_challenger_mae
    model = SimModel(cfg.champion_mae, challenger)
    return Backends(ci=ci, deploy=deploy, model=model)
