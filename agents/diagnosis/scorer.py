"""Step 2a: the RULE scorer. Evidence weights -> one probability per cause.

    logit(cause) = log(prior of the cause for this source) + sum of the evidence weights for it
    P(cause)     = softmax(logits)

Weights are log-odds: +1 makes a cause about e times more likely, -1 about e times less.
Works with zero training data and every point of probability can be traced to one Evidence id,
which is what the explanation and the approval page show ("why bad_deploy? E2 +1.8, E3 +0.8").
"""
from __future__ import annotations

import math

from common.schemas import Category

from .config import DiagnosisConfig

FLOOR = 0.02          # prior of a cause the source table does not list (still reachable by evidence)


def score(evidence, source: str, cfg: DiagnosisConfig) -> tuple[dict[Category, float], dict[Category, list]]:
    """Returns (probabilities, contributions): contributions[cause] = [(evidence id, weight), ...]."""
    prior = cfg.priors.get(source, {Category.UNKNOWN: 1.0})
    logits = {c: math.log(p) for c, p in prior.items()}
    why: dict[Category, list] = {c: [] for c in logits}
    for e in evidence:
        for c, w in e.supports.items():
            c = Category(c)
            logits[c] = logits.get(c, math.log(FLOOR)) + w
            why.setdefault(c, []).append((e.id, w))
    top = max(logits.values())
    z = sum(math.exp(v - top) for v in logits.values())
    return {c: math.exp(v - top) / z for c, v in logits.items()}, why
