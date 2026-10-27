"""Train the root-cause classifier on the TRAIN split of the benchmark.

    python -m agents.diagnosis.train             # builds the benchmark the first time (~1-2 min)
    python -m agents.diagnosis.train --rebuild   # rebuild it (after changing scenarios.py)
"""
from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np

from .classifier import RootCauseClassifier
from .config import DiagnosisConfig
from .evidence import collect
from .knowledge import KnowledgeBase
from .scenarios import Case, load_or_build


def featurize(cases: list[Case], cfg: DiagnosisConfig, kb: KnowledgeBase) -> np.ndarray:
    return np.array([collect(c.incident, c.ctx, cfg, kb).vector() for c in cases], float)


def train(cases: list[Case], cfg: DiagnosisConfig | None = None, kb: KnowledgeBase | None = None,
          folds: int = 3) -> RootCauseClassifier:
    """Classifier on the whole train split + a confidence calibration fitted OUT OF FOLD:
    each train case is diagnosed by an agent whose classifier never saw it."""
    from .agent import DiagnosisAgent

    cfg, kb = cfg or DiagnosisConfig(), kb or KnowledgeBase(embed_model=None)
    tr = [c for c in cases if c.split == "train"]
    X, y = featurize(tr, cfg, kb), [c.truth for c in tr]
    fold = np.random.default_rng(0).permutation(len(tr)) % folds
    raw, correct = [], []
    for k in range(folds):
        clf_k = RootCauseClassifier().fit(X[fold != k], [t for t, f in zip(y, fold) if f != k])
        agent = DiagnosisAgent(cfg, kb, clf_k)
        for c in (c for c, f in zip(tr, fold) if f == k):
            _, hyps, _ = agent.rank(c.incident, c.ctx)
            raw.append(hyps[0].probability)
            correct.append(hyps[0].category == c.truth)
    return RootCauseClassifier().fit(X, y).fit_calibration(raw, correct)


def main():
    t0 = time.time()
    print("benchmark:")
    cases = load_or_build(rebuild="--rebuild" in sys.argv)
    tr = [c for c in cases if c.split == "train"]
    print(f"  {len(cases)} incidents, {len(tr)} for training | "
          f"causes: {dict(Counter(c.truth.value for c in tr).most_common())}")
    clf = train(cases)
    clf.save()
    print(f"classifier saved ({len(clf.classes)} causes) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
