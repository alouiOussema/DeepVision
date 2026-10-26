"""Measure Diagnosis on the TEST split of the benchmark, against a baseline and against its own parts.

    python -m agents.diagnosis.evaluate                                   # all methods, no LLM
    python -m agents.diagnosis.evaluate --llm llama3.1:8b,mistral:instruct --n-llm 24   # + compare 2 LLMs

Methods
    baseline   a fixed table "source -> most frequent cause" (what a pipeline without agents does)
    rules      evidence + rule scorer
    ml         evidence + classifier only
    hybrid     rules and classifier blended (the agent as deployed)
    +llm       hybrid + LLM explanation with its fact-check (on a sample: a local LLM is slow)

Metrics
    accuracy, macro_f1      is the named cause right? (macro: every cause counts the same)
    sub_cause_acc           ... and the finer cause (leak vs latency vs errors), when the cause is right
    suspect_acc             bad deploy / config: did we blame the right release? (P5 rolls back THAT one)
    auto_rate               share P3 lets P4 act on without a human (needs_human = False)
    auto_accuracy           accuracy on those: the number that matters for autonomy
    wrong_auto_actions      rollback / rerun / retrain suggested without a human AND wrong cause or release
    ece                     calibration: |confidence - accuracy| averaged over 10 bins (0 = perfect)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from common.schemas import Category, Diagnosis

from . import ollama_http, paths
from .agent import DiagnosisAgent
from .config import DiagnosisConfig
from .knowledge import KnowledgeBase
from .llm import LLMExplainer
from .scenarios import Case, load_or_build
from .train import train

ACTING = {"rollback", "rerun", "retrain"}


# ------------------------------------------------------------------ baseline
class SourceBaseline:
    """No agent: each source always gets the cause it most often has (learned on the train split)."""
    def __init__(self, cases: list[Case], cfg: DiagnosisConfig):
        by = defaultdict(Counter)
        for c in cases:
            by[c.incident.source][c.truth] += 1
        self.table = {s: cnt.most_common(1)[0] for s, cnt in by.items()}
        self.share = {s: n / sum(by[s].values()) for s, (_, n) in self.table.items()}
        self.cfg = cfg

    def diagnose(self, inc, ctx=None) -> Diagnosis:
        cat = self.table.get(inc.source, (Category.UNKNOWN, 0))[0]
        return Diagnosis(inc.id, cat, "fixed rule", [], round(self.share.get(inc.source, 0.0), 3),
                         suggested_action=self.cfg.actions[cat], method="baseline")


# ------------------------------------------------------------------ scoring
def score(cases: list[Case], diags: list[Diagnosis]) -> dict:
    y = [c.truth.value for c in cases]
    p = [d.category.value for d in diags]
    ok = np.array([a == b for a, b in zip(y, p)])
    conf = np.array([d.confidence for d in diags])
    auto = np.array([not d.needs_human for d in diags])

    sub = [(c, d) for c, d in zip(cases, diags) if c.truth_sub and c.truth_sub != "no_fault" and c.truth == d.category]
    sus = [(c, d) for c, d in zip(cases, diags) if c.truth_suspect]

    def wrong_action(c: Case, d: Diagnosis) -> bool:
        if d.needs_human or d.suggested_action not in ACTING:
            return False
        wrong_target = d.suggested_action == "rollback" and d.suspect != c.truth_suspect
        return c.truth != d.category or wrong_target

    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(conf[bins == b].mean() - ok[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    fam = defaultdict(list)
    for c, good in zip(cases, ok):
        fam[c.family].append(good)
    return {
        "n": len(cases),
        "accuracy": round(float(ok.mean()), 3),
        "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 3),
        "sub_cause_acc": round(float(np.mean([c.truth_sub == d.sub_cause for c, d in sub])), 3) if sub else None,
        "suspect_acc": round(float(np.mean([c.truth_suspect == d.suspect for c, d in sus])), 3) if sus else None,
        "auto_rate": round(float(auto.mean()), 3),
        "auto_accuracy": round(float(ok[auto].mean()), 3) if auto.any() else None,
        "wrong_auto_actions": int(sum(wrong_action(c, d) for c, d in zip(cases, diags))),
        "ece": round(float(ece), 3),
        "per_family": {k: round(float(np.mean(v)), 3) for k, v in sorted(fam.items())},
    }


def run(method, cases: list[Case]) -> tuple[list[Diagnosis], float]:
    t0 = time.perf_counter()
    diags = [method.diagnose(c.incident, c.ctx) for c in cases]
    return diags, (time.perf_counter() - t0) / max(len(cases), 1)


def sample(cases: list[Case], n: int, seed: int = 0) -> list[Case]:
    """n cases spread over the true causes (a plain random sample would be mostly deploy)."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for c in cases:
        by[c.truth].append(c)
    out, k = [], 0
    while len(out) < min(n, len(cases)):
        group = list(by.values())[k % len(by)]
        if group:
            out.append(group.pop(int(rng.integers(len(group)))))
        k += 1
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild the benchmark")
    ap.add_argument("--llm", default="", help="comma-separated Ollama models to compare, e.g. llama3.1:8b,mistral:instruct")
    ap.add_argument("--n-llm", type=int, default=24, help="test cases explained by each LLM")
    args = ap.parse_args()

    cases = load_or_build(args.rebuild)
    test = [c for c in cases if c.split == "test"]
    cfg, kb = DiagnosisConfig(), KnowledgeBase(embed_model=None)
    clf = train(cases, cfg, kb)
    clf.save()
    print(f"benchmark: {len(cases)} incidents, test split {len(test)} "
          f"({dict(Counter(c.family for c in test))}) | causes {dict(Counter(c.truth.value for c in test))}\n")

    methods = {
        "baseline": SourceBaseline([c for c in cases if c.split == "train"], cfg),
        "rules": DiagnosisAgent(cfg, kb),
        "ml": DiagnosisAgent(DiagnosisConfig(ml_weight=1.0), kb, clf),
        "hybrid": DiagnosisAgent(cfg, kb, clf),
    }
    results, diags = {}, {}
    for name, m in methods.items():
        diags[name], secs = run(m, test)
        results[name] = {**score(test, diags[name]), "ms_per_incident": round(1000 * secs, 1)}

    if args.llm:
        sub = sample(test, args.n_llm)
        installed = ollama_http.models()
        kb_rag = KnowledgeBase()                  # with embeddings for the LLM's retrieval
        for model in [m.strip() for m in args.llm.split(",") if m.strip()]:
            if model not in installed and f"{model}:latest" not in installed:
                print(f"skip {model}: not installed in Ollama ({installed or 'Ollama not running'})")
                continue
            agent = DiagnosisAgent(cfg, kb_rag, clf, LLMExplainer(model))
            d, secs = run(agent, sub)
            warn = [w for x in d for w in x.warnings]
            results[f"hybrid+llm({model})"] = {
                **score(sub, d), "ms_per_incident": round(1000 * secs, 1),
                "llm_answer_passed_factcheck": round(np.mean([not any(w.startswith("LLM") and "argues" not in w
                                                                      for w in x.warnings) for x in d]), 3),
                "llm_invented_ids": sum("unknown id" in w for w in warn),
                "llm_invented_numbers": sum("number(s) found nowhere" in w for w in warn),
                "llm_disagreed": sum("LLM argues" in w for w in warn),
            }
            results[f"hybrid (same {len(sub)} cases)"] = score(sub, [methods["hybrid"].diagnose(c.incident, c.ctx)
                                                                     for c in sub])

    show(results)
    labels = sorted({c.truth.value for c in test} | {d.category.value for d in diags["hybrid"]})
    cm = confusion_matrix([c.truth.value for c in test], [d.category.value for d in diags["hybrid"]], labels=labels)
    print("\nconfusion matrix of hybrid (rows = truth, columns = diagnosis):")
    print(" " * 16 + "".join(f"{lab[:9]:>10s}" for lab in labels))
    for lab, row in zip(labels, cm):
        print(f"{lab:16s}" + "".join(f"{v:10d}" for v in row))
    paths.RESULTS.parent.mkdir(parents=True, exist_ok=True)
    paths.RESULTS.write_text(json.dumps({"results": results, "labels": labels, "confusion": cm.tolist()}, indent=2))
    print(f"\nsaved {paths.RESULTS}")


def show(results: dict) -> None:
    cols = ["n", "accuracy", "macro_f1", "sub_cause_acc", "suspect_acc", "auto_rate", "auto_accuracy",
            "wrong_auto_actions", "ece", "ms_per_incident"]
    print(f"{'method':32s}" + "".join(f"{c[:13]:>14s}" for c in cols))
    for name, r in results.items():
        print(f"{name:32s}" + "".join(f"{'-' if r.get(c) is None else r.get(c)!s:>14s}" for c in cols))
    print("\naccuracy per family:")
    for name, r in results.items():
        print(f"  {name:30s} {r['per_family']}")
    extra = {n: {k: v for k, v in r.items() if k.startswith("llm_")} for n, r in results.items() if "llm(" in n}
    for n, e in extra.items():
        print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
