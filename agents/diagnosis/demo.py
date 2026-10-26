"""Scenario 2 (bad deployment), end to end: detect -> triage -> DIAGNOSE -> approval inbox.

    python -m agents.diagnosis.demo                       # rules + classifier + RAG, template explanations
    python -m agents.diagnosis.demo --llm llama3.1:8b     # + LLM explanations (Ollama)
    python -m streamlit run ui/approval_app.py            # then approve / correct on the approval page

The chain is the production one: P1's deploy simulator (a held-out seed, never used for training)
-> P1's rolling detector -> P2's Triage rules -> only the incidents Triage sends to P3.
"""
from __future__ import annotations

import argparse
import textwrap
import time

from common.schemas import Decision

from .agent import DiagnosisAgent
from .classifier import RootCauseClassifier
from .feedback import save_inbox
from .knowledge import KnowledgeBase
from .llm import LLMExplainer
from .scenarios import deploy_cases

BAR = "-" * 96


def card(c, d) -> str:
    truth = c.truth.value + (f"/{c.truth_sub}" if c.truth_sub else "") + (f" [{c.truth_suspect}]" if c.truth_suspect else "")
    ok = "OK " if d.category == c.truth else "MISS"
    hyp = "  ".join(f"{h.category.value} {h.probability:.2f}" for h in d.hypotheses[:3])
    return "\n".join([
        BAR,
        f"{c.incident.id}  {c.incident.summary[:80]}",
        f"  diagnosis : {d.category.value}" + (f" ({d.sub_cause})" if d.sub_cause else "")
        + (f"  suspect={d.suspect}" if d.suspect else "")
        + f"  confidence={d.confidence:.2f}  -> {d.suggested_action}{'  [HUMAN]' if d.needs_human else ''}",
        f"  truth     : {truth}   {ok}",
        f"  ranking   : {hyp}",
        *[f"  [{e.id}] {e.statement}" for e in d.evidence[1:]],
        textwrap.fill(d.explanation, 94, initial_indent="  > ", subsequent_indent="    "),
        *[f"  ! {w}" for w in d.warnings],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4, help="simulation seed (4 and 5 are held out)")
    ap.add_argument("--llm", default="", help="Ollama model for the explanations, e.g. llama3.1:8b")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Scenario 2 - bad deployment | simulation seed {args.seed} -> P1 Monitor -> P2 Triage ...")
    cases = deploy_cases(args.seed, "test")
    sent = [c for c in cases if c.incident.decision == Decision.INCIDENT]
    print(f"  {len(cases)} incidents from Triage, {len(sent)} sent to Diagnosis ({time.time() - t0:.0f} s)\n")

    agent = DiagnosisAgent(kb=KnowledgeBase(embed_model="nomic-embed-text" if args.llm else None),
                           classifier=RootCauseClassifier.load(),
                           llm=LLMExplainer(args.llm) if args.llm else None)
    print(f"Diagnosis agent: {agent.method}")
    ctx = sent[0].ctx if sent else None
    diags = agent.run([c.incident for c in sent], ctx)
    for c, d in zip(sent, diags):
        print(card(c, d))
    right = sum(d.category == c.truth for c, d in zip(sent, diags))
    auto = [d for d in diags if not d.needs_human]
    print(f"{BAR}\n{right}/{len(sent)} right causes | {len(auto)} could be handled without a human, "
          f"{len(diags) - len(auto)} go to the approval page")
    save_inbox([(c.incident, d) for c, d in zip(sent, diags)])
    print("inbox saved: python -m streamlit run ui/approval_app.py")


if __name__ == "__main__":
    main()
