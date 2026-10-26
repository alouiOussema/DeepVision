"""Step 4: the LLM writes the explanation. The code checks every word it can.

Same split as Triage's layer 2 (agents/triage/llm_agent.py), for the same reason: a 7-8B local
model reads facts well but invents some. So:

    the CODE  ranks the causes (scorer + classifier) and gathers the evidence
    the LLM   writes 2-4 sentences a human can act on, citing evidence [E2] and runbooks [RB-...],
              and says if it disagrees with the ranking (and why)
    the CODE  verifies the answer:
                - every citation must exist         (an invented [E9] -> answer rejected)
                - every number must come from the evidence or the runbooks (a made-up "35 %" -> rejected)
                - at least one evidence citation    (an opinion without facts -> rejected)
              a rejected answer is replaced by the template explanation: never a silent failure.
    disagreement never changes the cause: it LOWERS the confidence (agent.py), so P4 asks a human.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from common.schemas import Category, Evidence, Hypothesis, Incident

from . import ollama_http

SYSTEM = """You are the Diagnosis agent of an MLOps pipeline. An incident was detected and the code
already ranked the possible root causes from the evidence. Your job: explain the most likely cause
to the on-call engineer, in 2 to 4 short sentences.

Rules:
- Use ONLY the evidence and runbook excerpts given. Cite them with their ids in brackets, e.g. [E2] [RB-bad-deploy].
- Never write a number that is not in the evidence. Never invent ids.
- Say what to check or do next, taken from the runbook.
- If the evidence clearly points to ANOTHER cause than the top one, set "agree" to false and name it.

Answer ONLY with this JSON:
{"explanation": "<2-4 sentences with citations>",
 "agree": true | false,
 "alternative": "<one of: %s> or null",
 "alternative_reason": "<one sentence, or empty>",
 "next_checks": ["<short action>", "..."]}""" % ", ".join(c.value for c in Category)

CITE = re.compile(r"\[([^\[\]]+)\]")
NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")


@dataclass
class LLMResult:
    ok: bool
    explanation: str = ""
    agree: bool = True
    alternative: Category | None = None
    alternative_reason: str = ""
    next_checks: list[str] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model: str = ""
    seconds: float = 0.0
    raw: str = ""


def prompt(inc: Incident, hyps: list[Hypothesis], evidence: list[Evidence], excerpts: list[tuple[str, str]]) -> str:
    h = "\n".join(f"{k}. {x.category.value}" + (f" ({x.sub_cause})" if x.sub_cause else "")
                  + f": probability {x.probability:.2f}" for k, x in enumerate(hyps[:3], 1))
    e = "\n".join(f"[{x.id}] {x.statement}" for x in evidence)
    r = "\n".join(f"[{ref}] {text[:700]}" for ref, text in excerpts)
    return (f"INCIDENT {inc.id} ({inc.source}, {inc.dataset}, severity {inc.severity.value})\n\n"
            f"RANKED CAUSES (from the code)\n{h}\n\nEVIDENCE\n{e}\n\nRUNBOOK EXCERPTS\n{r}")


def _numbers(text: str) -> set[float]:
    return {float(n) for n in NUMBER.findall(text)}


def verify(answer: dict, evidence: list[Evidence], excerpts: list[tuple[str, str]], context: str) -> LLMResult:
    """Fact-check one parsed answer. ok=False means: do not show this text."""
    text = str(answer.get("explanation", "")).strip()
    ev_ids = {e.id for e in evidence}
    rb_ids = {ref for ref, _ in excerpts}
    cited = [c.strip() for group in CITE.findall(text) for c in group.split(",")]
    bad = [c for c in cited if c not in ev_ids | rb_ids]
    warnings = [f"LLM cited unknown id(s): {', '.join(bad)}"] if bad else []
    if not any(c in ev_ids for c in cited):
        warnings.append("LLM cited no evidence")

    allowed = _numbers(context) | _numbers(" ".join(t for _, t in excerpts))
    allowed |= {round(x * 100) for x in list(allowed) if x <= 1}          # 0.85 may be written 85 %
    claimed = _numbers(CITE.sub(" ", text))
    invented = sorted(n for n in claimed if n not in allowed and n > 10)  # small counts ("2 metrics") are fine
    if invented:
        warnings.append(f"LLM wrote number(s) found nowhere in the evidence: {', '.join(f'{n:g}' for n in invented)}")

    alt = answer.get("alternative")
    alt = Category(alt) if alt in {c.value for c in Category} else None
    agree = bool(answer.get("agree", True)) or alt is None
    return LLMResult(ok=not warnings and bool(text), explanation=text, agree=agree, alternative=None if agree else alt,
                     alternative_reason=str(answer.get("alternative_reason", "") or ""),
                     next_checks=[str(x) for x in answer.get("next_checks", []) or []][:4],
                     cited=sorted(set(cited) - set(bad)), warnings=warnings)


class LLMExplainer:
    def __init__(self, model: str = "llama3.1:8b", chat=ollama_http.chat):
        self.model = model
        self.chat = chat              # replaceable by a fake in tests

    def explain(self, inc: Incident, hyps: list[Hypothesis], evidence: list[Evidence],
                excerpts: list[tuple[str, str]]) -> LLMResult:
        user = prompt(inc, hyps, evidence, excerpts)
        t0 = time.perf_counter()
        try:
            raw = self.chat(self.model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        except Exception as e:                                   # Ollama down, timeout, unknown model
            return LLMResult(False, warnings=[f"LLM unavailable: {type(e).__name__}: {e}"[:200]], model=self.model)
        secs = round(time.perf_counter() - t0, 2)
        try:
            answer = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            assert isinstance(answer, dict)
        except Exception:
            return LLMResult(False, warnings=[f"LLM answer is not JSON: {raw[:120]!r}"], model=self.model,
                             seconds=secs, raw=raw)
        res = verify(answer, evidence, excerpts, user)
        res.model, res.seconds, res.raw = self.model, secs, raw
        return res


def template_explanation(hyp: Hypothesis, confidence: float, evidence: list[Evidence],
                         contributions: list[tuple[str, float]], runbook_id: str | None) -> str:
    """The explanation without an LLM: the strongest evidence for the chosen cause, in words."""
    by_id = {e.id: e for e in evidence}
    top = [i for i, w in sorted(contributions, key=lambda x: -x[1]) if w > 0][:3]
    name = hyp.category.value.replace("_", " ") + (f" ({hyp.sub_cause.replace('_', ' ')})" if hyp.sub_cause else "")
    lines = [f"Most likely cause: {name}, confidence {confidence:.2f}."]
    lines += [f"{by_id[i].statement} [{i}]" for i in top if i in by_id]
    if not top:
        lines.append("No evidence points clearly to one cause: a human should look.")
    if runbook_id:
        lines.append(f"Follow [{runbook_id}].")
    return " ".join(lines)
