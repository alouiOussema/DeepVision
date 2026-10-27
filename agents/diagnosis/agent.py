"""The Diagnosis agent (P3): Incident -> Diagnosis, in 5 steps.

    1. collect    evidence from the incident and around it          (evidence/)
                  change log, raw metrics, log templates, other incidents, past incidents
    2. rank       every possible cause                              (scorer.py rules + classifier.py ML)
    3. retrieve   the runbook sections that match the top cause     (knowledge.py, hybrid RAG)
    4. explain    2-4 sentences with cited evidence                 (llm.py, fact-checked; template if no LLM)
    5. judge      confidence, needs_human, suggested action          (here, in code)

Confidence policy (written in code, not left to the model):
  - confidence = probability of the top cause after blending rules and classifier,
                 calibrated on held-out cases (classifier.py): 0.8 means right about 80 % of the time
  - x novelty_penalty      if the case looks like nothing in the knowledge base
  - x llm_disagree_penalty if the LLM argues for another cause (it can never change the cause itself)
  - top probability < min_probability  -> category UNKNOWN (we do not guess)
  - confidence < human_below           -> needs_human = True (P4 should not act alone)
  - UNKNOWN always needs a human, EXCEPT a recognised no-fault (recovery after a rollback, a known
    false alarm confirmed by an operator) with enough confidence: action "none", nobody paged
"""
from __future__ import annotations

from common.schemas import Category, Diagnosis, Hypothesis, Incident

from . import scorer
from .classifier import RootCauseClassifier
from .config import DiagnosisConfig
from .context import ContextStore
from .evidence import Findings, collect
from .knowledge import KnowledgeBase
from .llm import LLMExplainer, LLMResult, template_explanation

SUSPECT_KIND = {Category.BAD_DEPLOY: "deploy", Category.CONFIG_CHANGE: "config"}   # cause -> change to undo
# "unknown" sub-causes that are RECOGNISED as no fault: nothing to fix, nobody to page (if confident)
NO_FAULT = {"recovery", "known_false_alarm"}


def blend(rules: dict[Category, float], ml: dict[Category, float] | None, w: float) -> dict[Category, float]:
    if not ml:
        return dict(rules)
    cats = set(rules) | set(ml)
    out = {c: (1 - w) * rules.get(c, 0.0) + w * ml.get(c, 0.0) for c in cats}
    z = sum(out.values()) or 1.0
    return {c: v / z for c, v in out.items()}


class DiagnosisAgent:
    def __init__(self, cfg: DiagnosisConfig | None = None, kb: KnowledgeBase | None = None,
                 classifier: RootCauseClassifier | None = None, llm: LLMExplainer | None = None):
        self.cfg = cfg or DiagnosisConfig()
        self.kb = kb
        self.classifier = classifier if classifier is not None and classifier.trained else None
        self.llm = llm

    @property
    def method(self) -> str:
        parts = ["rules"] + (["ml"] if self.classifier else []) + (["rag"] if self.kb else [])
        return "+".join(parts) + (f"+llm({self.llm.model})" if self.llm else "")

    # ------------------------------------------------------------------ steps 1-2 (no LLM: fast)
    def rank(self, inc: Incident, ctx: ContextStore) -> tuple[Findings, list[Hypothesis], dict]:
        f = collect(inc, ctx, self.cfg, self.kb)
        rules, why = scorer.score(f.evidence, inc.source, self.cfg)
        ml = self.classifier.predict_proba(f.vector()) if self.classifier else None
        probs = blend(rules, ml, self.cfg.ml_weight)
        hyps = [Hypothesis(c, round(p, 4), f.sub_causes.get(c, "")) for c, p in
                sorted(probs.items(), key=lambda x: -x[1]) if p >= 0.005]
        return f, hyps, why

    # ------------------------------------------------------------------ all steps
    def diagnose(self, inc: Incident, ctx: ContextStore | None = None) -> Diagnosis:
        ctx = ctx or ContextStore()
        f, hyps, why = self.rank(inc, ctx)
        best = hyps[0]
        warnings: list[str] = []

        category = best.category if best.probability >= self.cfg.min_probability else Category.UNKNOWN
        if category is Category.UNKNOWN and best.category is not Category.UNKNOWN:
            warnings.append(f"best cause {best.category.value} only at {best.probability:.2f}: not enough to name it")
        # confidence = probability of what we SAY: the top cause, or "unknown" when we abstain,
        # mapped to the accuracy observed for that probability on held-out cases (when calibrated)
        confidence = best.probability if category is best.category else \
            next((h.probability for h in hyps if h.category is Category.UNKNOWN), 0.0)
        if self.classifier is not None and category is best.category:
            confidence = self.classifier.calibrate(confidence)
        if f.novel:
            confidence *= self.cfg.novelty_penalty
            warnings.append("pattern never seen in the knowledge base: confidence lowered")

        # step 3: runbooks for the chosen cause (and the runner-up when it is close)
        excerpts, runbooks = self._retrieve(inc, f, category, hyps)

        # step 4: explanation
        shown = Hypothesis(category, best.probability, best.sub_cause if category is best.category else "")
        explanation = template_explanation(shown, confidence, f.evidence, why.get(category, []),
                                           runbooks[0] if runbooks else None)
        if self.llm is not None:
            res = self.llm.explain(inc, hyps, f.evidence, excerpts)
            warnings += res.warnings
            if res.ok:
                explanation = res.explanation + self._next(res)
            if res.ok and not res.agree and res.alternative not in (None, category):
                confidence *= self.cfg.llm_disagree_penalty
                warnings.append(f"LLM argues for {res.alternative.value}: {res.alternative_reason} "
                                "(cause kept, confidence lowered)")

        # step 5: what P4 receives
        confidence = round(min(max(confidence, 0.0), 1.0), 3)
        no_fault = category is Category.UNKNOWN and shown.sub_cause in NO_FAULT
        needs_human = (category is Category.UNKNOWN and not no_fault) or confidence < self.cfg.human_below
        action = "escalate" if needs_human else "none" if no_fault else self.cfg.actions[category]
        kind = SUSPECT_KIND.get(category)
        suspect = f.suspect.id if f.suspect is not None and f.suspect.kind == kind else ""
        return Diagnosis(
            incident_id=inc.id, category=category, sub_cause=shown.sub_cause, suspect=suspect,
            explanation=explanation, evidence=f.evidence, confidence=confidence, hypotheses=hyps[:4],
            suggested_action=action, needs_human=needs_human, runbooks=runbooks, method=self.method,
            warnings=warnings)

    def run(self, incidents: list[Incident], ctx: ContextStore | None = None) -> list[Diagnosis]:
        """A whole batch: every incident sees the others as peers (several CI projects red at once...)."""
        ctx = (ctx or ContextStore()).with_peers(list((ctx.peers if ctx else [])) + list(incidents))
        return [self.diagnose(i, ctx) for i in incidents]

    # ------------------------------------------------------------------ helpers
    def _retrieve(self, inc: Incident, f: Findings, category: Category, hyps: list[Hypothesis]):
        if self.kb is None:
            return [], []
        query = " ".join([category.value.replace("_", " "), hyps[0].sub_cause.replace("_", " ")]
                         + [e.statement for e in f.evidence[1:4]])
        hits = self.kb.search(query, k=3, boost=category, source=inc.source)
        close = len(hyps) > 1 and hyps[1].probability > 0.25 and hyps[1].category is not category
        runner_up = self.kb.runbook_for(hyps[1].category, inc.source) if close else None
        if runner_up is not None:      # let the explanation say why it is NOT the runner-up
            hits += [(c, 0.0) for c in self.kb.index.chunks if c.doc is runner_up][:1]
        own = self.kb.runbook_for(category, inc.source)
        refs = ([own.id] if own else []) + [c.ref for c, _ in hits]
        excerpts = [(c.ref, f"{c.doc.title} / {c.section}: {c.text}") for c, _ in hits]
        return excerpts, list(dict.fromkeys(refs))

    @staticmethod
    def _next(res: LLMResult) -> str:
        return (" Next: " + "; ".join(res.next_checks) + ".") if res.next_checks else ""
