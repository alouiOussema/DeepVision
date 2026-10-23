"""Layer 2 of Triage: an LLM agent that investigates the cases the rules are unsure about.

    rules (agent.py) ──> INCIDENT / NOISE              -> kept as they are
                    └──> WATCH, and HDFS patterns      -> LLMTriage.review()
                                                          loop: the model calls tools,
                                                          reads the results, then decides

The agent loop (one incident):
  1. send the system prompt + "investigate INC-xxxx"
  2. the model answers with tool calls -> our Python runs them -> results go back
  3. repeat until the model answers with an EVIDENCE CHECKLIST in JSON (or max_steps is reached)
  4. decide_from_evidence() turns the checklist into a decision, always with the same rules

Why the model fills a checklist instead of deciding: tested on qwen2.5:7b, the model read the
facts correctly but mis-applied the ordered rules (2 cases out of 5). So the work is split:
the LLM investigates and explains, the code applies the policy.

Why the checklist is fact-checked: on the full evaluation (129 reviews) the model INVENTED
human feedback in 18% of cases ("false_alarm" when memory was empty) and silenced real HDFS
failures. Every field that can be checked against the tool outputs now is (facts_checklist):
the tools win, and each disagreement is logged as a "misread".

Guardrails (written in code, not left to the model):
  - if the model fails or answers nonsense, the rule decision is kept
  - the model may only DOWNGRADE an incident (stop paging a human) when human feedback in
    memory says "false_alarm": a hunch is not enough to silence a page
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import ollama

from common.schemas import Decision, Incident

from .memory import Memory, key_for
from .tools import facts_checklist, make_tools

SYSTEM = """You are the Triage agent of an MLOps monitoring pipeline.
Rules already grouped alerts into incidents. You review ONE incident the rules were unsure about
and decide if a human must be paged.

Investigate with the tools. Always call get_incident and check_memory first, then the other
tools if they can help.

Each tool returns a "fact" sentence. Base your decision ONLY on these facts. Never invent
numbers or comparisons that no tool gave you.

You do NOT take the decision. You fill in an evidence checklist from the facts you read.
When you are done, answer ONLY with this JSON, no other text:
{"feedback": "false_alarm" | "real" | "none",
 "bad_deploy_sign": true | false,      (true only if a fact says errors AND latency rise together)
 "features_together": <number of different abnormal features, in the incident or nearby>,
 "strength": "stronger" | "average" | "weaker" | "unknown",
 "isolated": true | false,
 "reason": "<one sentence summarising the evidence>"}"""


def _norm(field: str, v):
    """Compare like with like: features_together only matters as 'one' vs 'several'."""
    if field == "features_together":
        try:
            return int(v or 1) >= 2
        except (TypeError, ValueError):
            return None
    return v


def decide_from_evidence(ev: dict) -> tuple[str, str]:
    """The policy, in code: same checklist -> same decision, every time (first match wins)."""
    fb = ev.get("feedback", "none")
    n = int(ev.get("features_together") or 1)
    if fb == "false_alarm":
        return "noise", "human feedback: false alarm"
    if fb == "real":
        return "incident", "human feedback: real problem"
    if ev.get("bad_deploy_sign"):
        return "incident", "errors and latency rising together on deploy"
    if n >= 2:
        return "incident", f"{n} features abnormal together"
    if ev.get("strength") == "stronger":
        return "incident", "stronger than most similar incidents"
    if n == 1 and ev.get("isolated") and ev.get("strength") == "weaker":
        return "noise", "1 feature, isolated, weaker than most"
    return "watch", "evidence points both ways"


@dataclass
class Review:
    incident_id: str
    rule_decision: str
    decision: str
    reason: str
    tool_calls: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)   # the checklist filled by the model
    misread: list[str] = field(default_factory=list)   # fields where the model contradicted the tools
    facts: dict = field(default_factory=dict)          # the fact-checked values (from the tools)
    steps: list[dict] = field(default_factory=list)    # transcript: each tool call and what it returned
    overridden: bool = False      # True if a guardrail replaced the model's answer
    error: str = ""


class LLMTriage:
    def __init__(self, memory: Memory, model: str = "qwen2.5:7b", max_steps: int = 6, client=ollama):
        self.memory = memory
        self.model = model
        self.max_steps = max_steps
        self.client = client          # replaceable by a fake in tests

    # ------------------------------------------------------------------ which incidents
    @staticmethod
    def needs_review(i: Incident) -> bool:
        """WATCH on time series, and every HDFS pattern (where feedback can catch false alarms).
        CI watch cases (700+) are left to the rules: too many for a laptop model."""
        if i.source == "logs":
            return i.decision in (Decision.INCIDENT, Decision.WATCH)
        return i.decision == Decision.WATCH and i.source in ("model", "metrics") and i.kind == "anomaly"

    # ------------------------------------------------------------------ one incident
    def review(self, inc: Incident, tools: dict) -> Review:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Investigate incident {inc.id} and decide."}]
        calls: list[str] = []
        steps: list[dict] = []
        try:
            for _ in range(self.max_steps):
                r = self.client.chat(model=self.model, messages=messages, tools=list(tools.values()),
                                     options={"temperature": 0})
                messages.append(r.message)
                if not r.message.tool_calls:                         # the model has decided
                    return replace(self._final(inc, r.message.content, calls, tools), steps=steps)
                for c in r.message.tool_calls:                       # the model wants evidence
                    name, args = c.function.name, dict(c.function.arguments or {})
                    calls.append(f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})")
                    try:
                        out = tools[name](**args)
                    except Exception as e:                           # bad tool name / bad id
                        out = json.dumps({"error": str(e)})
                    messages.append({"role": "tool", "content": out, "tool_name": name})
                    steps.append({"tool": name, "args": args, "output": out})
            return replace(self._keep_rule(inc, calls, "no decision after max_steps"), steps=steps)
        except Exception as e:                                       # Ollama down, timeout, ...
            return replace(self._keep_rule(inc, calls, f"{type(e).__name__}: {e}"), steps=steps)

    def _final(self, inc: Incident, text: str, calls: list[str], tools: dict) -> Review:
        try:
            ev = json.loads(text[text.find("{"): text.rfind("}") + 1])
            assert isinstance(ev, dict)
        except Exception:
            return self._keep_rule(inc, calls, f"unreadable answer: {text[:120]!r}")
        # fact-check: every checkable field comes from the tools, not from the model
        facts = facts_checklist(inc, tools)
        misread = [k for k, v in facts.items() if _norm(k, ev.get(k)) != _norm(k, v)]
        decision, rule_used = decide_from_evidence({**ev, **facts})
        note = f" | model misread {', '.join(misread)} (tools win)" if misread else ""
        reason = f"{rule_used}{note} | model: {ev.get('reason', '')}"

        rule = inc.decision.value
        # guardrail: silencing a page needs human feedback, not a hunch
        fb = self.memory.lookup(key_for(inc))
        if rule == "incident" and decision != "incident" and not (fb and fb["verdict"] == "false_alarm"):
            return Review(inc.id, rule, rule, f"kept: evidence said {decision} without human feedback ({reason})",
                          calls, ev, misread, facts, overridden=True)
        return Review(inc.id, rule, decision, reason, calls, ev, misread, facts)

    @staticmethod
    def _keep_rule(inc: Incident, calls: list[str], error: str) -> Review:
        return Review(inc.id, inc.decision.value, inc.decision.value, "rule decision kept", calls, error=error)

    # ------------------------------------------------------------------ a whole run
    def run(self, incidents: list[Incident], limit: int | None = None, verbose: bool = False,
            context: list[Incident] | None = None, on_review=None):
        """Review the unsure incidents. Returns (updated incidents, reviews).
        context: older incidents the tools may compare with (history); on_review(k, total, review)
        is called after each review, so a UI can show progress live."""
        tools = make_tools(incidents + (context or []), self.memory)
        todo = [i for i in incidents if self.needs_review(i)][:limit]
        reviews = {}
        for k, inc in enumerate(todo, 1):
            rv = self.review(inc, tools)
            reviews[inc.id] = rv
            if on_review:
                on_review(k, len(todo), inc, rv)
            if verbose:
                flag = "" if rv.decision == rv.rule_decision else "  <- changed"
                print(f"[{k}/{len(todo)}] {inc.id} {rv.rule_decision} -> {rv.decision}{flag}  "
                      f"| {len(rv.tool_calls)} tool call(s) | {rv.error or rv.reason}"[:200])
        out = []
        for i in incidents:
            rv = reviews.get(i.id)
            if rv is None or rv.decision == i.decision.value:
                out.append(i)
            else:
                out.append(replace(i, decision=Decision(rv.decision),
                                   reasons=i.reasons + [f"LLM review: {rv.decision}: {rv.reason}",
                                                        f"tools used: {', '.join(rv.tool_calls)}"]))
        return out, list(reviews.values())