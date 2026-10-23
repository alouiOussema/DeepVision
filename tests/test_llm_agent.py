"""Tests for the LLM layer WITHOUT Ollama: a fake model plays a scripted investigation."""
import json

from ollama import ChatResponse, Message

from agents.triage.agent import RuleTriage
from agents.triage.llm_agent import LLMTriage, decide_from_evidence
from agents.triage.memory import Memory
from agents.triage.tools import make_tools
from common.schemas import Alert, Decision


class FakeModel:
    """Turn 1: calls get_incident + check_memory. Turn 2: returns the given checklist."""
    def __init__(self, evidence):
        self.evidence = evidence

    def chat(self, model, messages, tools=None, options=None):
        inc = messages[1]["content"].split()[2]
        if not any(isinstance(m, dict) and m.get("role") == "tool" for m in messages):
            call = lambda n: Message.ToolCall(function=Message.ToolCall.Function(name=n, arguments={"incident_id": inc}))
            return ChatResponse(model=model, message=Message(role="assistant",
                                                             tool_calls=[call("get_incident"), call("check_memory")]))
        return ChatResponse(model=model, message=Message(role="assistant", content=json.dumps(self.evidence)))


def blocks(n, events):
    return [Alert.from_dict(dict(detector="block_pca", source="logs", feature="block_events", score=5.0,
                                 threshold=1.0, timestamp="2026-10-20T11:00:00+00:00", value=5.0, dataset="hdfs",
                                 details={"block_id": f"blk_{k}", "top_events": [{"event": e} for e in events]}))
            for k in range(n)]


NOISY = {"feedback": "none", "bad_deploy_sign": False, "features_together": 1,
         "strength": "weaker", "isolated": True, "reason": "weak and alone"}


def test_policy_order():
    assert decide_from_evidence({**NOISY, "feedback": "false_alarm", "features_together": 3})[0] == "noise"
    assert decide_from_evidence({**NOISY, "bad_deploy_sign": True})[0] == "incident"
    assert decide_from_evidence({**NOISY, "features_together": 2})[0] == "incident"
    assert decide_from_evidence(NOISY)[0] == "noise"
    assert decide_from_evidence({**NOISY, "strength": "average"})[0] == "watch"


def test_memory_false_alarm_silences_a_page(tmp_path):
    inc = RuleTriage().process(blocks(12, ["E2", "E3", "E4"]))[0]
    assert inc.decision == Decision.INCIDENT
    mem = Memory(tmp_path / "m.json")
    mem.add_feedback("logs:E2+E3+E4", "false_alarm")
    rv = LLMTriage(mem, client=FakeModel({**NOISY, "feedback": "false_alarm"})).review(inc, make_tools([inc], mem))
    assert rv.decision == "noise" and rv.tool_calls[1].startswith("check_memory")


def test_guardrail_keeps_page_without_feedback(tmp_path):
    inc = RuleTriage().process(blocks(12, ["E2", "E3", "E4"]))[0]
    mem = Memory(tmp_path / "m.json")
    rv = LLMTriage(mem, client=FakeModel(NOISY)).review(inc, make_tools([inc], mem))
    assert rv.decision == "incident" and rv.overridden


def test_unreadable_answer_keeps_rule_decision(tmp_path):
    inc = RuleTriage().process(blocks(12, ["E2", "E3", "E4"]))[0]
    mem = Memory(tmp_path / "m.json")
    rv = LLMTriage(mem, client=FakeModel("I think it is fine")).review(inc, make_tools([inc], mem))
    assert rv.decision == "incident" and rv.error


def test_invented_feedback_is_corrected_by_the_fact_check(tmp_path):
    """The model claims a human said false_alarm, but memory is empty: the tools win."""
    inc = RuleTriage().process(blocks(12, ["E2", "E3", "E4"]))[0]
    mem = Memory(tmp_path / "m.json")
    rv = LLMTriage(mem, client=FakeModel({**NOISY, "feedback": "false_alarm"})).review(inc, make_tools([inc], mem))
    assert rv.decision == "incident" and "feedback" in rv.misread