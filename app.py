"""Triage playground: watch the pipeline run live, or dig into one scenario from the ideation phase.

    py -m pip install streamlit
    py -m streamlit run app.py

Everything runs the REAL code on P1's real alerts (data/alerts.csv): the rules, the tools,
the memory, and (if Ollama is running) the LLM agent with its fact-check.
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.triage import paths
from agents.triage.agent import RuleTriage
from agents.triage.config import TriageConfig
from agents.triage.io import load_alerts
from agents.triage.llm_agent import LLMTriage
from agents.triage.memory import Memory, key_for
from agents.triage.tools import make_tools
from common.schemas import Decision
from pipeline import WINDOWS, build_graph, make_layer2, replay_monitor

st.set_page_config(page_title="Triage Agent playground", layout="wide")
MEMORY_PATH = paths.APP_MEMORY

# ------------------------------------------------------------------ scenarios (ideation phase)
SCENARIOS = {
    "1 · Flaky test in CI": {
        "tag": "DEVOPS", "trigger": "A CI build turns red after a commit",
        "goal": "Nobody is woken up for a test that is not really broken",
        "expect": "Builds that look flaky become a LOW incident routed to an automatic re-run, not to a person.",
        "pick": lambda i: i.source == "ci" and i.decision.value == "incident" and i.severity.value == "low"},
    "1b · CI: a project that usually passes breaks": {
        "tag": "DEVOPS", "trigger": "Several real failures in one day on a usually green project",
        "goal": "Page a human only when the failure is news",
        "expect": "3+ real failures on a project that usually passes become a HIGH incident.",
        "pick": lambda i: i.source == "ci" and i.severity.value == "high"},
    "2 · Bad deployment": {
        "tag": "DEVOPS", "trigger": "Errors and latency rise right after a deployment",
        "goal": "Recover fast without an unsafe automatic rollback",
        "expect": "The rules leave it in WATCH (weak signal); the agent sees errors AND latency together and pages.",
        "pick": lambda i: i.dataset == "deploy" and len(i.features) == 2 and i.decision.value == "watch"},
    "3 · Data drift on the ML model": {
        "tag": "MLOPS", "trigger": "The distribution of model inputs slowly changes (PSI)",
        "goal": "Catch drift before customers see bad predictions, without paging at night",
        "expect": "Drift always goes to WATCH: it is a retraining question for P4, not an emergency.",
        "pick": lambda i: i.kind == "drift"},
    "4 · Unknown failure (log pattern never seen)": {
        "tag": "SAFETY", "trigger": "A rare HDFS log pattern with no human feedback",
        "goal": "Never silence what the system cannot justify",
        "expect": "Rare but strong pattern -> INCIDENT. Without feedback, the agent is not allowed to silence it.",
        "pick": lambda i: i.source == "logs" and len(i.alert_ids) <= 2 and i.peak_ratio >= 2},
    "4b · Known false alarm (operator feedback in memory)": {
        "tag": "SAFETY", "trigger": "4,235 HDFS blocks with pattern E2+E3+E4",
        "goal": "Learn from the operator: stop paging for a known false alarm",
        "expect": "The rules say CRITICAL. Run the agent once (it keeps the page: no feedback yet). Then click "
                  "'Mark as FALSE ALARM' at the bottom, as the operator, and run the agent again: it turns it into noise.",
        "pick": lambda i: key_for(i) == "logs:E2+E3+E4"},
}


# ------------------------------------------------------------------ data (cached)
@st.cache_resource
def run_rules():
    alerts = load_alerts(paths.ALERTS)
    return alerts, RuleTriage(TriageConfig()).process(alerts)


@st.cache_resource
def history():
    """Every incident of P1's full run, used by the tools as 'the past' in the live view."""
    return RuleTriage(TriageConfig(), id_prefix="HIST").process(load_alerts(paths.ALERTS))


def memory() -> Memory:
    return Memory(MEMORY_PATH)


alerts, incidents = run_rules()


# ------------------------------------------------------------------ live pipeline view
ACTIVE, DONE, IDLE = "#cfe3ff", "#e7f5e9", "#f2f2f2"


def flow_dot(c: dict, active: str) -> str:
    """The pipeline as a graphviz picture; counts in c, the running node highlighted."""
    def node(key, label, shape="box"):
        fill = ACTIVE if key == active else (DONE if c.get(f"done_{key}") else IDLE)
        return f'{key} [label="{label}", shape={shape}, style="rounded,filled", fillcolor="{fill}", fontcolor="#111111"];'
    n = lambda k: c.get(k, "·")
    return "\n".join([
        "digraph { rankdir=LR; bgcolor=transparent; nodesep=0.35; ranksep=0.5;",
        'node [fontname="Helvetica", fontsize=11, color="#888888"]; edge [color="#999999", fontcolor="#999999", fontsize=10];',
        node("monitor", f"P1 Monitor\\n{n('alerts')} alerts"),
        node("layer1", f"Layer 1 · rules\\n{n('groups')} incidents"),
        node("inc1", f"INCIDENT\\n{n('l1_incident')}", "box"),
        node("watch1", f"WATCH\\n{n('l1_watch')}", "box"),
        node("noise1", f"NOISE\\n{n('l1_noise')}", "box"),
        node("layer2", f"Layer 2 · LLM agent\\n{n('reviewed')} reviewed"),
        node("p3", f"P3 Diagnosis\\n{n('sent')} incidents"),
        node("log", f"Log only\\n{n('final_noise')} noise", "box"),
        node("watch2", f"Keep watching\\n{n('final_watch')} case(s)", "box"),
        "monitor -> layer1; layer1 -> inc1; layer1 -> watch1; layer1 -> noise1;",
        'watch1 -> layer2 [label="unsure"]; inc1 -> layer2 [label="HDFS only", style=dashed];',
        "inc1 -> p3; layer2 -> p3; layer2 -> watch2; layer2 -> log; noise1 -> log;",
        "}"])


def live_view(use_llm, model, mem):
    st.title("Live pipeline: Monitor → Triage → Diagnosis")
    key = st.radio("Window of P1's data to replay", list(WINDOWS), horizontal=True,
                   format_func=lambda k: WINDOWS[k].name)
    w = WINDOWS[key]
    st.caption(w.story + "  This is the same LangGraph graph as pipeline.py: monitor → layer1_rules → "
               "(only if some case is unsure) layer2_agent → to_diagnosis.")
    speed = st.slider("Speed of the replay (seconds per alert)", 0.0, 0.3, 0.05, 0.01)
    run = st.button("▶ Run the pipeline", type="primary")

    c = {}
    flow = st.empty()
    flow.graphviz_chart(flow_dot(c, ""))
    status = st.empty()
    col_in, col_out = st.columns([1, 2])
    inbox = col_in.empty()
    lanes = col_out.empty()
    agent_box = st.container()
    if not run:
        status.info("Pick a window and press ▶ to replay P1's alerts through the pipeline.")
        return

    def draw(active):
        flow.graphviz_chart(flow_dot(c, active))

    def show_lanes(incs, title):
        by = {d: [i for i in incs if i.decision == d] for d in Decision}
        with lanes.container():
            st.markdown(f"**{title}**")
            l1, l2, l3 = st.columns(3)
            for col, d, label in ((l1, Decision.INCIDENT, "🔴 INCIDENT → P3"), (l2, Decision.WATCH, "🟡 WATCH → layer 2"),
                                  (l3, Decision.NOISE, "⚪ NOISE → log")):
                col.markdown(f"**{label}** ({len(by[d])})")
                for i in by[d][:8]:
                    col.caption(f"{i.id} · {i.summary[:70]}")
                if len(by[d]) > 8:
                    col.caption(f"… and {len(by[d]) - 8} more")

    def on_review(k, total, inc, rv):
        c["reviewed"] = k
        draw("layer2")
        mark = "→ **changed**" if rv.decision != rv.rule_decision else ""
        tools = ", ".join(s["tool"] for s in rv.steps)
        agent_box.markdown(f"`{inc.id}` {rv.rule_decision.upper()} → **{rv.decision.upper()}** {mark}  \n"
                           f"<small>tools: {tools or '—'} · {rv.error or rv.reason}</small>",
                           unsafe_allow_html=True)

    graph = build_graph(replay_monitor(w),
                        make_layer2(mem, model=model, history=history(), on_review=on_review, use_llm=use_llm))
    state = {}
    for update in graph.stream({"log": []}, stream_mode="updates"):
        for node, out in update.items():
            state.update(out)
            if node == "monitor":
                c["alerts"] = len(out["alerts"])
                draw("monitor")
                status.info(f"P1's Monitor emits {len(out['alerts'])} alerts…")
                shown = []
                for a in out["alerts"]:
                    shown.append(f"`{a['timestamp'][5:16]}` {a['dataset']} · {a['signal']} · strength {a['score']:.2f}")
                    inbox.markdown("**Alerts arriving**  \n" + "  \n".join(shown[-12:]))
                    if speed and len(shown) <= 60:
                        time.sleep(speed)
                c["done_monitor"] = True
                draw("layer1")
                status.info("Layer 1 groups the alerts and judges each group…")
                time.sleep(0.6)
            elif node == "layer1_rules":
                incs = out["incidents"]
                c.update(groups=len(incs), done_layer1=True,
                         l1_incident=sum(i.decision == Decision.INCIDENT for i in incs),
                         l1_watch=sum(i.decision == Decision.WATCH for i in incs),
                         l1_noise=sum(i.decision == Decision.NOISE for i in incs),
                         done_inc1=True, done_watch1=True, done_noise1=True)
                show_lanes(incs, "After layer 1 (rules)")
                todo = sum(LLMTriage.needs_review(i) for i in incs)
                if todo:
                    draw("layer2")
                    msg = (f"{todo} unsure case(s) go to the LLM agent ({model})…" if use_llm
                           else f"{todo} unsure case(s) would go to the LLM agent, but it is switched off.")
                    status.warning(msg)
                    agent_box.markdown("#### Layer 2 · agent reviews")
                else:
                    draw("p3")
                    status.success("Nothing is unsure: the graph skips layer 2 and goes straight to Diagnosis.")
            elif node == "layer2_agent":
                c["done_layer2"] = True
                if state.get("final"):
                    show_lanes(state["final"], "After layer 2 (agent)")
            elif node == "to_diagnosis":
                final = state["final"]
                c.update(sent=len(out["to_diagnosis"]), done_p3=True, done_log=True, done_watch2=True,
                         final_noise=sum(i.decision == Decision.NOISE for i in final),
                         final_watch=sum(i.decision == Decision.WATCH for i in final))
                draw("")
                status.success(f"Done: {c['alerts']} alerts → {c['groups']} incidents → "
                               f"{c['sent']} sent to Diagnosis (P3).")
    with st.expander("Node log (what each LangGraph node wrote)"):
        st.json(state.get("log", []))


# ------------------------------------------------------------------ sidebar + routing
st.sidebar.title("Triage playground")
view = st.sidebar.radio("View", ["▶ Live pipeline", "🔍 Scenario deep-dive"])
if view == "🔍 Scenario deep-dive":
    sc_name = st.sidebar.radio("Scenario (from the ideation phase)", list(SCENARIOS))

st.sidebar.divider()
st.sidebar.subheader("LLM agent")
use_llm = st.sidebar.toggle("Use the LLM agent (needs Ollama running)", value=True)
model = st.sidebar.text_input("Model", "qwen2.5:7b")

st.sidebar.divider()
st.sidebar.subheader("Memory (human feedback)")
mem = memory()
st.sidebar.caption(f"{len(mem.data)} kind(s) of problem with feedback")
for k, v in mem.data.items():
    st.sidebar.write(f"`{k}` → **{v['verdict']}**")
if st.sidebar.button("Clear memory"):
    mem.clear(); mem.save(); st.rerun()


def deep_dive(sc_name, use_llm, model, mem):
    sc = SCENARIOS[sc_name]
    cases = [i for i in incidents if sc["pick"](i)]
    if not cases:
        st.error("No incident in P1's alerts matches this scenario.")
        return
    inc = st.sidebar.selectbox(f"Case ({len(cases)} found in P1's alerts)", cases,
                               format_func=lambda i: f"{i.id} · {i.summary[:60]}")
    name = sc_name
    st.title(name)
    c1, c2, c3 = st.columns(3)
    c1.metric("Domain", sc["tag"])
    c2.markdown(f"**Trigger**  \n{sc['trigger']}")
    c3.markdown(f"**Goal**  \n{sc['goal']}")
    st.info(f"**What we expect:** {sc['expect']}")

    # --- step 1: input
    st.header("① Alerts from P1's Monitor")
    st.caption(f"{len(inc.alerts)} alert(s) in this incident, out of {len(alerts):,} in the whole run")
    rows = [{"id": a.id, "time": a.start, "source": a.source, "dataset": a.dataset, "feature": a.feature,
             "kind": a.kind, "strength (score)": round(a.ratio, 2), "details": json.dumps(a.details)[:120]}
            for a in inc.alerts[:50]]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # --- step 2: rules
    st.header("② Layer 1 · rules (grouping, measures, decision)")
    r1, r2, r3 = st.columns(3)
    r1.metric("Rule decision", inc.decision.value.upper())
    r2.metric("Severity", inc.severity.value.upper())
    r3.metric("Alerts grouped", len(inc.alert_ids))
    st.write(f"**Summary:** {inc.summary}")
    with st.expander("Why: every reason the rules wrote", expanded=True):
        for r in inc.reasons:
            st.write(f"- {r}")

    # --- step 3: agent
    st.header("③ Layer 2 · LLM agent investigation")
    needs = LLMTriage.needs_review(inc)
    st.caption("The rules send this case to the agent." if needs else
               "The rules are confident here, so in the pipeline the agent would NOT be called. You can still ask it.")

    if st.button("Run the agent on this case", type="primary", disabled=not use_llm):
        tools = make_tools(incidents, mem)
        with st.spinner(f"{model} is investigating (about 10 s)…"):
            rv = LLMTriage(mem, model=model).review(inc, tools)
        st.session_state["review"] = (inc.id, rv)

    if not use_llm:
        st.warning("LLM agent switched off: only the rules are shown.")

    saved = st.session_state.get("review")
    if saved and saved[0] == inc.id:
        rv = saved[1]
        if rv.error and not rv.steps:
            st.error(f"The agent could not run ({rv.error}). Is Ollama open? The rule decision is kept.")
        for k, s in enumerate(rv.steps, 1):
            fact = json.loads(s["output"]).get("fact") or json.loads(s["output"]).get("note") or s["output"][:200]
            st.markdown(f"**Step {k} · the model called `{s['tool']}`**  \n→ {fact}")
        if rv.evidence:
            a, b = st.columns(2)
            a.markdown("**Checklist filled by the model**")
            a.json({k: v for k, v in rv.evidence.items() if k != "reason"})
            b.markdown("**Fact-check (the tools win)**")
            b.json(rv.facts)
            if rv.misread:
                st.warning(f"The model misread: {', '.join(rv.misread)}. The tools' values were used instead.")
            else:
                st.success("The model read every fact correctly.")
        d1, d2 = st.columns(2)
        d1.metric("Rules said", rv.rule_decision.upper())
        d2.metric("Agent decides", rv.decision.upper(),
                  delta=None if rv.decision == rv.rule_decision else "changed")
        st.write(f"**Reason:** {rv.reason}")
        if rv.overridden:
            st.info("Guardrail: the evidence said to silence this page, but there is no human feedback. The page is kept.")

    # --- step 4: output + feedback
    st.header("④ What Diagnosis (P3) receives, and operator feedback")
    final = saved[1].decision if saved and saved[0] == inc.id else inc.decision.value
    out = {k: v for k, v in asdict(inc).items() if k not in ("alerts",)}
    out.update(decision=final, start=str(inc.start), end=str(inc.end), severity=inc.severity.value,
               alert_ids=out["alert_ids"][:5] + (["…"] if len(inc.alert_ids) > 5 else []))
    if final == "incident":
        st.success("Sent to Diagnosis (P3).")
    elif final == "watch":
        st.warning("Kept in WATCH: not sent to P3 yet.")
    else:
        st.info("Noise: logged only, not sent.")
    st.json(out, expanded=False)

    st.markdown(f"**You are the operator.** Close this incident and the agent remembers it for every incident "
                f"of the same kind (`{key_for(inc)}`).")
    f1, f2 = st.columns(2)
    if f1.button("Mark as FALSE ALARM"):
        mem.add_feedback(key_for(inc), "false_alarm", "given in the playground"); mem.save()
        st.session_state.pop("review", None); st.rerun()
    if f2.button("Mark as REAL problem"):
        mem.add_feedback(key_for(inc), "real", "given in the playground"); mem.save()
        st.session_state.pop("review", None); st.rerun()

if view == "▶ Live pipeline":
    live_view(use_llm, model, mem)
else:
    deep_dive(sc_name, use_llm, model, mem)