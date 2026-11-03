"""Action approval page (P4): a human approves or rejects the actions the Decision agent will not take alone.

    python -m agents.decision.demo                 # optional: demo orders
    python -m streamlit run ui/decision_app.py

This is NOT ui/approval_app.py (that one reviews the DIAGNOSIS and teaches P3). This page gates the ACTION:
approve -> status "approved", the Action agent (P5) may run it; reject -> nothing is executed.
Every click is written to the decision log (traceability).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere

from agents.decision import kpis, paths, policy                  # noqa: E402
from agents.decision.demo import demo_inbox, run_demo            # noqa: E402
from agents.decision.inbox import ActionInbox, InboxError        # noqa: E402
from common.schemas import ActionStatus                          # noqa: E402

st.set_page_config(page_title="Action approval", layout="wide")
RISK_RANK = {"high": 0, "medium": 1, "low": 2, "none": 3, "": 4}
COLOR = {"rollback": "orange", "rerun": "blue", "retrain": "violet", "escalate": "red", "none": "green"}

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Action approval")
st.sidebar.caption("P4 decides what is safe. A human decides the rest.")
use_demo = st.sidebar.checkbox("Demo data", value=False, help="python -m agents.decision.demo: the deck's scenarios")
if st.sidebar.button("Load the deck's scenarios"):
    run_demo()
    use_demo = True
    st.sidebar.success("Demo orders loaded: tick 'Demo data'.")
inbox = demo_inbox() if use_demo else ActionInbox()
items = inbox.all()
show_done = st.sidebar.checkbox("Show resolved orders", value=False)
k = kpis.compute(items)
st.sidebar.metric("Waiting for a human", k["waiting_for_human"])
st.sidebar.metric("Ready for the Action agent", len(inbox.ready()))
for label, key in (("Act alone", "act_alone_pct"), ("Reached a person", "human_load_pct"),
                   ("False action rate", "false_action_rate"), ("Human override rate", "override_rate")):
    st.sidebar.metric(label, "-" if k[key] is None else f"{k[key]} %")

# ------------------------------------------------------------------ main
st.title("Actions waiting for you")
if not items:
    st.info("No orders yet. Run the pipeline (`python pipeline.py sim`), or tick **Demo data** and load the scenarios.")

tab_wait, tab_all, tab_log, tab_kpi = st.tabs(["Approvals", "All orders", "Decision log", "KPIs"])

with tab_wait:
    pending = sorted(inbox.pending(), key=lambda d: (RISK_RANK.get(d.risk, 4), d.confidence))
    if items and not pending:
        st.success("Nothing is waiting for you.")
    for d in pending:
        what = d.action.value + (f" `{d.target}`" if d.target else "")
        bar = d.snapshot.get("bar")
        with st.expander(f"{d.incident_id} - {d.action.value}{' ' + d.target if d.target else ''} - "
                         f"{d.category.value} - confidence {d.confidence:.2f} - risk {d.risk}", expanded=True):
            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"**Proposed action:** :{COLOR.get(d.action.value, 'gray')}[**{what}**]")
                if d.snapshot.get("incident_summary"):
                    st.markdown(f"**Incident:** {d.snapshot['incident_summary']}")
                st.markdown("**Why this needs you** (the decision trail)")
                for r in d.reasons:
                    st.markdown(f"- {r}")
                st.markdown("**Diagnosis**")
                st.info(d.snapshot.get("explanation", d.summary))
                for w in d.snapshot.get("warnings", []):
                    st.warning(w)
            with right:
                st.progress(min(d.confidence, 1.0), text=f"confidence {d.confidence:.2f}"
                            + (f" (bar to act alone: {bar:.2f})" if bar else ""))
                hyp = d.snapshot.get("hypotheses", [])
                if hyp:
                    st.markdown("**Hypotheses**")
                    st.bar_chart(pd.DataFrame({"probability": {h["category"]: h["probability"] for h in hyp}}),
                                 horizontal=True, height=160)
                c = d.constraints
                st.caption(f"P5 will verify after {c.get('verify_after_min', '?')} min, "
                           f"{'revert if worse, ' if c.get('revert_if_worse') else ''}max {c.get('max_retries', '?')} retries"
                           + (", and a human approves the deploy of the new model" if c.get("human_before_deploy") else ""))
            if d.snapshot.get("evidence"):
                st.markdown("**Evidence**")
                st.dataframe(pd.DataFrame(d.snapshot["evidence"]), hide_index=True, width="stretch")
            with st.form(f"form-{d.id}"):
                c1, c2 = st.columns(2)
                reviewer = c1.text_input("Your name", value="operator", key=f"who-{d.id}")
                note = c2.text_input("Note (optional)", key=f"note-{d.id}")
                ok, no = st.columns(2)
                approve, reject = ok.form_submit_button("Approve", type="primary"), no.form_submit_button("Reject")
                if approve or reject:
                    try:
                        inbox.resolve(d.id, "approve" if approve else "reject", reviewer, note)
                        st.rerun()
                    except InboxError as e:
                        st.error(str(e))

with tab_all:
    rows = [{"id": d.id, "incident": d.incident_id, "action": d.action.value, "target": d.target, "route": d.route.value,
             "status": d.status.value, "cause": d.category.value, "confidence": d.confidence, "risk": d.risk,
             "rule": d.rule, "decided by": d.decided_by, "last reason": d.reasons[-1] if d.reasons else ""}
            for d in items if show_done or d.status not in (ActionStatus.EXECUTED, ActionStatus.CLOSED, ActionStatus.REJECTED)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with tab_log:
    log = inbox.log.events()
    st.caption(f"{inbox.log.log_path}")
    if log:
        st.dataframe(pd.DataFrame(log).drop(columns=["data_used"], errors="ignore").iloc[::-1], hide_index=True, width="stretch")

with tab_kpi:
    st.json({k_: v for k_, v in k.items() if k_ != "per_cause"})
    if k["per_cause"]:
        st.dataframe(pd.DataFrame(k["per_cause"]).T, width="stretch")
    bars = {c.value: r.bar for c, r in policy.load().items()}
    for hint in kpis.suggest_thresholds(items, bars):
        st.warning(hint)
    st.caption("Threshold hints are a report only: the bars live in knowledge_base/action_risk_table.md.")

st.caption(f"Inbox: {inbox.path}")
