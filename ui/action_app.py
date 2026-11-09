"""Action results page (P5): what the Action agent executed, verified, failed or undid, and who it told.

    python -m agents.action.demo                   # optional: demo results
    python -m streamlit run ui/action_app.py

Read-only on purpose: approving is ui/decision_app.py's job; this page only shows outcomes, so nobody can trigger an
action from here. A retrained model waiting for its deploy is listed, but deploying it is not possible from this page.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere

from agents.action import kpis, paths                           # noqa: E402
from agents.action.demo import demo_inbox, run_demo             # noqa: E402
from agents.decision.inbox import ActionInbox                   # noqa: E402
from common.schemas import ActionStatus                         # noqa: E402

st.set_page_config(page_title="Action results", layout="wide")
DONE = (ActionStatus.EXECUTED, ActionStatus.FAILED, ActionStatus.REVERTED)
ICON = {"executed": "✅ executed", "failed": "❌ failed", "reverted": "↩️ reverted"}


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


st.sidebar.title("Action results")
st.sidebar.caption("P5 executes what was approved, verifies it, and reports honestly. Simulation by default.")
use_demo = st.sidebar.checkbox("Demo data", value=False, help="python -m agents.action.demo: the deck's scenarios")
if st.sidebar.button("Run the deck's scenarios"):
    run_demo()
    use_demo = True
    st.sidebar.success("Demo results written: tick 'Demo data'.")
inbox = demo_inbox() if use_demo else ActionInbox()
notes_path = paths.DEMO_NOTIFICATIONS if use_demo else paths.NOTIFICATIONS
deploy_path = paths.DEMO_DEPLOY_REQUESTS if use_demo else paths.DEPLOY_REQUESTS
items, events = inbox.all(), inbox.log.events()
k = kpis.compute(items, events)
st.sidebar.metric("Orders run", k["orders_run"])
st.sidebar.metric("Ready, not run yet", len(inbox.ready()))
for label, key, unit in (("MTTR (median)", "mttr_min_median", " min"), ("Resolved with no human", "auto_resolved_pct", " %"),
                         ("Fixes that worked", "fix_rate_pct", " %"), ("False actions", "false_action_pct", " %")):
    st.sidebar.metric(label, "-" if k[key] is None else f"{k[key]}{unit}")
if k["simulated"]:
    st.sidebar.warning("Simulation: no real system was touched, times are virtual.")

st.title("What the Action agent did")
done = [d for d in items if d.status in DONE]
if not done:
    st.info("Nothing executed yet. Run `python -m agents.action.demo` and tick **Demo data**, "
            "or `python pipeline.py sim --action`.")
tab_done, tab_notes, tab_deploy, tab_log = st.tabs(["Results", "Who was told", "Models waiting for a deploy", "Trail"])

with tab_done:
    for d in sorted(done, key=lambda x: x.status.value):
        steps = [e["detail"] for e in events if e.get("decision_id") == d.id and e.get("agent") == "action"]
        with st.expander(f"{ICON[d.status.value]} - {d.incident_id} - {d.action.value}"
                         f"{' ' + d.target if d.target else ''} - {d.category.value}", expanded=d.status is not ActionStatus.EXECUTED):
            st.markdown(f"**Result:** {d.result}")
            st.markdown(f"**Approved by:** {d.decided_by}  ·  **rule:** `{d.rule}`")
            st.markdown("**Steps**")
            for s in steps:
                st.markdown(f"- {s}")

with tab_notes:
    notes = jsonl(notes_path)
    st.caption(str(notes_path))
    if notes:
        st.dataframe(pd.DataFrame(notes)[["at", "to", "subject", "incident"]].iloc[::-1], hide_index=True, width="stretch")
        for n in notes[::-1][:5]:
            with st.expander(f"{n['to']}: {n['subject']}"):
                st.text(n["body"])

with tab_deploy:
    st.caption("A retrained model is NEVER deployed by the Action agent. A human decides (open question with P4: "
               "a follow-up order in the action inbox, or a gate of P5's own).")
    reqs = jsonl(deploy_path)
    if reqs:
        st.dataframe(pd.DataFrame([{**r, **{f"cmp.{a}": b for a, b in r.get("comparison", {}).items()}} for r in reqs])
                     .drop(columns=["comparison"], errors="ignore"), hide_index=True, width="stretch")

with tab_log:
    rows = [e for e in events if e.get("agent") == "action" or e.get("event") in {s.value for s in DONE}]
    if rows:
        st.dataframe(pd.DataFrame(rows)[["at", "event", "incident_id", "action", "target", "detail"]].iloc[::-1],
                     hide_index=True, width="stretch")
st.caption(f"Inbox: {inbox.path}")
