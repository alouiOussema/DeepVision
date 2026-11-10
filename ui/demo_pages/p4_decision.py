"""Step 4: Decision (P4) chooses what happens next: act alone, ask a human, hand over, or nothing."""
import pandas as pd
import streamlit as st

from demo_pages.common import explain, family_picker, finding, need_data, note

st.title("4 · Decision: what happens next, and who decides")
explain(
    "Decision multiplies **how risky the action is** by **how sure Diagnosis is**, and sends every incident one of "
    "four ways:\n\n"
    "- **act** → the Action agent may fix it alone (safe and confident)\n"
    "- **approval** → a human must approve this exact action first (risky, or not sure enough)\n"
    "- **escalate** → no automatic action, a person takes over (with the evidence)\n"
    "- **none** → nothing is broken, nobody is paged (logged)\n\n"
    "It has **no LLM**: rules in code, tested one by one, so every decision can be audited.",
    how=("| Cause | Action | Alone when | In our system |\n|---|---|---|---|\n"
         "| flaky test | re-run once | confidence ≥ 0.60 | **act alone** |\n"
         "| bad deploy / config change | rollback | (table: ≥ 0.80) | **always a human** (team decision, deck slide 14) |\n"
         "| data drift | retrain | ≥ 0.70, by day | retrain alone, **a human approves the new model's deploy** |\n"
         "| code regression, infra, data quality, unknown | escalate | never | **escalate** |\n\n"
         "Then the rule chain: confidence < 0.50 → escalate with no action; Diagnosis unsure → approval. Guards that "
         "remember: a flaky test is re-run **once** per project per day (red twice = a real bug); the same action on "
         "the same thing within 60 min is a **duplicate**; more than 5 automatic actions in an hour trips a **circuit "
         "breaker**. A kill switch and a dry-run mode exist."),
    measure=("- **Wrong automatic actions**: a fix run with no human on the wrong cause or release (deck target: 0).\n"
             "- **Silenced faults**: *none* decided on a real fault (target: 0).\n"
             "- **Closed as duplicate**: a real fault closed because an order on the same key was already open."))

d = need_data()
fam = family_picker("dec_f")
k = d["kpis"][fam]
recs = d["scored"] if fam == "all" else d["scored"][d["scored"]["family"] == fam]
dec = recs[recs["action"].astype(bool)]

routes = dec["route"].value_counts()
c = st.columns(4)
c[0].metric("Act alone", int(routes.get("act", 0)))
c[1].metric("Ask a human first", int(routes.get("approval", 0)))
c[2].metric("Hand over to a person", int(routes.get("escalate", 0)))
c[3].metric("Nothing to do", int(routes.get("none", 0)))
c = st.columns(3)
c[0].metric("Wrong automatic actions", k["wrong_auto_actions"], help="deck target: 0")
c[1].metric("Silenced faults", k["silenced_faults"], help="target: 0")
c[2].metric("Faults closed as a duplicate", k["closed_as_duplicate"])

st.subheader("Which rule decided")
if len(dec):
    t = dec.groupby(["truth", "rule", "route", "action"]).size().reset_index(name="incidents")
    t["truth"] = t["truth"].str.replace("_", " ")
    st.dataframe(t.sort_values("incidents", ascending=False), hide_index=True, width="stretch")

    st.subheader("Read one decision trail")
    ev = {e["incident_id"]: e for e in d["events"] if e["agent"] == "decision"}
    options = [r for r in dec.itertuples() if r.key in ev]
    pick = st.selectbox("Incident", options, format_func=lambda r: f"{r.key} · {r.category} → {r.route}: {r.action}"
                        f"{' ' + r.target if r.target else ''} [{r.rule}]")
    if pick:
        for step in ev[pick.key]["reason"].split(" -> "):
            st.markdown(f"- {step}")
note("Being right is not enough to act alone: a 0.98-confident rollback still waits for a human, because the team "
     "decided rollbacks are risky. This costs autonomy and buys trust.")
if k["closed_as_duplicate"]:
    finding(f"{k['closed_as_duplicate']} storage (HDFS) failures were closed as *duplicates*. For log incidents the "
            "guard's key is the same for every pattern (`logs:hdfs:block_events`): once one HDFS failure is escalated, "
            "a *different* one in the same hour is closed, and its own evidence reaches nobody. Fix: key on the pattern.")
