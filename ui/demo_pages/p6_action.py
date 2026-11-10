"""Step 6: the Action agent (P5) fixes, waits, verifies, undoes if worse, and reports honestly."""
import pandas as pd
import streamlit as st

from demo_pages.common import explain, family_picker, finding, fmt, need_data, note

st.title("6 · Action: fix it, then check that it really worked")
explain(
    "The Action agent runs **only approved orders**, never twice. It does not stop at *done*: it waits, measures "
    "again, and reports one of three honest results:\n\n"
    "- **executed** → fixed and verified\n- **failed** → done, but the problem is still there (a person is told)\n"
    "- **reverted** → it made things worse, so it undid its own action (a person is told)",
    how=("| Action | Does | Checks after | If it did not work |\n|---|---|---|---|\n"
         "| re-run | re-runs the failed CI build once | is the build green? | red again = a real bug → tell the author |\n"
         "| rollback | undoes the release / config | after 15 min, are errors and latency back to normal? | still bad → "
         "on-call; worse → redeploy it and tell on-call |\n"
         "| retrain | trains a challenger, compares it with the live model | ≥ 2 % better? | keep the live model, "
         "send the drift report. **Never deploys alone** |\n"
         "| escalate | sends the evidence to a person | delivered? | retry |\n\n"
         "**Simulation**: the CI, the deployments and the model registry are simulated. In this demo the simulated "
         "world knows the truth: only the release we broke hurts, so a rollback of the wrong one fails its check."),
    measure=("- **False action rate**: fixes run on the wrong cause or release, or that failed / were reverted.\n"
             "- **Resolved**: injected fixable faults that were fixed and verified, with or without a human.\n"
             "- **MTTR**: from the fault's start to the verified fix (virtual time)."))

d = need_data()
fam = family_picker("act_f")
k = d["kpis"][fam]
recs = d["scored"] if fam == "all" else d["scored"][d["scored"]["family"] == fam]
ran = recs[recs["outcome"].astype(bool) & (recs["outcome"] != "skipped")]

c = st.columns(5)
c[0].metric("Fixes run", k["fixes_run"], help="rerun, rollback, retrain (escalations not counted)")
c[1].metric("False action rate", fmt(k["false_action_pct"], " %"))
c[2].metric("Fixable faults resolved", fmt(k["resolved_pct"], " %"))
c[3].metric("... with no human at all", fmt(k["auto_resolved_pct"], " %"))
c[4].metric("MTTR (median)", fmt(k["mttr_min"], " min"), help="fault start → verified fix")

st.subheader("What the Action agent did")
if len(ran):
    t = pd.crosstab(ran["action"], ran["outcome"])
    t.index.name, t.columns.name = "action", None
    st.dataframe(t, width="stretch")
    told = ran.explode("notified")["notified"].dropna().value_counts()
    if len(told):
        st.caption("Who was told: " + ", ".join(f"{who} ({n})" for who, n in told.items()))

    st.subheader("Read one execution")
    ev = {e["incident_id"]: e for e in d["events"] if e["agent"] == "action"}
    options = [r for r in ran.itertuples() if r.key in ev]
    pick = st.selectbox("Order", options, format_func=lambda r: f"{r.key} · {r.action} {r.target or ''} → {r.outcome}")
    if pick:
        for step in ev[pick.key]["reason"].split(" | "):
            st.markdown(f"- {step}")
else:
    st.info("Nothing was executed in this scenario.")
note(f"The MTTR of a rollback = detection + {d['res']['config']['human_delay_min']:g} min for the human + 15 min "
     "waiting before checking the metrics (the bad-deploy runbook). Compare: 84 % of engineers need a day or more.")
bad = recs[recs["wrong_fix"] & (recs["route"] == "act")]
if len(bad):
    lines = [f"- truth **{r.truth}**, diagnosed **{r.category}** → {r.action} alone → reported **{r.outcome}**"
             f"{', told ' + ', '.join(r.notified) if r.notified else ''}" for r in bad.itertuples()]
    finding(f"{len(bad)} wrong automatic action(s). Each one was caught by the check after acting and reported, "
            "not hidden:\n" + "\n".join(lines))
