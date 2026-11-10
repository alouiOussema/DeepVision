"""Step 3: Diagnosis (P3) names the root cause, with evidence and a calibrated confidence."""
import pandas as pd
import streamlit as st

from demo_pages.common import FAMILIES, explain, family_picker, finding, fmt, need_data, note

st.title("3 · Diagnosis: find the WHY, and how sure we are")
explain(
    "For each incident Triage sends, Diagnosis names **one of 8 causes** (flaky test, code regression, bad deploy, "
    "config change, infra, data drift, data quality, unknown), the **release to undo** when there is one, and a "
    "**confidence** that means what it says (0.9 = right about 90 % of the time). Every claim cites a piece of "
    "evidence a human can check.",
    how=("1. **Collect evidence** (7 collectors): which change went out just before the metrics degraded, the shape "
         "of the metrics (step, ramp, blip, recovery), other projects red at the same time, log templates, history.\n"
         "2. **Rank causes**: rules + a gradient-boosting classifier, blended, then calibrated.\n"
         "3. **Retrieve** runbooks and past incidents from the knowledge base.\n"
         "4. **Explain** with an LLM that may only cite the evidence (fact-checked; off in this demo: template text).\n"
         "5. **Judge**: below 0.60 confidence it asks for a human; below 0.40 it says *unknown*."),
    measure=("- **Accuracy**: right cause, and the right release when there is one, on the incidents it received.\n"
             "- The table below compares the truth we injected with what Diagnosis said."))

d = need_data()
fam = family_picker("diag_f")
k = d["kpis"][fam]
recs = d["scored"] if fam == "all" else d["scored"][d["scored"]["family"] == fam]
sent = recs[recs["sent"] & recs["category"].astype(bool)]

right_rel = sent[sent["truth_suspect"].astype(bool)]
c = st.columns(4)
c[0].metric("Incidents diagnosed", len(sent))
c[1].metric("Right cause", fmt(k["diagnosis_accuracy_pct"], " %"))
c[2].metric("Right release to undo", f"{int((right_rel['suspect'] == right_rel['truth_suspect']).sum())} / {len(right_rel)}"
            if len(right_rel) else "–", help="when the fault is a release or a config push")
c[3].metric("Asked for a human (confidence < 0.60)", int((sent["confidence"] < 0.60).sum()))

st.subheader("Truth vs what Diagnosis said")
if len(sent):
    cm = pd.crosstab(sent["truth"].str.replace("_", " "), sent["category"].str.replace("_", " "))
    cm.index.name, cm.columns.name = "injected truth ↓  /  diagnosis →", None
    st.dataframe(cm, width="stretch")
    st.caption("The diagonal is right. Anything off the diagonal is a wrong diagnosis.")

    st.subheader("Read one diagnosis")
    ev = {(e["incident_id"]): e for e in d["events"] if e["agent"] == "diagnosis"}
    options = [r for r in sent.itertuples() if r.key in ev]
    pick = st.selectbox("Incident", options, format_func=lambda r: (
        f"{r.key} · truth {r.truth}{' ' + r.truth_suspect if r.truth_suspect else ''} → said {r.category}"
        f"{' ' + r.suspect if r.suspect else ''} ({r.confidence:.2f}) {'✅' if r.diagnosis_right else '❌'}"))
    if pick:
        e = ev[pick.key]
        st.markdown(f"**{e['action']}** · {e['result']}")
        st.markdown(e["reason"])
note("A wrong diagnosis is not automatically a wrong action: when Diagnosis says *infra* or *unknown*, Decision "
     "never acts alone, so the mistake reaches a person instead of the production system.")
wrong = sent[sent["diagnosis_right"] == False]                                 # noqa: E712 (pandas)
if len(wrong):
    what = wrong.groupby(["truth", "category", "action"]).size().reset_index(name="n")
    lines = [f"- {r.n} × truth **{r.truth}**, said **{r.category}** → decision **{r.action}**" for r in what.itertuples()]
    finding(f"{len(wrong)} wrong diagnosis(es) here, and what happened next:\n" + "\n".join(lines))
