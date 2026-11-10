"""Step 5: the human gate: a person approves every risky action before it runs."""
import pandas as pd
import streamlit as st

from demo_pages.common import explain, family_picker, need_data, note

st.title("5 · Human approval: the person stays in control")
explain(
    "When Decision says **approval**, nothing runs until a person reads the agent's explanation and approves (or "
    "rejects) **this exact action**: which release to roll back, why, how sure the system is. This is Yassine's "
    "need from the deck: *\"show me why before it acts\"*, and an override on risky actions.",
    how=("In the real system this is the approval page (`python -m streamlit run ui/decision_app.py`): pending orders "
         "come first, with the evidence, the decision trail and the risk. *Approve* → the Action agent may run it; "
         "*reject* → nothing runs. Every click is written to the decision log.\n\n"
         "**In this demo the human is simulated**: they approve every pending order 5 minutes after it appears. "
         "Those orders still count as *needed a human*: autonomy is never overstated."),
    measure="- How many incidents needed a person, and for what. The 5 minutes count in the time-to-repair (MTTR).")

d = need_data()
fam = family_picker("hum_f")
recs = d["scored"] if fam == "all" else d["scored"][d["scored"]["family"] == fam]
appr = recs[recs["route"] == "approval"]
esc = recs[recs["route"] == "escalate"]

c = st.columns(3)
c[0].metric("Actions a human approved", len(appr))
c[1].metric("Incidents handed over to a person", len(esc))
c[2].metric("Approval delay (simulated)", f"{d['res']['config']['human_delay_min']:g} min")

st.subheader("What the human approved")
if len(appr):
    ev = {e["incident_id"]: e for e in d["events"] if e["agent"] == "human"}
    st.dataframe(pd.DataFrame([{
        "incident": r.key, "action": f"{r.action} {r.target}".strip(), "diagnosis": f"{r.category} ({r.confidence:.2f})",
        "approved at": (ev[r.key]["at"] or "")[:16].replace("T", " ") if r.key in ev else "–",
        "was it the right release?": "✅ yes" if r.action_right else "❌ no",
        "result": r.outcome or "–"} for r in appr.itertuples()]), hide_index=True, width="stretch")
else:
    st.info("No action needed a human's approval in this scenario.")

if len(esc):
    st.subheader("What was handed over to a person")
    t = esc.groupby(["truth", "category"]).size().reset_index(name="incidents")
    t["who"] = t["category"].map(lambda c: {"code_regression": "the commit's author"}.get(c, "the on-call engineer"))
    st.dataframe(t, hide_index=True, width="stretch")
note("A human approving a rollback is only as good as what they see: that is why every order carries the evidence "
     "and the decision trail (open one on *Follow one incident*).")
