"""Results & what we learned: the deck's KPIs, where faults get lost, and the findings for the team."""
import altair as alt
import pandas as pd
import streamlit as st

from demo_pages.common import FAMILIES, ORDER, family_picker, fmt, need_data

from agents.evaluation.score import ROWS

BLUE = "#2a78d6"

st.title("Results & what we learned")
d = need_data()
k = d["kpis"]

st.subheader("The deck's KPIs, measured")
a = k["all"]
c = st.columns(5)
c[0].metric("MTTD", fmt(a["mttd_min"], " min"), help="fault injected → first alert (median, bad deploy)")
c[1].metric("Noise reduction", fmt(a["noise_reduction_pct"], " %"), help="1 − incidents sent / alerts")
c[2].metric("Diagnosis accuracy", fmt(a["diagnosis_accuracy_pct"], " %"))
c[3].metric("False action rate", fmt(a["false_action_pct"], " %"), help=f"{a['fixes_run']} fixes run")
c[4].metric("MTTR", fmt(a["mttr_min"], " min"), help="fault → verified fix (median)")
c = st.columns(5)
c[0].metric("Wrong automatic actions", a["wrong_auto_actions"], help="deck target: 0")
c[1].metric("Silenced faults", a["silenced_faults"], help="target: 0")
c[2].metric("Fixable faults resolved", fmt(a["resolved_pct"], " %"))
c[3].metric("... with no human", fmt(a["auto_resolved_pct"], " %"))
c[4].metric("Wrong fixes reported as OK", a["wrong_but_reported_executed"], help="the Action agent's honesty")

with st.expander("Every KPI, per scenario", expanded=True):
    cols = [f for f in ORDER if f in k] + ["all"]
    name = lambda f: "All" if f == "all" else f"{FAMILIES[f]['scenario']} · {FAMILIES[f]['short']}"
    st.dataframe(pd.DataFrame({name(f): {label: fmt(k[f][key], unit) for key, label, unit in ROWS} for f in cols}),
                 width="stretch", height=36 * (len(ROWS) + 1))

st.subheader("Where do real faults get lost?")
fam = family_picker("res_f")
inj = d["scored_inj"] if fam == "all" else d["scored_inj"][d["scored_inj"]["family"] == fam]
recs = d["scored"]
faults = inj[inj["is_fault"]]
by = {i: recs[recs["injection"] == i] for i in faults["id"]}
steps = [
    ("injected", len(faults)),
    ("raised an alert (Monitor)", int((faults["n_alerts"] > 0).sum())),
    ("reached Diagnosis (Triage)", sum(by[i]["sent"].any() for i in faults["id"])),
    ("right cause found (Diagnosis)", sum((by[i]["diagnosis_right"] == True).any() for i in faults["id"])),  # noqa: E712
    ("handled right (Decision + Action)", sum(((by[i]["action_right"] == True) & (by[i]["status"] != "closed")  # noqa: E712
                                              & ~by[i]["wrong_fix"]).any() for i in faults["id"])),
]
fun = pd.DataFrame(steps, columns=["step", "faults"])
fun["share"] = fun["faults"] / max(len(faults), 1)
st.altair_chart(alt.Chart(fun).mark_bar(color=BLUE, cornerRadiusEnd=4, height=22).encode(
    x=alt.X("faults:Q", title="real faults", axis=alt.Axis(tickMinStep=1)),
    y=alt.Y("step:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
    tooltip=["step:N", "faults:Q", alt.Tooltip("share:Q", format=".0%")]).properties(height=200) +
    alt.Chart(fun).mark_text(align="left", dx=6, color="#52514e").encode(
        x="faults:Q", y=alt.Y("step:N", sort=None), text=alt.Text("share:Q", format=".0%")),
    use_container_width=True)
st.caption("\"Handled right\" = fixed and verified, or handed to the right person when no machine fix exists.")

st.subheader("What we learned")
st.markdown(f"""
**1. The middle of the chain works.** When an incident reaches Diagnosis, the cause is right {fmt(a['diagnosis_accuracy_pct'], ' %')}
of the time, only {a['wrong_auto_actions']} fix ran alone on the wrong cause (and it was caught and reported), no real fault was
silenced, and a verified fix takes {fmt(a['mttr_min'], ' min')} instead of the day or more most engineers need today.

**2. Most faults are lost at the start.** {a['missed_by_monitor']} faults raised no alert (slow leaks, small slowdowns) and
{a['missed_by_triage']} were kept as *watch* or *noise* by Triage. Cutting {fmt(a['noise_reduction_pct'], ' %')} of the noise has a price.

**3. Data drift never reaches Diagnosis without Triage's LLM** (P2): drift is always *watch* for the rules.

**4. The duplicate guard merges different storage failures** (P4): {a['closed_as_duplicate']} HDFS faults closed as copies of
another one. A key that includes the log pattern fixes it.

**5. Rollbacks are never automatic, on purpose**: trust before autonomy. Every bad-deploy fix waited for a human.
""")

with st.expander("Limits (say them before someone asks)"):
    st.markdown(
        "- Only the bad-deploy scenario goes through P1's real detector; the other alerts are made in its format.\n"
        "- The human is simulated (approves after 5 min) and time is virtual.\n"
        "- Triage's LLM layer is off (no Ollama): findings 3 and 4 may change with it on.\n"
        "- Only cases the Diagnosis agent never trained on are graded.\n"
        "- These are our injected failures, not production.")
with st.expander("Next steps"):
    st.markdown(
        "- Run the evaluation with Triage's LLM on, and compare.\n"
        "- Fix the HDFS duplicate key (P4) and the drift path (P2), then re-run: the funnel shows the gain.\n"
        "- Real adapters for the Action agent (CI API, `kubectl rollout undo`, MLflow, Slack).\n"
        "- Inject failures into a real running pipeline instead of simulations.")
