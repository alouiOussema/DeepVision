"""Step 2: Triage (P2) groups alerts into incidents and decides which ones deserve attention."""
import pandas as pd
import streamlit as st

from demo_pages.common import FAMILIES, ORDER, explain, family_picker, finding, fmt, need_data, note

st.title("2 · Triage: turn many alarms into a few real incidents")
explain(
    "Many alerts describe the **same problem** (six failed builds of one commit, error rate and latency after one "
    "release). Triage **groups** them into one **incident**, measures it (how strong, how wide, how long, how many "
    "detectors agree) and decides:\n\n"
    "- **incident** → sent to Diagnosis\n- **watch** → kept and re-checked if it continues\n"
    "- **noise** → logged, nobody is paged",
    how=("**Layer 1, rules** (always on): group by time, commit, service or log pattern, then one rule set per kind "
         "of signal. For CI, a *high* score means *probably flaky*, so it is *less* serious.\n\n"
         "**Layer 2, an LLM agent** (only for unsure cases: *watch* and HDFS patterns): it calls tools, fills an "
         "evidence checklist, and the code applies the policy. It may only silence an incident when a human has "
         "already said it was a false alarm. **It is off in this demo** (no Ollama)."),
    measure=("- **Noise reduction** = 1 − incidents sent / alerts in: the share of alarm load nobody has to read.\n"
             "- **Faults held back**: real injected faults whose alerts all ended up as *watch* or *noise*: the price "
             "of cutting noise."))

d = need_data()
fam = family_picker("triage_f")
k = d["kpis"][fam]
recs = d["scored"] if fam == "all" else d["scored"][d["scored"]["family"] == fam]

c = st.columns(4)
c[0].metric("Alerts in", k["alerts"])
c[1].metric("Incidents made", k["incidents"])
c[2].metric("Sent to Diagnosis", k["sent_to_diagnosis"])
c[3].metric("Noise reduction", fmt(k["noise_reduction_pct"], " %"), help="1 − sent / alerts")

st.subheader("What Triage decided")
tab = pd.crosstab(recs["family"].map(lambda f: f"{FAMILIES[f]['scenario']} · {FAMILIES[f]['short']}"),
                  recs["triage"]).reindex(columns=["incident", "watch", "noise"], fill_value=0)
tab.index.name, tab.columns.name = "Scenario", None
st.dataframe(tab, width="stretch")

st.subheader("Was it right? Incidents by truth")
faults = recs[recs["is_fault"]]
calm = recs[~recs["is_fault"]]
c = st.columns(3)
c[0].metric("Real-fault incidents sent on", int(faults["sent"].sum()), help="incidents whose alerts come from a real fault")
c[1].metric("Real-fault incidents held back", int((~faults["sent"]).sum()), help="kept as watch or noise")
c[2].metric("Harmless incidents sent on anyway", int(calm["sent"].sum()), help="controls: nothing was broken")
held = faults[~faults["sent"]]
if len(held):
    t = held.groupby(["truth", "triage"]).size().unstack(fill_value=0)
    t.index = t.index.str.replace("_", " ")
    t.index.name, t.columns.name = "real cause held back", None
    st.dataframe(t, width="stretch")

note("Every alert ends in exactly one incident, so nothing is thrown away: *watch* and *noise* incidents are logged "
     "and can be re-opened.")
if fam in ("all", "model"):
    finding("**Every data-drift incident stays in *watch*.** Triage's rules file drift as \"check before retraining\" "
            "and only its LLM layer can send it on. With the LLM off, deck scenario 3 never reaches Diagnosis.")
if fam in ("all", "deploy"):
    finding(f"In the bad-deploy scenario, {d['kpis'].get('deploy', {}).get('missed_by_triage', 0)} real faults raised "
            "alerts but were kept as watch / noise: mostly short or weak episodes.")
