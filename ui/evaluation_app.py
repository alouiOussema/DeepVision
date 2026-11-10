"""Evaluation dashboard (read-only).

    python -m streamlit run ui/evaluation_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.evaluation import paths                             # noqa: E402
from agents.evaluation.report import verdict                    # noqa: E402
from agents.evaluation.score import ROWS, Record                # noqa: E402
from agents.evaluation.trace import load, timeline              # noqa: E402

st.set_page_config(page_title="System evaluation", layout="wide")

st.sidebar.title("System evaluation")
st.sidebar.caption("We break the pipeline on purpose, run all five agents, and grade them against the truth.")
use_demo = st.sidebar.checkbox("Demo data", value=False, help="python -m agents.evaluation.run --demo")
run_dir = paths.DEMO_DIR if use_demo else paths.RUN_DIR
results_path = run_dir / paths.RESULTS

if not results_path.exists():
    st.title("System evaluation")
    st.info(f"No evaluation yet in `{run_dir}`. Run `python -m agents.evaluation.run` "
            "(or `--demo`, then tick **Demo data**).")
    st.stop()

res = json.loads(results_path.read_text(encoding="utf-8"))
k = res["kpis"]
fams = list(k)
fam = st.sidebar.radio("Family", fams, index=fams.index("all"),
                       format_func=lambda f: {"deploy": "Bad deploy (scenario 2)", "ci": "Flaky test (scenario 1)",
                                              "model": "Data drift (scenario 3)", "logs": "Unknown failure (scenario 4)",
                                              "all": "All"}.get(f, f))
st.sidebar.caption(f"Run of {res['created']} · seeds {res['config']['seeds']} · "
                   f"human approves after {res['config']['human_delay_min']:g} min")
st.sidebar.warning("Simulation: injected failures, a simulated human, virtual time. Not production numbers.")

st.title("Does the whole system work?")
kk = k[fam]
f = lambda v, u="": "-" if v is None else f"{v}{u}"
c = st.columns(5)
c[0].metric("MTTD (median)", f(kk["mttd_min"], " min"), help="injected -> first alert (bad-deploy family only)")
c[1].metric("Noise reduction", f(kk["noise_reduction_pct"], " %"), help="1 - incidents sent / alerts in")
c[2].metric("Diagnosis accuracy", f(kk["diagnosis_accuracy_pct"], " %"), help="right cause (and right release)")
c[3].metric("False action rate", f(kk["false_action_pct"], " %"),
            help="fixes run on the wrong cause / release, or failed / reverted")
c[4].metric("MTTR (median)", f(kk["mttr_min"], " min"), help="injected (or detected) -> verified fix")
c = st.columns(5)
c[0].metric("Faults injected", kk["faults_injected"])
c[1].metric("Fixable faults resolved", f(kk["resolved_pct"], " %"))
c[2].metric("... with no human", f(kk["auto_resolved_pct"], " %"))
c[3].metric("Wrong automatic actions", kk["wrong_auto_actions"], help="deck scenario 4 target: 0")
c[4].metric("Silenced faults", kk["silenced_faults"], help="'nothing to do' on a real fault; target: 0")

tab_kpi, tab_inc, tab_funnel = st.tabs(["KPIs per family", "Incidents", "Where faults get lost"])

with tab_kpi:
    table = pd.DataFrame({x: {label: f(k[x][key], unit) for key, label, unit in ROWS} for x in fams})
    st.dataframe(table, width="stretch")

recs = [r for r in res["records"] if r["scored"] and (fam == "all" or r["family"] == fam)]

with tab_inc:
    show = st.radio("Show", ["sent to Diagnosis", "wrong or missed", "all"], horizontal=True)
    if show == "sent to Diagnosis":
        recs_shown = [r for r in recs if r["sent"]]
    elif show == "wrong or missed":
        recs_shown = [r for r in recs if r["silenced"] or r["wrong_fix"] or r["closed_duplicate"]
                      or r["diagnosis_right"] is False or (r["is_fault"] and not r["sent"])]
    else:
        recs_shown = recs
    df = pd.DataFrame([{"incident": r["key"], "truth": r["truth"] + (f" ({r['truth_suspect']})" if r["truth_suspect"] else ""),
                        "diagnosis": r["category"] + (f" ({r['suspect']})" if r["suspect"] else ""),
                        "decision": f"{r['route']}: {r['action']}" if r["action"] else "-",
                        "outcome": r["outcome"] or r["status"] or r["triage"],
                        "verdict": verdict(Record(**{**r, "start": pd.Timestamp(r["start"]).to_pydatetime(),
                                                     "finished_at": None}))} for r in recs_shown])
    if df.empty:
        st.info("Nothing to show for this filter.")
    else:
        st.dataframe(df, hide_index=True, width="stretch")
        pick = st.selectbox("Open an incident", df["incident"].tolist())
        events = load(run_dir / paths.TRACE)
        st.subheader("Timeline")
        st.dataframe(pd.DataFrame([{"time": e.at.strftime("%Y-%m-%d %H:%M") if e.at else "-", "agent": e.agent,
                                    "did": e.action, "result": e.result} for e in timeline(events, pick)]),
                     hide_index=True, width="stretch")
        ep, iid = pick.split(":", 1)
        page = run_dir / paths.REPORTS / ep / f"{iid}.md"
        if page.exists():
            md = page.read_text(encoding="utf-8")
            with st.expander("Incident report (auto-drafted)"):
                st.markdown(md)
            st.download_button("Download the report", md, file_name=f"{ep}_{iid}.md")

with tab_funnel:
    st.caption("Every injected fault, followed down the chain: where does it stop?")
    st.dataframe(pd.DataFrame([
        {"step": "faults injected", "count": kk["faults_injected"]},
        {"step": "missed by the Monitor (no alert)", "count": kk["missed_by_monitor"]},
        {"step": "kept as watch / noise by Triage", "count": kk["missed_by_triage"]},
        {"step": "closed as a duplicate by Decision", "count": kk["closed_as_duplicate"]},
        {"step": "silenced by Decision", "count": kk["silenced_faults"]},
        {"step": "wrong fix run with no human", "count": kk["wrong_auto_actions"]},
    ]), hide_index=True, width="stretch")
