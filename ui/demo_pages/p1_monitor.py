"""Step 1: the Monitor (P1) watches the pipeline and raises alerts."""
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from demo_pages.common import FAMILIES, ORDER, explain, finding, fmt, need_data, note

BLUE, RED, ORANGE = "#2a78d6", "#e34948", "#eb6834"

st.title("1 · Monitor: notice that something is wrong")
explain(
    "The Monitor watches four kinds of signals and raises an **alert** when one leaves its normal range. "
    "An alert says *what* is strange, *how much* (how many times over its limit) and *which detectors agree*. "
    "It does not say why: that is the next agents' job.",
    how=("One detector per scenario, each chosen to fit its data:\n\n"
         "| Signal | Detector | Normal = |\n|---|---|---|\n"
         "| Service metrics (bad deploy) | rolling baseline | the hour just before |\n"
         "| CI builds (flaky test) | gradient boosting | the project's history: probability the failure is flaky |\n"
         "| Model inputs (drift) | seasonal z-score + PSI | the same weekday-hour last year; PSI > 0.25 = drift |\n"
         "| Storage logs (unknown failure) | PCA on log blocks | patterns of normal blocks |\n\n"
         "Consecutive abnormal minutes are merged into **one** alert (an episode), so one bad release does not "
         "send 60 alerts."),
    measure=("- **MTTD (time to detect)**: from the minute we injected the fault to its first alert. Only for the "
             "bad-deploy scenario, the only one that goes through the real detector with a known start time.\n"
             "- **Missed by the Monitor**: injected faults that raised no alert at all."))

d = need_data()
k = d["kpis"]
c = st.columns(4)
c[0].metric("Alerts raised", k["all"]["alerts"], help="all four scenarios, graded cases")
c[1].metric("Real faults injected", k["all"]["faults_injected"])
c[2].metric("MTTD (median)", fmt(k.get("deploy", {}).get("mttd_min"), " min"), help="injected → first alert (bad deploy)")
c[3].metric("Faults missed by the Monitor", k["all"]["missed_by_monitor"], help="no alert at all")

st.subheader("Alerts per scenario")
st.dataframe(pd.DataFrame([{
    "Scenario": f"{FAMILIES[f]['scenario']} · {FAMILIES[f]['short']}", "Alerts": k[f]["alerts"],
    "Real faults": k[f]["faults_injected"], "MTTD": fmt(k[f]["mttd_min"], " min"),
    "Missed (no alert)": str(k[f]["missed_by_monitor"]) if f == "deploy" else "– (alerts made directly)"}
    for f in ORDER if f in k]), hide_index=True, width="stretch")

st.subheader("See it happen: one injected fault, on the real simulated metrics")


@st.cache_resource(show_spinner="Re-running the deploy simulation through P1's detector...")
def episode(seed: int):
    from agents.evaluation.injector import deploy_episode
    return deploy_episode(seed)


seeds = d["res"]["config"]["seeds"]
if "deploy" not in d["res"]["config"]["families"] or not seeds:
    st.info("This run has no bad-deploy simulation.")
    st.stop()
seed = seeds[0]
ep = episode(seed)
faults = [i for i in ep.injections if i.is_fault]
label = lambda i: (f"{i.fault_start:%b %d %H:%M} · {i.truth.value.replace('_', ' ')}"
                   f"{' ' + i.truth_suspect if i.truth_suspect else ''} ({i.truth_sub.replace('_', ' ')}) · "
                   f"{len(i.alert_ids)} alert(s)")
default = next((n for n, i in enumerate(faults) if i.truth_suspect == "release-064"), 0)
pick = st.selectbox(f"Injected fault (deploy simulation, seed {seed})", range(len(faults)), index=default,
                    format_func=lambda n: label(faults[n]))
inj = faults[pick]
metric = st.segmented_control("Metric", ["error_rate", "latency_ms"], default="error_rate",
                              format_func=lambda m: "Error rate" if m == "error_rate" else "Latency (ms)") or "error_rate"

lo, hi = inj.fault_start - timedelta(minutes=90), inj.fault_end + timedelta(minutes=60)
m = ep.context.metrics["deploy"]
win = m[(m["timestamp"] >= lo) & (m["timestamp"] <= hi)][["timestamp", metric]].rename(columns={metric: "value"})
win["series"] = "metric"
alerts = [a for a in ep.alerts if lo <= a.start <= hi]
pts = pd.DataFrame([{"timestamp": pd.Timestamp(a.start),
                     "value": float(win.iloc[(win["timestamp"] - pd.Timestamp(a.start)).abs().argmin()]["value"]),
                     "what": f"{a.feature} {a.ratio:.1f}x its limit", "series": "alert raised"} for a in alerts])
band = pd.DataFrame([{"start": pd.Timestamp(inj.fault_start), "end": pd.Timestamp(inj.fault_end),
                      "series": "injected fault"}])
color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                  scale=alt.Scale(domain=["metric", "injected fault", "alert raised"], range=[BLUE, RED, ORANGE]))
fmt_y = ".1%" if metric == "error_rate" else ".0f"
layers = [
    alt.Chart(band).mark_rect(opacity=0.12).encode(x="start:T", x2="end:T", color=color),
    alt.Chart(win).mark_line(strokeWidth=2).encode(
        x=alt.X("timestamp:T", title=None), y=alt.Y("value:Q", title=None, axis=alt.Axis(format=fmt_y)), color=color,
        tooltip=[alt.Tooltip("timestamp:T", format="%H:%M"), alt.Tooltip("value:Q", format=fmt_y)]),
]
if not pts.empty:
    layers.append(alt.Chart(pts).mark_point(size=120, filled=True, stroke="white", strokeWidth=2).encode(
        x="timestamp:T", y="value:Q", color=color, tooltip=[alt.Tooltip("timestamp:T", format="%H:%M"), "what:N"]))
st.altair_chart(alt.layer(*layers).properties(height=320), use_container_width=True)

first = min((a.start for a in ep.alerts if a.id in inj.alert_ids), default=None)
if first:
    st.success(f"Injected at **{inj.fault_start:%H:%M}**, first alert at **{first:%H:%M}**: detected after "
               f"**{(first - inj.fault_start).total_seconds() / 60:.0f} min**.")
else:
    st.error("This fault raised **no alert**: the Monitor missed it (typically a slow memory leak or a small "
             "slowdown that stays inside the normal band).")
note("Alerts outside the red band are either the metric coming back to normal after the fault (a *recovery*) or "
     "a harmless wiggle: the next agents must recognise them and page nobody.")

with st.expander("Every injected bad-deploy fault: detected or not, and how fast"):
    inj_df = d["scored_inj"]
    recs = d["scored"]
    rows = []
    for _, i in inj_df[(inj_df["family"] == "deploy") & inj_df["is_fault"]].iterrows():
        mine = recs[recs["injection"] == i["id"]]
        delay = (mine["start"].min() - i["fault_start"]).total_seconds() / 60 if len(mine) else None
        rows.append({"fault": i["id"], "cause": i["truth"], "release / config": i["truth_suspect"] or "–",
                     "injected": i["fault_start"].strftime("%b %d %H:%M"), "alerts": i["n_alerts"],
                     "detected after": fmt(None if delay is None else round(delay), " min")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

finding(f"{k['all']['missed_by_monitor']} injected faults raised no alert at all. They are mostly slow memory leaks "
        "and small slowdowns: P1's own measurement says the detector catches about 62 % of bad deploys.")
