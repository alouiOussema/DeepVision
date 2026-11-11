"""The pipeline DeepVision operates, and where its data comes from (reads demo_data/, works offline)."""
import altair as alt
import pandas as pd
import streamlit as st

from demo_pages.common import ROOT

DATA = ROOT / "demo_data"
BLUE, RED, ORANGE, GREY = "#2a78d6", "#e34948", "#eb6834", "#8a8984"


@st.cache_data(show_spinner=False)
def read(name: str, dates: tuple = ()) -> pd.DataFrame:
    df = pd.read_csv(DATA / name)
    for c in dates:
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601", errors="coerce")
    return df


def source_box(origin: str, link: str | None, did: str, kind: str) -> None:
    c = st.columns([1, 2])
    badge = {"real": ":green-background[REAL PUBLIC DATA]", "format": ":blue-background[GENERATED IN THE DATASET'S FORMAT]",
             "sim": ":orange-background[SIMULATED]"}[kind]
    c[0].markdown(f"{badge}\n\n**Source:** {origin}" + (f"\n\n[{link}]({link})" if link else ""))
    c[1].markdown(f"**What we did with it:** {did}")


st.title("The pipeline we operate, and its data")
st.markdown("DeepVision does not run a business. It is the **operations team**: it watches an MLOps pipeline and "
            "keeps it healthy. Before the agents, here is that pipeline and where every piece of its data comes from.")

c = st.columns(3)
with c[0].container(border=True):
    st.markdown("#### 1 · Public data we brought in")
    st.markdown("NYC taxi trips, CI build history, storage logs. **Downloaded from public datasets**, not made by us.")
with c[1].container(border=True):
    st.markdown("#### 2 · The pipeline it stands for")
    st.markdown("A **taxi-fare model** with its CI, its web service and its storage. Built by the team on top of "
                "layer 1, plus a simulated web service.")
with c[2].container(border=True):
    st.markdown("#### 3 · Failures we injected")
    st.markdown("Bad releases, flaky tests, drift, storage failures, **each with its right answer**. Added by us to "
                "test the agents: **not in the original data**.")

st.info("**The project:** a model that predicts the **average taxi fare of the coming hour** in New York, from the "
        "hour, the weekday, the average distance and the number of trips. Every sample shown here is in the repo, in "
        "`demo_data/` (see its README).", icon="🚕")

if not DATA.exists():
    st.error("demo_data/ is missing. Run `python -m scripts.make_demo_data`.")
    st.stop()

tab_nyc, tab_ci, tab_logs, tab_svc, tab_inj = st.tabs(
    ["🚕 NYC Taxi (model data)", "🧪 CI builds", "🗄️ Storage logs", "🌐 Web service", "💉 What we injected"])

# ------------------------------------------------------------------ NYC taxi
with tab_nyc:
    summary = read("nyc_taxi/monthly_summary.csv")
    source_box("NYC Taxi & Limousine Commission, yellow-taxi trip records (6 monthly files, about 60 MB each)",
               "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
               f"{summary['raw_trips'].sum():,} raw trips → invalid trips and trips outside their month removed → "
               f"{summary['trips_kept'].sum():,} trips → **one row per hour** (trip count, average fare, average "
               "distance). That hourly table is what the Monitor watches and what the model learns from.", "real")

    st.markdown("##### Step 1: the raw data, as published (15 trips out of millions)")
    st.dataframe(read("nyc_taxi/raw_trips_sample.csv"), hide_index=True, width="stretch")

    st.markdown("##### Step 2: what we turned it into, one row per hour")
    hourly = read("nyc_taxi/hourly.csv", ("timestamp",))
    st.dataframe(hourly.head(48), hide_index=True, width="stretch", height=250)

    st.markdown("##### The daily rhythm the Monitor learns (one week)")
    hourly["week"] = hourly["timestamp"].dt.tz_localize(None).dt.to_period("W").dt.start_time.dt.strftime("week of %d %b %Y")
    weeks = list(dict.fromkeys(hourly["week"]))
    week = st.selectbox("Week", weeks, index=min(1, len(weeks) - 1))
    w = hourly[hourly["week"] == week]
    st.altair_chart(alt.Chart(w).mark_line(color=BLUE, strokeWidth=2).encode(
        x=alt.X("timestamp:T", title=None), y=alt.Y("trip_count:Q", title="trips per hour"),
        tooltip=[alt.Tooltip("timestamp:T", format="%a %d %b %H:00"), "trip_count:Q",
                 alt.Tooltip("avg_fare:Q", format="$.2f")]).properties(height=260), width="stretch")
    st.caption("Quiet nights, busy evenings, a different weekend: a value is only abnormal compared with the same "
               "weekday and hour. That is why the Monitor uses a seasonal detector.")

    st.markdown("##### A real drift in the real data (not injected)")
    s = summary.assign(label=pd.to_datetime(summary["month"]).dt.strftime("%b %Y"))
    bars = alt.Chart(s).mark_bar(color=BLUE, cornerRadiusEnd=4, size=46).encode(
        x=alt.X("label:N", sort=list(s["label"]), title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("avg_fare:Q", title="average fare per trip ($)", scale=alt.Scale(zero=True)),
        tooltip=["label:N", alt.Tooltip("avg_fare:Q", format="$.2f"), alt.Tooltip("trips_kept:Q", format=",")])
    text = bars.mark_text(dy=-8, color="#52514e").encode(text=alt.Text("avg_fare:Q", format="$.2f"))
    st.altair_chart((bars + text).properties(height=280), width="stretch")
    jan = summary.set_index("month")["avg_fare"]
    if {"2025-01", "2026-01"} <= set(jan.index):
        st.success(f"The average fare per trip went from **${jan['2025-01']:.2f}** (Jan 2025) to "
                   f"**${jan['2026-01']:.2f}** (Jan 2026): **{100 * (jan['2026-01'] / jan['2025-01'] - 1):+.0f} %** "
                   "year over year. A model trained on 2025 prices is now wrong: this is the data drift of scenario 3, "
                   "and it is in the real data.")
    st.caption("Months shown are the ones we downloaded (Jan–Feb 2025, Nov 2025–Feb 2026), not a continuous series.")

# ------------------------------------------------------------------ CI
with tab_ci:
    source_box("TravisTorrent: build history of open-source projects on Travis CI",
               "https://travistorrent.testroots.org",
               "jobs merged into builds; a failed build is **flaky** when the same code also passed. The full file is "
               "several GB and stays on the Monitor author's machine, so the rows below are **generated in the same "
               "format** by the evaluation, each with its true cause.", "format")
    ci = read("ci_builds/builds_example.csv")
    st.dataframe(ci, hide_index=True, width="stretch")
    st.caption("`flaky_probability` is the field the Monitor's model fills for each failed build. `true_cause` is the "
               "answer we wrote down, used to grade the agents (the agents never see it).")

# ------------------------------------------------------------------ logs
with tab_logs:
    source_box("Loghub: HDFS logs of a Hadoop cluster, labelled per block", "https://github.com/logpai/loghub",
               "each storage block becomes a count of 29 log event types; a block whose pattern looks like no normal "
               "block is an anomaly. Rows below are **generated in the same format**, with their true cause.", "format")
    st.dataframe(read("hdfs_logs/blocks_example.csv"), hide_index=True, width="stretch")
    st.caption("`events` are the log templates that stand out in the block (E1 to E29). Some patterns are real "
               "failures (`infra`), others are harmless (`unknown`): the agents must tell them apart.")

# ------------------------------------------------------------------ service
with tab_svc:
    source_box("None: no real deployment data exists, so the Monitor's author wrote a simulator", None,
               "a web service with daily traffic, latency, error rate and releases, some of them bad. We add bad "
               "config pushes and server overloads to it. **Everything on this tab is simulated.**", "sim")
    m = read("service/metrics_around_release-064.csv", ("timestamp",))
    rel = read("service/releases_and_config_pushes.csv", ("timestamp",))
    rel = rel[(rel["timestamp"] >= m["timestamp"].min()) & (rel["timestamp"] <= m["timestamp"].max())]
    line = alt.Chart(m).mark_line(color=BLUE, strokeWidth=2).encode(
        x=alt.X("timestamp:T", title=None), y=alt.Y("error_rate:Q", title="error rate", axis=alt.Axis(format=".1%")),
        tooltip=[alt.Tooltip("timestamp:T", format="%H:%M"), alt.Tooltip("error_rate:Q", format=".2%")])
    rules = alt.Chart(rel).mark_rule(color=ORANGE, strokeWidth=2, strokeDash=[4, 3]).encode(
        x="timestamp:T", tooltip=["id:N", "kind:N", alt.Tooltip("timestamp:T", format="%H:%M")])
    st.altair_chart((line + rules).properties(height=280), width="stretch")
    st.caption("Dashed orange line = a release going out (release-064, which we made bad). The error rate jumps "
               "right after it. This is the incident you follow on the Monitor page and in *Follow one incident*.")

# ------------------------------------------------------------------ injected
with tab_inj:
    st.markdown(":red-background[ADDED BY US: NOT IN THE ORIGINAL DATA]")
    st.markdown("To grade the agents we need failures **whose cause we already know**. So we inject them, write the "
                "right answer down, and never show that answer to the agents.")
    inj = read("injected/faults.csv", ("injected_at",))
    real = inj[inj["real_fault"]]
    c = st.columns(4)
    c[0].metric("Injected entries", len(inj))
    c[1].metric("Real faults", len(real))
    c[2].metric("Harmless controls", int((~inj["real_fault"]).sum()), help="nothing broken: the right answer is to page nobody")
    c[3].metric("Real faults graded", int(real["graded"].sum()), help="the others trained the Diagnosis classifier")
    t = real.groupby(["scenario", "cause"]).size().reset_index(name="how many")
    t["cause"] = t["cause"].str.replace("_", " ")
    st.dataframe(t, hide_index=True, width="stretch")
    with st.expander("The full list (every injected fault, with its right answer)"):
        st.dataframe(inj, hide_index=True, width="stretch")
