"""Start here: the project, what we run today, the scenarios and the failures we inject."""
import pandas as pd
import streamlit as st

from demo_pages.common import CAUSES, FAMILIES, ORDER, RUNS, data, run_key, run_now

st.title("DeepVision: an autonomous MLOps pipeline you can trust")
st.markdown("##### Five AI agents that notice a problem, find its cause, decide what to do, fix it and check the fix: "
            "**and that ask a human before anything risky.**")

st.header("1. The problem")
c = st.columns(3)
c[0].metric("Test failures at Google that involve a flaky test", "84 %", help="Google Testing Blog, 2016")
c[1].metric("Engineers who need 1 day or more to find and fix an issue", "84 %", help="Catchpoint SRE Report 2025")
c[2].metric("SRE work that is repetitive toil", "30 %", help="Catchpoint SRE Report 2025")
st.markdown(
    "Today's pipelines follow **rules, not situations**. When a test is flaky, a release is bad or the data drifts, "
    "**a human must notice, diagnose and fix it by hand**. Our two users:\n"
    "- **Yassine, on-call SRE**: too many false alerts, slow manual diagnosis, *\"show me why before it acts\"*.\n"
    "- **Mehdi, ML engineer**: finds out about bad models from customers, can't tell which feature drifted.\n\n"
    "Our principle: **trust before autonomy**. No action without a visible reason, and a human gate on risky ones.")

st.header("2. The system: five agents in one chain")
st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#52514e" fillcolor="#f4f3ef"];
  edge [color="#52514e" fontname="Helvetica" fontsize=9];
  M [label="1 · Monitor\\ndetects anomalies"]; T [label="2 · Triage\\nmerges alerts,\\ncuts the noise"];
  D [label="3 · Diagnosis\\nfinds the cause\\n(evidence + classifier)"]; P [label="4 · Decision\\nrisk x confidence\\npolicy table"];
  H [label="5 · Human\\napproves risky actions" fillcolor="#fde9d9"]; A [label="6 · Action\\nfixes, then verifies"];
  M -> T -> D -> P; P -> A [label="safe + sure"]; P -> H [label="risky / unsure"]; H -> A [label="approved"];
}
""")
st.caption("Every agent writes its input, its reason, its action and its result: you can follow any incident from "
           "the first alarm to the verified fix (page *Follow one incident*).")

st.header("3. What we run today")
st.markdown(
    "We cannot wait for real outages, so **we break the pipeline on purpose, and we write the right answer down "
    "before we start**. Then the real agents run, and we grade them like an exam:\n\n"
    "1. **Inject** failures of the deck's four scenarios (plus *controls*: alerts where nothing is broken, to check "
    "the system stays calm).\n"
    "2. **Run the five real agents** on them, exactly as in production (the team's LangGraph pipeline).\n"
    "3. A **simulated human** approves every risky action after 5 minutes (the approval page's role).\n"
    "4. The Action agent fixes in a **simulated world that knows the truth**: rolling back the wrong release does "
    "not help, so a wrong fix shows up as a failure.\n"
    "5. **Grade** every step against the truth: the deck's KPIs (MTTD, noise reduction, accuracy, false actions, MTTR).")

st.subheader("The four scenarios")
cards = {
    "ci": ("A CI build turns red after a commit.", "TravisTorrent CI builds",
           "flaky tests · real bugs · CI machines down", "Flaky → re-run once, alone. Real bug → tell the author."),
    "deploy": ("Errors or slowness right after a new release.", "Minute metrics of a service (P1's simulator)",
               "bad releases (slow, errors, memory leak) · bad config pushes · server overload",
               "Bad release → rollback, a human approves first. Overload → on-call."),
    "model": ("The taxi model's inputs change.", "NYC Taxi features",
              "slow drift of a feature · broken data (missing hours, bursts)",
              "Drift → retrain, a human approves the deploy. Broken data → a person."),
    "logs": ("A log pattern never seen before.", "Loghub HDFS storage logs",
             "datanode / replication failures · known false alarms · normal blocks",
             "Never act alone: escalate with the evidence, or close if nothing is broken."),
}
cols = st.columns(4)
for col, f in zip(cols, ORDER):
    trig, data_src, broken, right = cards[f]
    with col.container(border=True):
        st.markdown(f"**{FAMILIES[f]['scenario']}**  \n### {FAMILIES[f]['name']}")
        st.markdown(f"**Trigger:** {trig}\n\n**Data:** {data_src}\n\n**We inject:** {broken}\n\n**Right answer:** {right}")

st.subheader("Exactly what we inject")
d = data()
if d is None:
    st.info("Run the demo once (button below) to see the exact list of injected failures.")
else:
    inj = d["scored_inj"]
    rows = []
    for f in ORDER:
        sub = inj[inj["family"] == f]
        for truth, g in sub.groupby("truth"):
            what, right = CAUSES.get(truth, ("", ""))
            rows.append({"Scenario": f"{FAMILIES[f]['scenario']} · {FAMILIES[f]['short']}",
                         "Injected cause": truth.replace("_", " "),
                         "What it is": what, "How many": len(g),
                         "Real fault?": "yes" if g["is_fault"].iloc[0] else "no (control)",
                         "Right reaction": right})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=36 * (len(rows) + 1) + 4,
                 column_config={"What it is": st.column_config.TextColumn(width="large"),
                                "Right reaction": st.column_config.TextColumn(width="large"),
                                "How many": st.column_config.NumberColumn(width="small"),
                                "Real fault?": st.column_config.TextColumn(width="small")})
    n_f = int(inj["is_fault"].sum())
    st.caption(f"{n_f} real faults and {len(inj) - n_f} controls, graded. Only cases the Diagnosis agent never "
               "trained on are graded (the others still run, because incidents are judged together).")

with st.expander("The rules of the test (be honest about them)"):
    st.markdown(
        "- **Bad deploy** goes through P1's **real detector**: we know the minute each fault started, so we measure "
        "time-to-detect. The other scenarios' alerts are created directly in the Monitor's format.\n"
        "- The **human is simulated**: they approve every risky order after 5 minutes.\n"
        "- **Time is virtual**: waiting 15 minutes to verify a fix takes no real time.\n"
        "- Triage's **LLM layer is off** (no Ollama on this machine): Triage uses its rules only.\n"
        "- These numbers describe our injected failures, **not production**.")

st.header("4. Run it")
st.markdown(f"Current run: **{RUNS[run_key()][0]}** ({RUNS[run_key()][1]}). Change it in the sidebar.")
if st.button("▶ Run the demo now", type="primary"):
    run_now()
    st.rerun()
if d is not None:
    st.success("Results are ready. Walk through the steps in the sidebar: **1 · Monitor** first.")
