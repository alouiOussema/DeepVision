"""Human approval page (P3): read the Diagnosis agent's explanation, then approve, correct or reject.

    python -m agents.diagnosis.demo                  # fills the inbox (Scenario 2, bad deployment)
    python -m streamlit run ui/approval_app.py

Cases P3 is unsure about (needs_human) come first. Every review is logged, and approve / correct
are written back to the knowledge base: the agent recognises the next similar incident.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere

from agents.diagnosis import paths                               # noqa: E402
from agents.diagnosis.feedback import accuracy_report, load_inbox, record_review, reviewed_ids  # noqa: E402
from agents.diagnosis.knowledge import KnowledgeBase              # noqa: E402
from common.schemas import Category                               # noqa: E402

st.set_page_config(page_title="Diagnosis approval", layout="wide")
ACTION_COLOR = {"rollback": "orange", "rerun": "blue", "retrain": "violet", "escalate": "red", "none": "green"}


@st.cache_resource
def knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(embed_model=None)


def load_demo():
    from agents.diagnosis.agent import DiagnosisAgent
    from agents.diagnosis.classifier import RootCauseClassifier
    from agents.diagnosis.feedback import save_inbox
    from agents.diagnosis.scenarios import deploy_cases
    from common.schemas import Decision

    cases = [c for c in deploy_cases(4, "test") if c.incident.decision == Decision.INCIDENT]
    agent = DiagnosisAgent(kb=knowledge_base(), classifier=RootCauseClassifier.load())
    diags = agent.run([c.incident for c in cases], cases[0].ctx if cases else None)
    save_inbox([(c.incident, d) for c, d in zip(cases, diags)])


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Diagnosis approval")
st.sidebar.caption("P3 explains, a human decides when P3 is unsure.")
if st.sidebar.button("Load Scenario 2 (bad deployment)", help="Simulation -> Monitor -> Triage -> Diagnosis"):
    with st.spinner("Running the pipeline (about 15 s)..."):
        load_demo()
items = load_inbox()
done = reviewed_ids()
show_done = st.sidebar.checkbox("Show reviewed diagnoses", value=False)
only_human = st.sidebar.checkbox("Only cases that need a human", value=False)
todo = [(i, d) for i, d in items if (show_done or d.id not in done) and (d.needs_human or not only_human)]
todo.sort(key=lambda x: (not x[1].needs_human, x[1].confidence))
st.sidebar.metric("Waiting for review", sum(d.id not in done for _, d in items))
st.sidebar.metric("Need a human", sum(d.needs_human and d.id not in done for _, d in items))
report = accuracy_report()
if report:
    st.sidebar.subheader("What reviewers said")
    st.sidebar.dataframe(pd.DataFrame(report).T, width="stretch")

# ------------------------------------------------------------------ main
st.title("Root-cause diagnoses")
if not items:
    st.info("The inbox is empty. Click **Load Scenario 2** in the sidebar, or run `python -m agents.diagnosis.demo`.")
for inc, d in todo:
    name = d.category.value + (f" ({d.sub_cause})" if d.sub_cause else "")
    flag = "NEEDS A HUMAN" if d.needs_human else "can act alone"
    with st.expander(f"{inc.id} - {name} - confidence {d.confidence:.2f} - {flag}", expanded=d.needs_human):
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**Incident** ({inc.source} / {inc.dataset}, Triage severity *{inc.severity.value}*)  \n"
                        f"{inc.summary}")
            st.markdown(f"### {name}")
            st.markdown(f"Suggested action: :{ACTION_COLOR.get(d.suggested_action, 'gray')}[**{d.suggested_action}**]"
                        + (f" &nbsp; target: `{d.suspect}`" if d.suspect else ""))
            st.progress(min(d.confidence, 1.0), text=f"confidence {d.confidence:.2f}")
            st.markdown("**Explanation**")
            st.info(d.explanation)
            for w in d.warnings:
                st.warning(w)
        with right:
            st.markdown("**Hypotheses**")
            st.bar_chart(pd.DataFrame({"probability": {h.category.value: h.probability for h in d.hypotheses}}),
                         horizontal=True, height=180)
            st.markdown(f"**Runbooks used:** {', '.join(f'`{r}`' for r in d.runbooks) or '-'}")
            for ref in d.runbooks[:1]:
                doc = knowledge_base().by_id.get(ref)
                if doc:
                    with st.popover(f"Open {ref}"):
                        st.markdown(f"#### {doc.title}\n\n{doc.text}")

        st.markdown("**Evidence** (the explanation may only cite these)")
        st.dataframe(pd.DataFrame([{"id": e.id, "fact": e.statement, "source": e.source,
                                    "supports": ", ".join(f"{k} {v:+.1f}" for k, v in e.supports.items())}
                                   for e in d.evidence]), hide_index=True, width="stretch")

        if d.id in done:
            st.success("Reviewed.")
            continue
        with st.form(f"review-{d.id}"):
            verdict = st.radio("Your verdict", ["approve", "correct", "reject"], horizontal=True,
                               captions=["the cause is right", "it is another cause", "not a real problem / no idea"])
            cats = [c.value for c in Category]
            c1, c2 = st.columns(2)
            corrected = c1.selectbox("Right cause (if correcting)", cats, index=cats.index(d.category.value))
            sub = c2.text_input("Sub-cause (optional)", value=d.sub_cause)
            note = st.text_input("Note for the next on-call (optional)")
            reviewer = st.text_input("Your name", value="operator")
            if st.form_submit_button("Submit review"):
                path = record_review(inc, d, verdict, Category(corrected) if verdict == "correct" else None,
                                     sub, note, reviewer, kb=knowledge_base())
                st.success("Saved." + (f" Added to the knowledge base: `{path.name}`" if path else ""))
                st.rerun()

st.caption(f"Inbox: {paths.INBOX} - reviews: {paths.REVIEWS}")
