"""DeepVision demo app.

    python -m streamlit run ui/demo_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_pages.common import RUNS, data, run_key, run_now               # noqa: E402

st.set_page_config(page_title="DeepVision demo", page_icon="🛰️", layout="wide")

PAGES = Path(__file__).resolve().parent / "demo_pages"
nav = st.navigation({
    "Start here": [st.Page(PAGES / "p0_project.py", title="The project & today's demo", icon="🎯", default=True)],
    "The pipeline, step by step": [
        st.Page(PAGES / "p1_monitor.py", title="1 · Monitor: detect", icon="📡"),
        st.Page(PAGES / "p2_triage.py", title="2 · Triage: cut the noise", icon="🧹"),
        st.Page(PAGES / "p3_diagnosis.py", title="3 · Diagnosis: find the why", icon="🔍"),
        st.Page(PAGES / "p4_decision.py", title="4 · Decision: what happens next", icon="⚖️"),
        st.Page(PAGES / "p5_human.py", title="5 · Human approval", icon="🙋"),
        st.Page(PAGES / "p6_action.py", title="6 · Action: fix and verify", icon="🛠️"),
    ],
    "Proof": [
        st.Page(PAGES / "p7_follow.py", title="Follow one incident", icon="🧭"),
        st.Page(PAGES / "p8_results.py", title="Results & what we learned", icon="📊"),
    ],
})

with st.sidebar:
    st.radio("Which run", list(RUNS), key="run", format_func=lambda k: RUNS[k][0],
             captions=[RUNS[k][1] for k in RUNS], index=0)
    d = data()
    if d:
        st.caption(f"Last run: {d['res']['created'].replace('T', ' ')[:16]} UTC")
    if st.button("Run it again" if d else "Run it now", use_container_width=True):
        run_now(run_key())
        st.rerun()
    st.caption("Simulation: injected failures, a simulated human, virtual time. No real system is touched.")

nav.run()
