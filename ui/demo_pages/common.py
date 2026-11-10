"""Shared data and widgets of the demo app: one evaluation run, read once, used by every page."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.evaluation import paths                        # noqa: E402
from agents.evaluation.trace import load as load_trace     # noqa: E402

RUNS = {"full": ("Full run", "2 deploy simulations + CI + model + logs (~20 s)", paths.RUN_DIR),
        "quick": ("Quick run", "1 deploy simulation + CI + model + logs (~10 s)", paths.DEMO_DIR)}

FAMILIES = {
    "ci": {"scenario": "Scenario 1", "name": "Flaky test in CI", "short": "Flaky test"},
    "deploy": {"scenario": "Scenario 2", "name": "Bad deployment", "short": "Bad deploy"},
    "model": {"scenario": "Scenario 3", "name": "Data drift on the ML model", "short": "Data drift"},
    "logs": {"scenario": "Scenario 4", "name": "Unknown failure (system logs)", "short": "Unknown failure"},
}
ORDER = ["ci", "deploy", "model", "logs"]

CAUSES = {
    "flaky_test": ("A test that fails at random; the code is fine", "re-run it once, alone"),
    "code_regression": ("A commit really broke the build", "tell the commit's author"),
    "bad_deploy": ("A new release made the service slower or failing", "roll the release back, after a human approves"),
    "config_change": ("A config push (no new code) broke the service", "roll the config back, after a human approves"),
    "infra": ("Machines, network or storage are failing: nothing we shipped", "tell the on-call engineer"),
    "data_drift": ("The model's inputs slowly changed", "retrain a new model; a human approves its deploy"),
    "data_quality": ("The data itself is broken (missing hours, a burst of odd values)", "tell a person"),
    "unknown": ("Nothing is broken (a control: harmless oddity, recovery, known false alarm)", "page nobody"),
}


def run_key() -> str:
    return st.session_state.get("run", "full")


def run_dir() -> Path:
    return RUNS[run_key()][2]


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> dict:
    d = Path(path)
    res = json.loads((d / paths.RESULTS).read_text(encoding="utf-8"))
    recs = pd.DataFrame(res["records"])
    recs["start"] = pd.to_datetime(recs["start"], utc=True, format="ISO8601")
    recs["finished_at"] = pd.to_datetime(recs["finished_at"], utc=True, format="ISO8601")
    inj = pd.DataFrame(res["injections"])
    for c in ("fault_start", "fault_end"):
        inj[c] = pd.to_datetime(inj[c], utc=True, format="ISO8601")
    inj["n_alerts"] = inj["alert_ids"].map(len)
    events = [e.to_dict() for e in load_trace(d / paths.TRACE)]
    return {"res": res, "recs": recs, "inj": inj, "events": events}


def data() -> dict | None:
    """The current run, or None when it has never been run (cached until the run changes)."""
    d = run_dir()
    f = d / paths.RESULTS
    if not f.exists():
        return None
    out = _load(str(d), f.stat().st_mtime)
    scored = out["recs"][out["recs"]["scored"]]
    return {**out, "kpis": out["res"]["kpis"], "scored": scored, "scored_inj": out["inj"][out["inj"]["scored"]]}


def run_now(key: str | None = None) -> None:
    """Run the evaluation from the app: inject, run the 5 agents, grade."""
    from agents.evaluation.config import EvalConfig
    from agents.evaluation.run import evaluate

    key = key or run_key()
    cfg = EvalConfig()
    if key == "quick":
        cfg.seeds = (4,)
    with st.status(f"Running the {RUNS[key][0].lower()}: injecting failures and running the 5 agents...",
                   expanded=True) as s:
        st.write("1. Breaking the pipeline on purpose (with the right answers written down)")
        st.write("2. Monitor → Triage → Diagnosis → Decision → simulated human → Action")
        evaluate(cfg, RUNS[key][2], verbose=False)
        st.write("3. Grading every incident against the truth")
        s.update(label="Done: every page now shows this run.", state="complete", expanded=False)
    st.cache_data.clear()


def need_data() -> dict:
    """Stop the page with a run button when there is nothing to show yet."""
    d = data()
    if d is None:
        st.warning(f"No results yet for the **{RUNS[run_key()][0].lower()}**. Run it once (it takes a few seconds).")
        if st.button("Run it now", type="primary"):
            run_now()
            st.rerun()
        st.stop()
    return d


def family_picker(key: str, allow_all: bool = True) -> str:
    opts = (["all"] if allow_all else []) + ORDER
    label = lambda f: "All four scenarios" if f == "all" else f"{FAMILIES[f]['scenario']} · {FAMILIES[f]['name']}"
    return st.segmented_control("Scenario", opts, format_func=label, default=opts[0], key=key) or opts[0]


def fmt(v, unit: str = "") -> str:
    return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v}{unit}"


def explain(what: str, how: str | None = None, measure: str | None = None) -> None:
    """The three boxes every step page starts with."""
    st.markdown(f"#### What this step does\n{what}")
    if how:
        with st.expander("How it works"):
            st.markdown(how)
    if measure:
        with st.expander("How we measure it"):
            st.markdown(measure)


def note(text: str) -> None:
    st.info(text, icon="💡")


def finding(text: str) -> None:
    st.warning(text, icon="🔎")
