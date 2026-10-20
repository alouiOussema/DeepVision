"""The Monitor's single entry point for the rest of the team (P4's orchestrator).

    from agents.monitor.node import monitor_node
    graph.add_node("monitor", monitor_node)          # LangGraph: a node is just a function

Input  state: {"signals": {"nyc": <hourly DataFrame>, ...},        # time series (CI, metrics, model inputs)
               "blocks":  (block_ids, event_count_matrix),         # log blocks (optional)
               "builds":  (builds_table, since_timestamp)}         # CI builds with history (optional)
Output       : {"alerts": [ {id, timestamp, source, signal, score, kind, dataset, details}, ... ]}
The first five keys are the team's Alert contract. The rest is extra evidence for Triage/Diagnosis.
Run scripts/02_train.py once first: it creates all the trained models.
"""
import pickle
from functools import lru_cache
from config import settings
from .series_monitor import SeriesMonitor
from .log_monitor import LogMonitor
from .ci_monitor import CIMonitor


@lru_cache(maxsize=1)
def _series_models():
    with open(settings.MODELS_DIR / "monitor_state.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _log_model():
    with open(settings.MODELS_DIR / "hdfs_block_pca.pkl", "rb") as f:
        m = pickle.load(f)
    return m["detector"], m["events"]


@lru_cache(maxsize=1)
def _flaky_model():
    with open(settings.MODELS_DIR / "flaky_model.pkl", "rb") as f:
        return pickle.load(f)


def detect(signals=None, blocks=None, builds=None):
    """Returns a list of Alert objects."""
    alerts = []
    for dataset, df in (signals or {}).items():
        bundle = _series_models().get(dataset)
        if bundle:
            alerts += list(SeriesMonitor(bundle, dataset=dataset, min_votes=settings.MIN_VOTES,
                                         max_gap_hours=settings.MERGE_GAP_HOURS[dataset]).run(df))
    if blocks is not None:
        ids, X = blocks
        det, events = _log_model()
        alerts += list(LogMonitor(det, events, dataset="hdfs",
                                  max_alerts=settings.LOG_MAX_ALERTS).run(ids, X))
    if builds is not None:
        df, since = builds
        alerts += list(CIMonitor(_flaky_model()).run(df, since))
    return alerts


def monitor_node(state: dict) -> dict:
    alerts = detect(state.get("signals"), state.get("blocks"), state.get("builds"))
    return {"alerts": [
        {**a.to_contract(), "timestamp": a.timestamp.isoformat(),
         "kind": a.kind, "dataset": a.dataset, "details": a.details}
        for a in alerts]}
