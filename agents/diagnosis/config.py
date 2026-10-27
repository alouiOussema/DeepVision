"""All Diagnosis limits in one place.

Starting values chosen on the scenario benchmark (evaluate.py); the sweep there shows
how accuracy and the share of cases sent to a human move when they change.
"""
from dataclasses import dataclass, field

from common.schemas import Category

C = Category


@dataclass
class DiagnosisConfig:
    # ---------------- A. changes (deploys, config pushes) ----------------
    # A metric incident is credited to a change if it starts at most this many minutes after it.
    # 150 min: a slow leak can take more than an hour to cross the Monitor's limit.
    change_window_min: float = 150.0
    quick_effect_min: float = 20.0      # an effect visible within this delay is "right after" the change

    # ---------------- B. metric shape (minute metrics around the incident) ----------------
    before_min: int = 60                # "before" picture: this many minutes before the anchor
    after_min: int = 90                 # "after" picture: this many minutes after the anchor
    # a metric "moved" if its level changed by more than this share. The error rate is much noisier
    # than latency in the minute data (about 10 % vs 3 % on a 30-minute mean), hence two limits.
    moved_change: dict = field(default_factory=lambda: {"latency_ms": 0.10, "error_rate": 0.35})
    moved_default: float = 0.15
    ramp_limit: float = 0.5             # ramp score above this = slow climb (leak), below = step
    detect_lag_min: int = 10            # no change to anchor on: start the "after" picture this much before the alert

    # ---------------- C. ci ----------------
    flaky_high: float = 0.5             # flaky probability above this points to a flaky test
    flaky_low: float = 0.2              # below this points to a real failure
    infra_projects: int = 3             # this many projects failing in the same hour = shared CI infrastructure

    # ---------------- D. decision of the agent ----------------
    # prior belief per source, before any evidence (log-odds are added to these)
    priors: dict = field(default_factory=lambda: {
        "ci":      {C.FLAKY_TEST: 0.40, C.CODE_REGRESSION: 0.40, C.INFRA: 0.15, C.UNKNOWN: 0.05},
        "metrics": {C.BAD_DEPLOY: 0.30, C.CONFIG_CHANGE: 0.15, C.INFRA: 0.25, C.UNKNOWN: 0.30},
        "logs":    {C.INFRA: 0.60, C.UNKNOWN: 0.40},
        "model":   {C.DATA_DRIFT: 0.40, C.DATA_QUALITY: 0.35, C.UNKNOWN: 0.25},
    })
    ml_weight: float = 0.5              # final = (1 - w) * rules + w * classifier, when a classifier is trained
    min_probability: float = 0.40       # best hypothesis below this -> category UNKNOWN
    human_below: float = 0.60           # confidence below this -> needs_human
    llm_disagree_penalty: float = 0.75  # confidence x this when the LLM argues for another cause
    novelty_penalty: float = 0.70       # confidence x this when the case looks like nothing in the knowledge base

    # what P4 would usually do for each cause: a HINT, the decision is P4's
    actions: dict = field(default_factory=lambda: {
        C.FLAKY_TEST: "rerun", C.CODE_REGRESSION: "escalate", C.BAD_DEPLOY: "rollback",
        C.CONFIG_CHANGE: "rollback", C.INFRA: "escalate", C.DATA_DRIFT: "retrain",
        C.DATA_QUALITY: "escalate", C.UNKNOWN: "escalate",
    })
