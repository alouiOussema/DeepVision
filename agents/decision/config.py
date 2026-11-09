"""All Decision limits in one place.

The confidence bars per cause are NOT here: they come from knowledge_base/action_risk_table.md
(policy.py), the table P3 and P4 agreed on. Only what the table does not say is set here.
"""
from dataclasses import dataclass


@dataclass
class DecisionConfig:
    # ---------------- the floor (slide 11: "< 0.5 -> escalate") ----------------
    escalate_below: float = 0.50        # confidence under this: no action is even proposed, a human takes over

    # ---------------- rollback ----------------
    # Team decision (Nov 2026): a rollback always needs a human, as in the deck (scenario 2).
    # True = follow the table instead: act alone when confidence >= the table's bar (0.80).
    auto_rollback: bool = False

    # ---------------- retrain ----------------
    # The table says "day time only". Hours are read on the clock given to the agent (UTC by default).
    day_start_hour: int = 7
    day_end_hour: int = 19              # exclusive: 19:00 is already night
    retrain_human_before_deploy: bool = True    # the new model never replaces the live one without a human

    # ---------------- no-fault (action "none") ----------------
    # "recovery" / "known false alarm" recognised by Diagnosis: nobody is paged, so be sure.
    no_fault_min_confidence: float = 0.70

    # ---------------- guards that remember (use the decision history) ----------------
    rerun_budget: int = 1               # a flaky test is re-run once per incident; failing again = a real bug
    cooldown_min: float = 60.0          # the same target (release / config / job) is not acted on twice within this
    max_auto_per_hour: int = 5          # circuit breaker: more automatic actions than this in 1 h -> ask a human
    history_window_h: float = 24.0      # the rerun budget counts what was granted in this window
    batch_dedupe: bool = True           # two diagnoses blaming the same release in one batch -> one action

    # ---------------- switches ----------------
    enabled: bool = True                # False = kill switch: nothing is ever sent alone, everything waits for a human
    dry_run: bool = False               # True = decisions are made and logged, but nothing is "approved" automatically

    # ---------------- what P5 must respect (slide 11: "verify after acting, max 2 retries, revert if worse") ----------------
    max_retries: int = 2
    verify_after_min: int = 15          # check the metrics this long after acting (bad_deploy runbook: 15 min)
    revert_if_worse: bool = True
