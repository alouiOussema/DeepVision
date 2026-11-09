"""All Action limits in one place.

The per-order limits (max_retries, verify_after_min, revert_if_worse, human_before_deploy, reruns, notify) come
from the ORDER (Decision sets them in `constraints`). The values here are caps and what the order does not say.
"""
from dataclasses import dataclass, field


@dataclass
class ActionConfig:
    # ---------------- the switch ----------------
    # True = every backend must be a simulation (nothing outside this repo is touched). Real execution is an
    # explicit choice: simulate=False AND real backends passed by the caller. The default is the safe one.
    simulate: bool = True

    # ---------------- retries (slide 11: "max 2 retries") ----------------
    max_retries_cap: int = 2            # even if an order asks for more, never more than this
    retry_wait_min: float = 2.0         # pause before a retry (CI API down, notification not delivered...)

    # ---------------- verification ----------------
    # If the order has no verify_after_min we do NOT guess: the order fails (contract broken).
    worse_margin: float = 0.10          # rollback: error rate / latency 10 % above the level before acting = worse

    # ---------------- retrain (champion vs challenger) ----------------
    retrain_min_gain: float = 0.02      # the challenger must cut the error by >= 2 % to be proposed for deploy
    retrain_window_days: int = 28       # challenger is trained on the last 28 days (the drift window of P1's PSI)
    retrain_holdout_days: int = 7       # both models are scored on the most recent 7 days, never seen in training

    # ---------------- who is told when something goes wrong ----------------
    # The deck: a red re-run -> the developer; a rollback that did not help -> on-call; a worse model -> ML engineer.
    escalate_to: dict = field(default_factory=lambda: {"rerun": "author", "rollback": "on-call",
                                                       "retrain": "ml-engineer", "escalate": "on-call"})
    notify_channel: str = "file"        # "file" = data/action/notifications.jsonl (the only channel this repo has)
