"""Every limit of the Evaluation layer in one place."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    seeds: tuple[int, ...] = (4, 5)
    families: tuple[str, ...] = ("deploy", "ci", "logs", "model")

    human_delay_min: float = 5.0
    human_name: str = "simulated-human"

    match_before_min: float = 5.0
    match_after_min: float = 20.0
    recovery_margin_min: float = 5.0

    harm: tuple[float, float] = (0.05, 60.0)
    infra_ambient: tuple[float, float] = (0.04, 50.0)
    champion_mae: float = 2.10
    better_challenger_mae: float = 1.70
    worse_challenger_mae: float = 2.30

    deck_targets: dict = field(default_factory=lambda: {
        "wrong_auto_actions": 0,
        "silenced_faults": 0,
    })
