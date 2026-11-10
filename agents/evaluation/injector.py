"""Failures injected on purpose, with the right answer written down, so the whole pipeline can be graded.

The deploy family goes through P1's real detector, so we know when each fault started. ci, model and logs reuse the
labelled cases of the Diagnosis benchmark; only the ones Diagnosis never trained on are graded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from common.schemas import Alert, Category

from .config import EvalConfig

NO_FAULT_SUBS = {"no_fault", "recovery", "known_false_alarm"}


@dataclass
class Injection:
    """One failure we caused on purpose (or a control where nothing is broken), with its truth."""
    id: str
    family: str
    truth: Category
    truth_sub: str = ""
    truth_suspect: str = ""
    fault_start: datetime | None = None
    fault_end: datetime | None = None
    alert_ids: list[str] = field(default_factory=list)
    origin: str = "synthetic"
    scored: bool = True  # False = Diagnosis trained on it: it runs, but is not graded

    @property
    def is_fault(self) -> bool:
        return not (self.truth is Category.UNKNOWN and self.truth_sub in NO_FAULT_SUBS)

    def to_dict(self) -> dict:
        return {"id": self.id, "family": self.family, "truth": self.truth.value, "truth_sub": self.truth_sub,
                "truth_suspect": self.truth_suspect, "is_fault": self.is_fault, "origin": self.origin,
                "fault_start": self.fault_start.isoformat() if self.fault_start else None,
                "fault_end": self.fault_end.isoformat() if self.fault_end else None, "alert_ids": self.alert_ids,
                "scored": self.scored}

    @classmethod
    def from_dict(cls, d: dict) -> "Injection":
        t = lambda x: datetime.fromisoformat(x) if x else None
        return cls(d["id"], d["family"], Category(d["truth"]), d.get("truth_sub", ""), d.get("truth_suspect", ""),
                   t(d.get("fault_start")), t(d.get("fault_end")), list(d.get("alert_ids", [])),
                   d.get("origin", "synthetic"), d.get("scored", True))


@dataclass
class Episode:
    """One batch for the graph: the alerts the Monitor emits, the context Diagnosis reads, and the truth."""
    name: str
    family: str
    seed: int | None
    alerts: list[Alert]
    context: object
    injections: list[Injection]

    def injection_of(self) -> dict[str, str]:
        """alert id -> injection id."""
        return {a: inj.id for inj in self.injections for a in inj.alert_ids}

    def injection_for(self, alert_ids: list[str]) -> Injection | None:
        """The injection most of an incident's alerts come from (ties: the real fault wins over a control)."""
        of = self.injection_of()
        votes: dict[str, int] = {}
        for a in alert_ids:
            if a in of:
                votes[of[a]] = votes.get(of[a], 0) + 1
        if not votes:
            return None
        by_id = {i.id: i for i in self.injections}
        best = max(votes, key=lambda k: (votes[k], by_id[k].is_fault))
        return by_id[best]

    @property
    def start(self) -> datetime:
        return min(a.start for a in self.alerts) if self.alerts else datetime.now().astimezone()

    def save(self, directory: Path) -> Path:
        """The truth, for P3's benchmark or anyone who wants to check our scoring (one JSON line per injection)."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for inj in self.injections:
                f.write(json.dumps(inj.to_dict()) + "\n")
        return path


def _dt(x) -> datetime:
    return pd.Timestamp(x).to_pydatetime()


def deploy_episode(seed: int, cfg: EvalConfig | None = None) -> Episode:
    """P1's simulator + our config / infra faults -> P1's rolling detector (agents/diagnosis/scenarios.py)."""
    from agents.diagnosis.context import ContextStore
    from agents.diagnosis.scenarios import simulate_deploy
    from agents.monitor.factory import slice_period
    from config import settings

    cfg = cfg or EvalConfig()
    p1_alerts, ctx, faults = simulate_deploy(seed)
    alerts = [Alert.from_dict({**a.to_row(), "details": a.details}) for a in p1_alerts]

    test = slice_period(ctx.metrics["deploy"], *settings.SPLITS["deploy"]["test"])["timestamp"]
    lo, hi = _dt(test.min()), _dt(test.max())
    faults = sorted((f for f in faults if lo <= _dt(f[0]) <= hi), key=lambda f: _dt(f[0]))

    injections = [Injection(f"INJ-D{seed}-{k:03d}", "deploy", cat, sub, suspect, _dt(t),
                            _dt(t) + timedelta(minutes=dur), origin="simulation")
                  for k, (t, dur, cat, sub, suspect) in enumerate(faults, 1)]
    controls = {sub: Injection(f"INJ-D{seed}-{sub}", "deploy", Category.UNKNOWN, sub, origin="simulation")
                for sub in ("no_fault", "recovery")}
    before, after = timedelta(minutes=cfg.match_before_min), timedelta(minutes=cfg.match_after_min)
    for a in alerts:
        hit = [i for i in injections if i.fault_start - before <= a.start <= i.fault_end + after]
        if not hit:
            controls["no_fault"].alert_ids.append(a.id)
            continue
        inj = min(hit, key=lambda i: a.start - i.fault_start)
        if a.start >= inj.fault_end - timedelta(minutes=cfg.recovery_margin_min):  # metrics coming back = recovery
            controls["recovery"].alert_ids.append(a.id)
        else:
            inj.alert_ids.append(a.id)
    clean = ContextStore(ctx.changes, ctx.metrics)  # no peers: Diagnosis adds the batch itself
    return Episode(f"deploy-seed{seed}", "deploy", seed, alerts, clean,
                   injections + [c for c in controls.values() if c.alert_ids])


def synthetic_cases() -> dict[str, list]:
    """Synthetic cases of the Diagnosis benchmark, with the same train/test split.

    Keep this in sync with agents/diagnosis/scenarios.py::build() (same seed, same order).
    """
    from agents.diagnosis.scenarios import ci_cases, log_cases, model_cases

    rng = np.random.default_rng(7)
    synth = ci_cases(rng) + log_cases(rng) + model_cases(rng)
    for c in synth:
        c.split = "test" if rng.random() < 0.4 else "train"
    out: dict[str, list] = {}
    for c in synth:
        out.setdefault(c.family, []).append(c)
    return out


def synthetic_episode(family: str, cases: list) -> Episode:
    from agents.diagnosis.context import ContextStore

    alerts, injections = [], []
    for k, c in enumerate(cases, 1):
        alerts += c.incident.alerts
        injections.append(Injection(f"INJ-{family.upper()}-{k:03d}", family, c.truth, c.truth_sub, c.truth_suspect,
                                    alert_ids=list(c.incident.alert_ids), origin=c.origin,
                                    scored=getattr(c, "split", "test") == "test"))
    alerts.sort(key=lambda a: a.start)
    return Episode(family, family, None, alerts, ContextStore(), injections)


def episodes(cfg: EvalConfig | None = None) -> list[Episode]:
    cfg = cfg or EvalConfig()
    out = [deploy_episode(s, cfg) for s in cfg.seeds] if "deploy" in cfg.families else []
    wanted = [f for f in cfg.families if f != "deploy"]
    if wanted:
        cases = synthetic_cases()
        out += [synthetic_episode(f, cases.get(f, [])) for f in wanted]
    return out
