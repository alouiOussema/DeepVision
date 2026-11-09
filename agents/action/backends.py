"""The outside world the Action agent talks to: CI, deployments, the model registry, a notification channel.

The repo has NO CI API, NO Docker / Kubernetes and NO MLflow, so every backend here is a SIMULATION that keeps
its state in memory (or in a local file for notifications). Each one is a small class behind a Protocol: a real
adapter (GitHub Actions, kubectl / Argo, MLflow, Slack) only has to implement the same few methods and set
`touches_production = True`. The agent refuses to use such a backend unless simulation is switched off.

    CIBackend      rerun(project, build_id) -> run id          result(run id) -> "passed" | "failed"
    DeployBackend  is_live(change) / health() / rollback(change) / redeploy(change)
    ModelBackend   retrain_and_compare(...) -> Comparison       discard(challenger)      (NO deploy method, on purpose)
    Notifier       send(to, subject, body) -> message id        delivered(message id)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4


class BackendError(RuntimeError):
    """A temporary failure of an outside system (API down, timeout): the action did NOT happen, retrying is safe."""


class BackendRefused(RuntimeError):
    """The outside system says the order cannot be done as asked (unknown release, missing data): never retried."""


# ------------------------------------------------------------------ CI
class CIBackend(Protocol):
    touches_production: bool
    def rerun(self, project: str, build_id: str, now: datetime) -> str: ...
    def result(self, run_id: str, now: datetime) -> str: ...


class SimCI:
    """A CI server in memory. `outcomes[build_id]` = what successive re-runs of that build return
    (default: "passed", the flaky-test story). `api_errors` = how many calls fail before the API answers."""
    touches_production = False

    def __init__(self, outcomes: dict[str, list[str]] | None = None, default: str = "passed", api_errors: int = 0):
        self.outcomes = {k: list(v) for k, v in (outcomes or {}).items()}
        self.default, self.api_errors = default, api_errors
        self.runs: dict[str, dict] = {}

    def rerun(self, project: str, build_id: str, now: datetime) -> str:
        if self.api_errors > 0:
            self.api_errors -= 1
            raise BackendError("CI API did not answer (simulated timeout)")
        queue = self.outcomes.get(str(build_id), [])
        outcome = queue.pop(0) if queue else self.default
        run_id = f"run-{uuid4().hex[:6]}"
        self.runs[run_id] = {"project": project, "build_id": str(build_id), "outcome": outcome, "started": now}
        return run_id

    def result(self, run_id: str, now: datetime) -> str:
        return self.runs[run_id]["outcome"]


# ------------------------------------------------------------------ deployments
@dataclass(frozen=True)
class Health:
    error_rate: float          # share of failed requests
    latency_ms: float

    def as_dict(self) -> dict:
        return {"error_rate": round(self.error_rate, 4), "latency_ms": round(self.latency_ms, 1)}


@dataclass(frozen=True)
class Band:
    """The normal band of the service (bad_deploy runbook: "back in their band")."""
    error_rate_max: float = 0.02
    latency_ms_max: float = 160.0

    def contains(self, h: Health) -> bool:
        return h.error_rate <= self.error_rate_max and h.latency_ms <= self.latency_ms_max


class DeployBackend(Protocol):
    touches_production: bool
    band: Band
    def is_live(self, change_id: str) -> bool: ...
    def health(self, now: datetime) -> Health: ...
    def rollback(self, change_id: str, now: datetime) -> None: ...
    def redeploy(self, change_id: str, now: datetime) -> None: ...


class SimDeploy:
    """A service whose health is a base level plus the effect of every change still live.

    effects[change] = (extra error rate, extra latency) that change adds while it is live: a bad release adds
    errors, rolling it back removes them. `ambient` = trouble no change explains (infra): a rollback cannot
    remove it. A NEGATIVE effect = the change was helping (e.g. it carried a hotfix): undoing it makes things worse."""
    touches_production = False

    def __init__(self, live: list[str], effects: dict[str, tuple[float, float]] | None = None,
                 base: Health = Health(0.01, 120.0), ambient: tuple[float, float] = (0.0, 0.0),
                 band: Band = Band(), api_errors: int = 0):
        self.live = list(live)
        self.effects = dict(effects or {})
        self.base, self.ambient, self.band, self.api_errors = base, ambient, band, api_errors
        self.history: list[tuple[datetime, str, str]] = []

    @classmethod
    def from_changes(cls, change_ids: list[str], bad: str | None = None) -> "SimDeploy":
        """From the pipeline's change log (P3's ContextStore): every known change is live, `bad` hurts."""
        return cls(change_ids, {bad: (0.05, 60.0)} if bad else {})

    def _call(self) -> None:
        if self.api_errors > 0:
            self.api_errors -= 1
            raise BackendError("deploy API did not answer (simulated timeout)")

    def is_live(self, change_id: str) -> bool:
        return change_id in self.live

    def health(self, now: datetime) -> Health:
        err = self.base.error_rate + self.ambient[0] + sum(self.effects.get(c, (0, 0))[0] for c in self.live)
        lat = self.base.latency_ms + self.ambient[1] + sum(self.effects.get(c, (0, 0))[1] for c in self.live)
        return Health(max(err, 0.0), max(lat, 0.0))

    def rollback(self, change_id: str, now: datetime) -> None:
        self._call()
        if change_id not in self.live:
            raise BackendRefused(f"{change_id} is not live: nothing to roll back")
        self.live.remove(change_id)
        self.history.append((now, "rollback", change_id))

    def redeploy(self, change_id: str, now: datetime) -> None:
        self._call()
        if change_id not in self.live:
            self.live.append(change_id)
        self.history.append((now, "redeploy", change_id))


# ------------------------------------------------------------------ model registry
@dataclass
class Comparison:
    """Champion (live model) vs challenger (just retrained), scored on the SAME recent data neither was trained on."""
    metric: str                 # e.g. "mae_avg_fare"
    champion: float             # error of the live model (lower is better)
    challenger: float
    challenger_id: str
    n_holdout: int = 0
    notes: str = ""

    @property
    def gain(self) -> float:
        """Relative error reduction of the challenger: 0.10 = 10 % fewer errors."""
        return (self.champion - self.challenger) / self.champion if self.champion else 0.0

    def as_dict(self) -> dict:
        return {"metric": self.metric, "champion": round(self.champion, 4), "challenger": round(self.challenger, 4),
                "gain": round(self.gain, 4), "challenger_id": self.challenger_id, "n_holdout": self.n_holdout,
                "notes": self.notes}


class ModelBackend(Protocol):
    """No `deploy` here on purpose: P5 cannot replace the live model, whatever the order says."""
    touches_production: bool
    def retrain_and_compare(self, order_id: str, now: datetime, window_days: int, holdout_days: int) -> Comparison: ...
    def discard(self, challenger_id: str) -> None: ...


class SimModel:
    """A model registry in memory: the comparison numbers are given (a scenario decides if the challenger wins)."""
    touches_production = False

    def __init__(self, champion_mae: float = 2.10, challenger_mae: float = 1.70, errors: int = 0):
        self.champion_mae, self.challenger_mae, self.errors = champion_mae, challenger_mae, errors
        self.challengers: dict[str, Comparison] = {}
        self.live = "champion"                      # never changed by this class

    def retrain_and_compare(self, order_id: str, now: datetime, window_days: int, holdout_days: int) -> Comparison:
        if self.errors > 0:
            self.errors -= 1
            raise BackendError("training job crashed (simulated)")
        c = Comparison("mae_avg_fare", self.champion_mae, self.challenger_mae, f"challenger-{order_id}",
                       n_holdout=holdout_days * 24, notes="simulated scores")
        self.challengers[c.challenger_id] = c
        return c

    def discard(self, challenger_id: str) -> None:
        self.challengers.pop(challenger_id, None)


# ------------------------------------------------------------------ notifications
class Notifier(Protocol):
    touches_production: bool
    def send(self, to: str, subject: str, body: str, now: datetime, meta: dict | None = None) -> str: ...
    def delivered(self, message_id: str) -> bool: ...


class FileNotifier:
    """The only channel this repo has: one JSON line per message in data/action/notifications.jsonl.
    A Slack / e-mail adapter would implement the same two methods."""
    touches_production = False

    def __init__(self, path: Path):
        self.path = Path(path)

    def send(self, to: str, subject: str, body: str, now: datetime, meta: dict | None = None) -> str:
        mid = f"msg-{uuid4().hex[:8]}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": mid, "at": now.isoformat(timespec="seconds"), "to": to, "subject": subject,
                                "body": body, **(meta or {})}, default=str) + "\n")
        return mid

    def delivered(self, message_id: str) -> bool:
        return self.path.exists() and any(json.loads(line).get("id") == message_id
                                          for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())

    def messages(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------------ the bundle
@dataclass
class Backends:
    ci: CIBackend = field(default_factory=SimCI)
    deploy: DeployBackend = field(default_factory=lambda: SimDeploy([]))
    model: ModelBackend = field(default_factory=SimModel)
    notifier: Notifier | None = None           # set by the agent (needs a path)

    def all(self) -> dict[str, object]:
        return {"ci": self.ci, "deploy": self.deploy, "model": self.model, "notifier": self.notifier}

    def touching_production(self) -> list[str]:
        return [name for name, b in self.all().items() if b is not None and getattr(b, "touches_production", True)]
