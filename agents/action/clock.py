"""Time, injected: the agent waits "15 minutes" before verifying, tests and simulations must not.

    RealClock   the wall clock; wait() really sleeps (real execution)
    SimClock    virtual time; wait() moves the clock forward instantly (simulation, tests, demo)
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...
    def wait(self, minutes: float) -> None: ...


class RealClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def wait(self, minutes: float) -> None:
        time.sleep(max(minutes, 0) * 60)


class SimClock:
    def __init__(self, start: datetime | None = None):
        self._now = start or datetime.now(timezone.utc)

    def now(self) -> datetime:
        return self._now

    def wait(self, minutes: float) -> None:
        self._now += timedelta(minutes=max(minutes, 0))
