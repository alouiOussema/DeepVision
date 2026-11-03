"""The action risk table, read from the knowledge base: the ONE source of the policy.

knowledge_base/action_risk_table.md is shared with P3 (Diagnosis names the cause and only SUGGESTS the
action). Editing a bar there changes the policy here, with no code change. A table we cannot read raises:
a policy that silently falls back to defaults is worse than one that refuses to start.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from common.schemas import Action, Category

from . import paths

RISKS = ("none", "low", "medium", "high")


@dataclass(frozen=True)
class PolicyRow:
    category: Category
    action: Action
    risk: str
    reversible: bool | None        # None = not applicable ("-")
    bar: float | None              # confidence needed to act alone; None = never alone
    daytime_only: bool = False
    detail: str = ""               # what is in brackets in the table, e.g. "once", "author", "config"

    @property
    def never_alone(self) -> bool:
        return self.bar is None


class PolicyError(ValueError):
    pass


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse(text: str) -> dict[Category, PolicyRow]:
    rows: dict[Category, PolicyRow] = {}
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _cells(line)
        if len(cells) < 5 or cells[0].lower() == "cause" or set(cells[0]) <= set("-: "):
            continue                                       # header and separator
        cause, action, risk, reversible, bar = cells[:5]
        try:
            category = Category(cause)
        except ValueError as e:
            raise PolicyError(f"unknown cause {cause!r} in the action risk table") from e
        m = re.match(r"([a-z]+)\s*(?:\((.*)\))?$", action)
        if not m or m.group(1) not in {a.value for a in Action}:
            raise PolicyError(f"{cause}: cannot read the action {action!r}")
        if risk not in RISKS:
            raise PolicyError(f"{cause}: risk must be one of {RISKS}, got {risk!r}")
        if bar.lower().startswith("never"):
            value = None
        else:
            n = re.match(r"([01](?:\.\d+)?)", bar)
            if not n:
                raise PolicyError(f"{cause}: cannot read the confidence bar {bar!r}")
            value = float(n.group(1))
        rows[category] = PolicyRow(category, Action(m.group(1)), risk,
                                   {"yes": True, "no": False}.get(reversible.lower()), value,
                                   "day time" in bar.lower(), m.group(2) or "")
    missing = set(Category) - set(rows)
    if missing:
        raise PolicyError("the action risk table has no row for: " + ", ".join(sorted(c.value for c in missing)))
    return rows


def load(path: Path = paths.ACTION_RISK_TABLE) -> dict[Category, PolicyRow]:
    return parse(Path(path).read_text(encoding="utf-8"))
