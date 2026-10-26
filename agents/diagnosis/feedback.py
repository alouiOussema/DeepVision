"""Human in the loop: the approval inbox, and how the agent LEARNS from what humans say.

    inbox      diagnoses waiting for a human (written by the pipeline / demo, read by ui/approval_app.py)
    review     approve | correct (with the right cause) | reject
               -> appended to reviews.jsonl (the KPI "how often was P3 right, by cause")
               -> approve / correct: written to knowledge_base/incidents/ as a CONFIRMED past incident
                  with the incident's tags. The next similar incident finds it (evidence/history.py),
                  so one correction teaches the agent without retraining anything.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from common.schemas import Alert, Category, Decision, Diagnosis, Incident, Severity

from . import paths
from .evidence.history import tags
from .knowledge import KnowledgeBase

VERDICTS = {"approve", "correct", "reject"}


# ------------------------------------------------------------------ (de)serialising incidents
def incident_from_dict(d: dict) -> Incident:
    d = dict(d)
    d["alerts"] = [Alert.from_dict(a) for a in d.get("alerts", [])]
    d["start"], d["end"] = datetime.fromisoformat(d["start"]), datetime.fromisoformat(d["end"])
    d["severity"], d["decision"] = Severity(d["severity"]), Decision(d["decision"])
    return Incident(**{k: d[k] for k in Incident.__dataclass_fields__ if k in d})


# ------------------------------------------------------------------ inbox
def save_inbox(items: list[tuple[Incident, Diagnosis]], path: Path = paths.INBOX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"incident": i.to_dict(), "diagnosis": d.to_dict()} for i, d in items],
                               indent=1, default=str))


def load_inbox(path: Path = paths.INBOX) -> list[tuple[Incident, Diagnosis]]:
    if not Path(path).exists():
        return []
    return [(incident_from_dict(x["incident"]), Diagnosis.from_dict(x["diagnosis"]))
            for x in json.loads(Path(path).read_text())]


def reviewed_ids(path: Path = paths.REVIEWS) -> set[str]:
    if not Path(path).exists():
        return set()
    return {json.loads(line)["diagnosis_id"] for line in Path(path).read_text().splitlines() if line.strip()}


# ------------------------------------------------------------------ one review
def record_review(inc: Incident, diag: Diagnosis, verdict: str, corrected: Category | None = None,
                  sub_cause: str = "", note: str = "", reviewer: str = "operator",
                  kb: KnowledgeBase | None = None, reviews: Path = paths.REVIEWS,
                  incidents_dir: Path = paths.PAST_INCIDENTS) -> Path | None:
    """Log the review; approve/correct also become a confirmed past incident. Returns its file, if any."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    if verdict == "correct" and corrected is None:
        raise ValueError("a correction needs the right cause")
    final = corrected if verdict == "correct" else diag.category
    row = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reviewer": reviewer,
           "incident_id": inc.id, "diagnosis_id": diag.id, "verdict": verdict,
           "diagnosed": diag.category.value, "confidence": diag.confidence, "final": final.value,
           "sub_cause": sub_cause or (diag.sub_cause if verdict == "approve" else ""), "note": note}
    reviews.parent.mkdir(parents=True, exist_ok=True)
    with open(reviews, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    if verdict == "reject":
        return None

    out = _write_past_incident(inc, diag, final, row["sub_cause"], note, reviewer, incidents_dir)
    if kb is not None:
        kb.reload()
    return out


def _write_past_incident(inc, diag, cause: Category, sub: str, note: str, reviewer: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{inc.id}-{cause.value}".lower()).strip("-")
    path = folder / f"HR-{slug}.md"
    tag_list = ", ".join(sorted(tags(inc, diag.evidence)))
    evidence = "\n".join(f"- [{e.id}] {e.statement}" for e in diag.evidence)
    agent_said = f"{diag.category.value}" + (f" ({diag.sub_cause})" if diag.sub_cause else "")
    path.write_text(f"""---
id: HR-{inc.id}
title: {inc.summary[:90].replace(':', ' -')}
kind: incident
category: {cause.value}
sub_cause: {sub}
confirmed: true
tags: [{tag_list}]
---
Confirmed by {reviewer} on the approval page ({datetime.now(timezone.utc):%Y-%m-%d}).

## What happened
{inc.summary}

## Cause
{cause.value}{f' ({sub})' if sub else ''}. The agent said {agent_said} with confidence {diag.confidence:.2f}.
{note}

## Evidence
{evidence}
""", encoding="utf-8")
    return path


def accuracy_report(path: Path = paths.REVIEWS) -> dict:
    """What humans said so far: share of diagnoses approved, per diagnosed cause."""
    if not Path(path).exists():
        return {}
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    out: dict = {}
    for r in rows:
        s = out.setdefault(r["diagnosed"], {"reviewed": 0, "approved": 0})
        s["reviewed"] += 1
        s["approved"] += r["verdict"] == "approve"
    return {k: {**v, "approval_rate": round(v["approved"] / v["reviewed"], 2)} for k, v in out.items()}
