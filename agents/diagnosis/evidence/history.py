"""Have we seen this before? Case-based reasoning on the past incidents of the knowledge base.

Every incident is described by a set of TAGS (source, dataset, features, which evidence fired,
log events). A past incident with a similar tag set (Jaccard similarity) is evidence for the cause
written in it, more so when a human confirmed it on the approval page. This is how the agent
learns: an approved diagnosis becomes a past incident (feedback.py), and the next similar case finds it.
"""
from __future__ import annotations

from common.schemas import Incident

MIN_SIMILARITY = 0.6


def tags(inc: Incident, f) -> set[str]:
    out = {f"source:{inc.source}", f"dataset:{inc.dataset}", f"kind:{inc.kind}"}
    out |= {f"feature:{x}" for x in inc.features}
    out |= {f"evidence:{e.kind}" for e in f.evidence if e.kind not in ("triage_summary", "similar_incident")}
    out |= {f"event:{e['event']}" for a in inc.alerts[:1] for e in a.details.get("top_events", [])}
    return out


def collect(inc: Incident, f, kb) -> None:
    best = kb.similar_incidents(tags(inc, f), k=1)
    if not best:
        return
    doc, sim = best[0]
    if sim < MIN_SIMILARITY or doc.category is None:
        return
    weight = sim * (1.5 if doc.confirmed else 0.8)
    f.add("similar_incident", f"Past incident {doc.id} ('{doc.title}') looks like this one (similarity {sim:.2f}); "
                              f"its cause was {doc.category.value}"
                              + (f" ({doc.sub_cause})" if doc.sub_cause else "")
                              + (", confirmed by a human." if doc.confirmed else "."),
          "knowledge_base", {doc.category: weight}, doc_id=doc.id, similarity=round(sim, 2))
    if doc.sub_cause and doc.category not in f.sub_causes:
        f.sub_causes[doc.category] = doc.sub_cause


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0
