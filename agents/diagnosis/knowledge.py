"""The knowledge base: runbooks + past incidents, and how the agent searches them (RAG).

    knowledge_base/runbooks/*.md     how to recognise and fix one kind of cause
    knowledge_base/incidents/*.md    past incidents with their confirmed cause (hand-written examples,
                                     then every diagnosis a human approves on the approval page)

Every file starts with a small header:
    ---
    id: RB-bad-deploy
    title: Bad deployment
    category: bad_deploy
    tags: [source:metrics, evidence:deploy_before]
    ---

Search is HYBRID, because each half misses different things:
    BM25        exact words ("PacketResponder", "rollback", "E29")
    embeddings  meaning ("requests got slower" ~ "latency regression"), via Ollama nomic-embed-text
The two rankings are merged with reciprocal rank fusion (RRF). Without Ollama, BM25 alone is used.

Why not ChromaDB: 20 short documents fit in memory; a numpy matrix gives the same answer with no
server and no dependency. `Index` is the one class to replace if the knowledge base grows.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.schemas import Category

from . import ollama_http, paths

TOKEN = re.compile(r"[a-z0-9_]+")
RRF_K = 60            # standard constant of reciprocal rank fusion


# ------------------------------------------------------------------ documents
@dataclass
class Doc:
    id: str
    title: str
    kind: str                         # "runbook" | "incident" | "table"
    text: str
    category: Category | None = None
    sub_cause: str = ""
    tags: set[str] = field(default_factory=set)
    confirmed: bool = False           # a human confirmed the cause (past incidents)
    path: Path | None = None


@dataclass
class Chunk:
    doc: Doc
    section: str
    text: str

    @property
    def ref(self) -> str:
        return self.doc.id


def parse(text: str) -> tuple[dict, str]:
    """Header between two '---' lines -> dict; the rest is the body."""
    if not text.startswith("---"):
        return {}, text
    _, head, body = text.split("---", 2)
    meta = {}
    for line in head.strip().splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        elif value.lower() in ("true", "false"):
            value = value.lower() == "true"
        meta[key.strip()] = value
    return meta, body.strip()


def load_doc(path: Path) -> Doc:
    meta, body = parse(path.read_text(encoding="utf-8"))
    cat = meta.get("category")
    return Doc(id=meta.get("id", path.stem), title=meta.get("title", path.stem), kind=meta.get("kind", "runbook"),
               text=body, category=Category(cat) if cat else None, sub_cause=meta.get("sub_cause", ""),
               tags=set(meta.get("tags", [])), confirmed=bool(meta.get("confirmed", False)), path=path)


def chunks_of(doc: Doc) -> list[Chunk]:
    """One chunk per '## ' section (runbooks are short: a section is one idea)."""
    parts = re.split(r"^## ", doc.text, flags=re.M)
    out = [Chunk(doc, "summary", parts[0].strip())] if parts[0].strip() else []
    for p in parts[1:]:
        title, _, body = p.partition("\n")
        out.append(Chunk(doc, title.strip(), body.strip()))
    return out


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


# ------------------------------------------------------------------ the two rankers
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = sum(map(len, docs)) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avg)
            out[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf)
        return out


class Index:
    """Hybrid index over chunks. Embeddings are cached on disk by text hash."""
    def __init__(self, chunks: list[Chunk], embed_model: str | None = "nomic-embed-text",
                 cache: Path = paths.EMBEDDINGS):
        self.chunks = chunks
        self.bm25 = BM25([tokens(f"{c.doc.title} {c.section} {c.text}") for c in chunks])
        self.embed_model = embed_model
        self.cache = cache
        self.vectors = self._embed_all() if embed_model else None

    def _embed_all(self) -> np.ndarray | None:
        texts = [f"{c.doc.title}. {c.section}. {c.text}" for c in self.chunks]
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        stored = json.loads(self.cache.read_text()) if self.cache.exists() else {}
        todo = [(k, t) for k, t in zip(keys, texts) if k not in stored]
        if todo:
            try:
                vecs = ollama_http.embed(self.embed_model, [t for _, t in todo])
            except ollama_http.OllamaError:
                return None
            stored.update({k: v for (k, _), v in zip(todo, vecs)})
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            self.cache.write_text(json.dumps(stored))
        m = np.array([stored[k] for k in keys], float)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)

    def search(self, query: str, k: int = 3, boost: Category | None = None,
               kinds: tuple[str, ...] = ("runbook", "incident")) -> list[tuple[Chunk, float]]:
        keep = [i for i, c in enumerate(self.chunks) if c.doc.kind in kinds]
        if not keep:
            return []
        rankings = [np.argsort(-self.bm25.scores(tokens(query))[keep])]
        if self.vectors is not None:
            try:
                q = np.array(ollama_http.embed(self.embed_model, [query])[0], float)
                rankings.append(np.argsort(-(self.vectors[keep] @ (q / (np.linalg.norm(q) + 1e-12)))))
            except ollama_http.OllamaError:
                pass
        fused = np.zeros(len(keep))
        for r in rankings:
            fused[r] += 1.0 / (RRF_K + np.arange(1, len(r) + 1))
        if boost is not None:     # the hypothesis' own runbook counts as one more first-place vote
            fused += np.array([1.0 / (RRF_K + 1) if self.chunks[i].doc.category == boost else 0.0 for i in keep])
        out, seen = [], set()
        for j in np.argsort(-fused):          # best chunk of each document only: k different sources
            c = self.chunks[keep[j]]
            if c.doc.id not in seen:
                seen.add(c.doc.id)
                out.append((c, round(float(fused[j]), 4)))
            if len(out) == k:
                break
        return out

    @property
    def mode(self) -> str:
        return "bm25+embeddings" if self.vectors is not None else "bm25"


# ------------------------------------------------------------------ the knowledge base
class KnowledgeBase:
    def __init__(self, root: Path = paths.KB, embed_model: str | None = "nomic-embed-text"):
        self.root = Path(root)
        self.embed_model = embed_model
        self.reload()

    def reload(self) -> None:
        self.docs = [load_doc(p) for p in sorted(self.root.rglob("*.md")) if p.name.lower() != "readme.md"]
        self.by_id = {d.id: d for d in self.docs}
        self.index = Index([c for d in self.docs for c in chunks_of(d)], self.embed_model)

    @property
    def runbooks(self) -> list[Doc]:
        return [d for d in self.docs if d.kind == "runbook"]

    @property
    def incidents(self) -> list[Doc]:
        return [d for d in self.docs if d.kind == "incident"]

    def runbook_for(self, category: Category) -> Doc | None:
        return next((d for d in self.runbooks if d.category == category), None)

    def search(self, query: str, k: int = 3, boost: Category | None = None) -> list[tuple[Chunk, float]]:
        return self.index.search(query, k, boost)

    def mentions(self, pattern: str) -> bool:
        """Is this HDFS event pattern (e.g. 'E2+E3+E4') described anywhere?"""
        events = {f"event:{e}" for e in pattern.split("+") if e}
        return any(pattern in d.text or (events and events <= d.tags) for d in self.docs)

    def similar_incidents(self, tags: set[str], k: int = 3) -> list[tuple[Doc, float]]:
        from .evidence.history import jaccard
        scored = [(d, jaccard(tags, d.tags)) for d in self.incidents if d.tags]
        return sorted(scored, key=lambda x: -x[1])[:k]
