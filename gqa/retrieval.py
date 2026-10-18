"""Retrieval_Layer: a thin adapter over Rag_Lite (Req 3.1, 3.2, 3.4).

All scoring is Rag_Lite's: ``build_index`` builds the TF-IDF, BM25, dense-embedding, and
cross-encoder components, and ``retrieve_hybrid`` scores BM25-only, hybrid (weighted TF-IDF +
BM25 + dense), or hybrid followed by cross-encoder reranking. This module only maps Rag_Lite's
row indices to Passage_IDs, orders and de-duplicates the results, and applies the scoring floor.

Rag_Lite always returns the top ``k`` rows, even when nothing matches, so the floor is lexical:
a passage is dropped unless it shares at least one content word (not an English stop word) with
the query. A query with no content word in common with any passage returns ``[]``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.rag import build_index, retrieve_hybrid  # Rag_Lite

from gqa.hf_models import require_pins
from gqa.ingest import Passage

_TOKEN = re.compile(r"[a-z0-9]+")


def content_terms(text: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS)


@dataclass(frozen=True)
class RetrievedPassage:
    passage_id: str
    score: float
    rank: int


class RetrievalMode(str, Enum):
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class RetrievalError(RuntimeError):
    """The backend cannot serve the requested mode."""


class RagLiteBackend:
    """Owns one Rag_Lite index over the Corpus passages (row ``i`` is ``passages[i]``)."""

    def __init__(self, passages: Sequence[Passage], *, use_embeddings: bool = True, use_reranker: bool = True,
                 weights: dict[str, float] | None = None, candidates: int = 20, check_pins: bool = True) -> None:
        if not passages:
            raise RetrievalError("cannot index an empty corpus")
        if check_pins and (use_embeddings or use_reranker):
            require_pins()
        self.passages = list(passages)
        self.by_id = {p.passage_id: p for p in self.passages}
        self.terms = [content_terms(p.text) for p in self.passages]
        self.candidates = candidates
        self.weights = weights or {"tfidf": 0.5, "bm25": 0.3, "embeddings": 0.2}
        self.use_embeddings = use_embeddings
        self.use_reranker = use_reranker
        self.index = build_index([p.text for p in self.passages], use_bm25=True,
                                 use_embeddings=use_embeddings, use_reranker=use_reranker)

    def search(self, query: str, depth: int, mode: RetrievalMode) -> list[tuple[int, float]]:
        if mode is RetrievalMode.HYBRID_RERANK and not self.use_reranker:
            raise RetrievalError("this index was built without the cross-encoder")
        method = "bm25" if mode is RetrievalMode.BM25 else "hybrid"
        rerank = mode is RetrievalMode.HYBRID_RERANK
        hits = retrieve_hybrid(
            self.index, query, k=depth, method=method,
            tfidf_weight=self.weights["tfidf"], bm25_weight=self.weights["bm25"],
            embedding_weight=self.weights["embeddings"] if self.use_embeddings else 0.0,
            rerank=rerank, rerank_top_k=depth,
        )
        return [(int(i), float(score)) for i, score, _ in hits]


class RagLiteRetriever:
    def __init__(self, mode: RetrievalMode, backend: RagLiteBackend, seed: int = 0) -> None:
        self.mode = RetrievalMode(mode)
        self.backend = backend
        self.seed = seed  # retrieval is deterministic; kept so every component exposes a seed

    def retrieve(self, query: str, k: int) -> list[RetrievedPassage]:
        """Up to ``k`` passages by descending score, unique Passage_IDs, ``[]`` on no match."""
        if k <= 0:
            return []
        q_terms = content_terms(query)
        if not q_terms:
            return []
        depth = max(k, self.backend.candidates)
        hits = sorted(self.backend.search(query, depth, self.mode), key=lambda h: (-h[1], h[0]))
        seen: set[str] = set()
        kept: list[tuple[str, float]] = []
        for row, score in hits:
            pid = self.backend.passages[row].passage_id
            if pid in seen or not (q_terms & self.backend.terms[row]):
                continue
            seen.add(pid)
            kept.append((pid, score))
        return [RetrievedPassage(pid, score, rank) for rank, (pid, score) in enumerate(kept[:k])]

    def passages_for(self, retrieved: Sequence[RetrievedPassage]) -> list[Passage]:
        return [self.backend.by_id[r.passage_id] for r in retrieved]
