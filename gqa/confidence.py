"""Retrieval confidence and the refusal decision (Req 4.4, 4.6).

Confidence is the top retrieval score mapped into [0, 1]: the logistic of the cross-encoder
logit in rerank mode, and the fused score clipped to [0, 1] otherwise. Empty retrieval has
confidence 0.0. Rag_Lite divides BM25 scores by the per-query maximum, so BM25-only confidence
is 1.0 whenever anything is retrieved (a known limitation, documented in the README).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from gqa.config import validate_threshold
from gqa.retrieval import RetrievalMode, RetrievedPassage


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def estimate_confidence(ranked: Sequence[RetrievedPassage], mode: RetrievalMode | str = RetrievalMode.HYBRID) -> float:
    if not ranked:
        return 0.0
    top = max(r.score for r in ranked)
    if math.isnan(top):
        return 0.0
    if RetrievalMode(mode) is RetrievalMode.HYBRID_RERANK:
        return _logistic(top)
    return min(1.0, max(0.0, top))


def should_refuse(confidence: float, threshold: float) -> bool:
    return confidence < validate_threshold(threshold, "threshold")
