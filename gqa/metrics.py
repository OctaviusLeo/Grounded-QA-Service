"""Evaluation metrics and bootstrap confidence intervals (Req 3.3, 7.1-7.6, 8.2).

Recall@K is Rag_Lite's ``calculate_recall_at_k`` applied to the top ``k`` results. nDCG@K is
computed here with the standard ideal DCG over ``min(|gold|, k)`` positions: Rag_Lite's
``calculate_ndcg_at_k`` sizes the ideal DCG by the length of the list it is given, which
overstates nDCG when the scoring floor returns fewer than ``k`` results. The two agree whenever
the list has ``k`` entries (tested).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence, Set
from dataclasses import dataclass

import numpy as np
from src.evaluate import calculate_recall_at_k  # Rag_Lite

MIN_RESAMPLES = 1000


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Set[str], k: int) -> float:
    """Fraction of gold passages among the top ``k`` retrieved (0.0 when there is no gold)."""
    _check_k(k)
    return float(calculate_recall_at_k(sorted(gold_ids), list(retrieved_ids)[:k]))


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Set[str], k: int) -> float:
    """Binary-relevance nDCG@K; each gold passage counts once."""
    _check_k(k)
    if not gold_ids:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for rank, pid in enumerate(list(retrieved_ids)[:k], start=1):
        if pid in gold_ids and pid not in seen:
            dcg += 1.0 / math.log2(rank + 1)
            seen.add(pid)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / idcg


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float
    n: int = 0
    resamples: int = MIN_RESAMPLES
    confidence: float = 0.95

    def as_dict(self) -> dict[str, float | int]:
        return {"point": self.point, "lower": self.lower, "upper": self.upper, "n": self.n,
                "resamples": self.resamples, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict) -> BootstrapCI:
        return cls(float(d["point"]), float(d["lower"]), float(d["upper"]), int(d.get("n", 0)),
                   int(d.get("resamples", MIN_RESAMPLES)), float(d.get("confidence", 0.95)))


def bootstrap_ci(samples: Sequence[float], seed: int, resamples: int = MIN_RESAMPLES, confidence: float = 0.95,
                 statistic: Callable[[np.ndarray], np.ndarray] | None = None) -> BootstrapCI:
    """Percentile bootstrap CI of the mean (or ``statistic``) of per-question ``samples``.

    Deterministic for a given seed. If floating-point rounding puts the point estimate a hair
    outside the percentile interval, the interval is widened to include it.
    """
    if resamples < MIN_RESAMPLES:
        raise ValueError(f"need at least {MIN_RESAMPLES} resamples, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    stat = statistic or (lambda a: a.mean(axis=-1))
    point = float(stat(values[np.newaxis, :])[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(resamples, values.size))
    dist = stat(values[idx])
    alpha = (1.0 - confidence) / 2.0
    lower, upper = (float(x) for x in np.quantile(dist, [alpha, 1.0 - alpha]))
    return BootstrapCI(point, min(lower, point), max(upper, point), int(values.size), resamples, confidence)
