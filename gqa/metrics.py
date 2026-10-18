"""Evaluation metrics and bootstrap confidence intervals (Req 3.3, 7.1-7.6, 8.2).

Recall@K is Rag_Lite's ``calculate_recall_at_k`` applied to the top ``k`` results. nDCG@K is
computed here with the standard ideal DCG over ``min(|gold|, k)`` positions: Rag_Lite's
``calculate_ndcg_at_k`` sizes the ideal DCG by the length of the list it is given, which
overstates nDCG when the scoring floor returns fewer than ``k`` results. The two agree whenever
the list has ``k`` entries (tested).
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Callable, Hashable, Sequence, Set
from dataclasses import dataclass

import numpy as np
from src.evaluate import calculate_recall_at_k  # Rag_Lite

MIN_RESAMPLES = 1000

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD's official normalization: lower case, drop punctuation and articles, squeeze spaces."""
    return " ".join(_ARTICLES.sub(" ", text.lower().translate(_PUNCT)).split())


def exact_match(pred: str, gold: str | Sequence[str]) -> float:
    """1.0 if the normalized prediction equals a normalized gold answer (the best over several)."""
    golds = [gold] if isinstance(gold, str) else list(gold)
    if not golds:
        return 0.0
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def _f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, gold: str | Sequence[str]) -> float:
    """SQuAD token-level F1, the best over the gold answers."""
    golds = [gold] if isinstance(gold, str) else list(gold)
    return max((_f1(pred, g) for g in golds), default=0.0)


def citation_precision(cited: Set[str], gold: Set[str]) -> float:
    """Fraction of cited passages that are gold (0.0 when nothing is cited)."""
    return len(cited & gold) / len(cited) if cited else 0.0


def citation_recall(cited: Set[str], gold: Set[str]) -> float:
    """Fraction of gold passages that are cited (0.0 when there is no gold)."""
    return len(cited & gold) / len(gold) if gold else 0.0


def refusal_accuracy(refused: Sequence[bool], unanswerable: Sequence[bool]) -> float:
    """Fraction of Unanswerable_Questions answered with a Refusal."""
    if len(refused) != len(unanswerable):
        raise ValueError("refused and unanswerable flags differ in length")
    flags = [r for r, u in zip(refused, unanswerable) if u]
    if not flags:
        raise ValueError("no unanswerable questions")
    return sum(flags) / len(flags)


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa between two raters. Identical label sequences give 1.0, including the
    degenerate case where both raters use a single category (expected agreement 1)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label sequences of equal length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), ``q`` in [0, 100]."""
    if not values:
        raise ValueError("no values")
    return float(np.percentile(np.asarray(values, dtype=float), q))


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
