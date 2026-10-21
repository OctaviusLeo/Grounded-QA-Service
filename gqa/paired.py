"""Paired bootstrap comparison of two configurations on the same evaluation questions.

The per-configuration CIs in the Results_Table answer "how precise is each estimate"; overlapping
intervals do not show whether two configurations differ. Every configuration answered the same
questions, so the difference can be bootstrapped directly: each resample draws questions once and
scores both configurations on that draw, which removes the question-to-question variance the
two share.

The metrics are defined exactly as in :func:`gqa.eval.compute_metrics`: Recall@K over answerable
questions with gold passages, and citation precision over the subset of those a configuration
answered. A question one configuration refused counts only toward the other's mean, so the point
estimate is the difference of the two reported point estimates.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gqa.eval import Record
from gqa.evalset import EvalQuestion
from gqa.metrics import BootstrapCI, bootstrap_ci, citation_precision, recall_at_k

METRICS = ("recall@5", "citation_precision")


def per_question(records: Sequence[Record], questions: Sequence[EvalQuestion]) -> dict[str, list[float | None]]:
    """Each metric's value for every answerable question with gold passages, in ``questions`` order.

    ``None`` marks a question the metric does not score for this configuration (a refusal, for
    citation precision).
    """
    by_id = {r.question_id: r for r in records}
    out: dict[str, list[float | None]] = {m: [] for m in METRICS}
    for q in questions:
        if not (q.answerable and q.gold_passage_ids):
            continue
        r = by_id[q.question_id]
        gold = set(q.gold_passage_ids)
        out["recall@5"].append(recall_at_k(r.candidate_ids, gold, 5))
        out["citation_precision"].append(None if r.refused else citation_precision(set(r.cited_ids), gold))
    return out


def paired_difference(a: Sequence[float | None], b: Sequence[float | None], seed: int) -> BootstrapCI:
    """CI of mean(a) - mean(b), resampling questions jointly; each mean skips its own ``None`` values."""
    if len(a) != len(b):
        raise ValueError(f"paired samples differ in length: {len(a)} vs {len(b)}")
    va = np.array([np.nan if x is None else x for x in a], dtype=float)
    vb = np.array([np.nan if x is None else x for x in b], dtype=float)

    def diff(idx: np.ndarray) -> np.ndarray:
        i = idx.astype(int)
        return np.nanmean(va[i], axis=-1) - np.nanmean(vb[i], axis=-1)

    return bootstrap_ci(np.arange(va.size), seed=seed, statistic=diff)


def compare(records_a: Sequence[Record], records_b: Sequence[Record], questions: Sequence[EvalQuestion],
            seed: int) -> dict[str, dict[str, float | int]]:
    """``{metric: CI of (a - b) as a dict, plus the number of questions each side scored}``."""
    pa, pb = per_question(records_a, questions), per_question(records_b, questions)
    out = {}
    for m in METRICS:
        ci = paired_difference(pa[m], pb[m], seed).as_dict()
        ci.update(n_a=sum(x is not None for x in pa[m]), n_b=sum(x is not None for x in pb[m]))
        out[m] = ci
    return out
