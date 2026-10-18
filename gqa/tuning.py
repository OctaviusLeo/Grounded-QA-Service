"""Refusal-threshold selection on the tuning split (Req 4.6).

A question is refused when its retrieval confidence is below the threshold. The chosen threshold
maximizes balanced accuracy, the mean of (unanswerable questions refused) and (answerable
questions answered); ties go to the lower threshold, which refuses less.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    balanced_accuracy: float
    refuse_unanswerable: float
    answer_answerable: float


def balanced_accuracy(confidences: Sequence[float], answerable: Sequence[bool], threshold: float) -> tuple[float, float, float]:
    una = [c for c, a in zip(confidences, answerable) if not a]
    ans = [c for c, a in zip(confidences, answerable) if a]
    if not una or not ans:
        raise ValueError("need both answerable and unanswerable questions to tune a threshold")
    tpr = sum(c < threshold for c in una) / len(una)
    tnr = sum(c >= threshold for c in ans) / len(ans)
    return (tpr + tnr) / 2.0, tpr, tnr


def choose_threshold(confidences: Sequence[float], answerable: Sequence[bool]) -> ThresholdChoice:
    if len(confidences) != len(answerable):
        raise ValueError("confidences and labels differ in length")
    # Every distinct confidence is a cut point; 0.0 refuses nothing (confidence is never < 0).
    candidates = sorted({0.0, *(min(1.0, max(0.0, c)) for c in confidences)})
    best: ThresholdChoice | None = None
    for t in candidates:
        ba, tpr, tnr = balanced_accuracy(confidences, answerable, t)
        if best is None or ba > best.balanced_accuracy + 1e-12:
            best = ThresholdChoice(t, ba, tpr, tnr)
    assert best is not None
    return best
