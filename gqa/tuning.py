"""Refusal-threshold selection on the tuning split (Req 4.6).

The pipeline refuses a question when its retrieval confidence is below the threshold, and the
model can also refuse ("not found in the documents") or have every claim rejected by citation
validation. Tuning is end-to-end: the pipeline is run once on every tuning question with the
gate open (threshold 0), which records each question's confidence and whether the rest of the
pipeline refused it. For any threshold ``t`` the final decision is then exact:
refused = confidence < t or the rest of the pipeline refused.

The chosen threshold maximizes balanced accuracy, the mean of (unanswerable questions refused)
and (answerable questions answered); ties go to the lower threshold, which refuses less.
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


def balanced_accuracy(confidences: Sequence[float], answerable: Sequence[bool], threshold: float,
                      downstream_refused: Sequence[bool] | None = None) -> tuple[float, float, float]:
    later = downstream_refused if downstream_refused is not None else [False] * len(confidences)
    refused = [c < threshold or r for c, r in zip(confidences, later)]
    una = [f for f, a in zip(refused, answerable) if not a]
    ans = [f for f, a in zip(refused, answerable) if a]
    if not una or not ans:
        raise ValueError("need both answerable and unanswerable questions to tune a threshold")
    tpr = sum(una) / len(una)
    tnr = sum(not f for f in ans) / len(ans)
    return (tpr + tnr) / 2.0, tpr, tnr


def choose_threshold(confidences: Sequence[float], answerable: Sequence[bool],
                     downstream_refused: Sequence[bool] | None = None) -> ThresholdChoice:
    if len(confidences) != len(answerable) or (downstream_refused is not None and len(downstream_refused) != len(answerable)):
        raise ValueError("confidences, labels, and downstream refusals differ in length")
    # Every distinct confidence is a cut point; 0.0 refuses nothing (confidence is never < 0).
    candidates = sorted({0.0, *(min(1.0, max(0.0, c)) for c in confidences)})
    best: ThresholdChoice | None = None
    for t in candidates:
        ba, tpr, tnr = balanced_accuracy(confidences, answerable, t, downstream_refused)
        if best is None or ba > best.balanced_accuracy + 1e-12:
            best = ThresholdChoice(t, ba, tpr, tnr)
    assert best is not None
    return best
