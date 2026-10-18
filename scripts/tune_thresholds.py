"""Tune refusal thresholds on the tuning split only: python scripts/tune_thresholds.py

Writes config/thresholds.yaml (generated; do not edit). The reported Evaluation_Set is never read.
"""

from __future__ import annotations

import sys

import yaml

from gqa.config import load_config
from gqa.confidence import estimate_confidence
from gqa.evalset import load_questions
from gqa.ingest import read_passages
from gqa.repro import make_stamp, stamp_to_dict
from gqa.retrieval import RagLiteBackend, RagLiteRetriever, RetrievalMode
from gqa.tuning import choose_threshold


def main() -> int:
    cfg = load_config()
    stamp = make_stamp(seed=cfg.seed)
    questions = load_questions(cfg.tuning_path)
    backend = RagLiteBackend(read_passages(cfg.corpus_path), weights=cfg.hybrid_weights, candidates=cfg.candidates)
    thresholds, details = {}, {}
    for mode in RetrievalMode:
        r = RagLiteRetriever(mode, backend, seed=cfg.seed)
        conf = [estimate_confidence(r.retrieve(q.question, cfg.k), mode) for q in questions]
        choice = choose_threshold(conf, [q.answerable for q in questions])
        thresholds[mode.value] = round(choice.threshold, 6)
        details[mode.value] = {
            "balanced_accuracy": round(choice.balanced_accuracy, 6),
            "unanswerable_refused": round(choice.refuse_unanswerable, 6),
            "answerable_answered": round(choice.answer_answerable, 6),
        }
        print(mode.value, thresholds[mode.value], details[mode.value])
    body = {
        "generated_by": "scripts/tune_thresholds.py (do not edit by hand)",
        "tuning_split": cfg.tuning_path,
        "questions": len(questions),
        "rule": "refuse when retrieval confidence < threshold; threshold maximizes balanced accuracy on the tuning split, ties to the lower value",
        "thresholds": thresholds,
        "tuning_split_scores": details,
        "stamp": stamp_to_dict(stamp),
    }
    with open(cfg.thresholds_path, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(body, fh, sort_keys=False, width=120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
