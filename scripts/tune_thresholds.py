"""Tune refusal thresholds end-to-end on the tuning split only: python scripts/tune_thresholds.py

For each retrieval mode, run the pipeline with the large model and the gate open on every tuning
question, then choose the threshold (gqa/tuning.py). Writes config/thresholds.yaml (generated; do
not edit). The reported Evaluation_Set is never read. Raw traces go to $GQA_RUNS/tuning/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

from gqa.config import load_config
from gqa.evalset import load_questions
from gqa.repro import make_stamp, stamp_to_dict
from gqa.retrieval import RetrievalMode
from gqa.runtime import load_backend, make_pipeline
from gqa.tuning import choose_threshold


def main() -> int:
    cfg = load_config()
    stamp = make_stamp(seed=cfg.seed)
    model = cfg.models["large"]
    questions = load_questions(cfg.tuning_path)
    backend = load_backend(cfg)
    out_dir = Path(os.environ.get("GQA_RUNS", Path.home() / "gqa-runs")) / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds, details = {}, {}
    for mode in RetrievalMode:
        pipe = make_pipeline(cfg, backend, mode.value, model, threshold=0.0)
        conf, later, rows = [], [], []
        for i, q in enumerate(questions, 1):
            t = pipe.run(q.question)
            conf.append(t.confidence)
            later.append(t.answer.refused)
            rows.append({"question_id": q.question_id, "answerable": q.answerable, "confidence": t.confidence,
                         "refused_downstream": t.answer.refused, "raw_output": t.raw_output})
            if i % 25 == 0:
                print(f"{mode.value}: {i}/{len(questions)}", flush=True)
        (out_dir / f"{mode.value}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        choice = choose_threshold(conf, [q.answerable for q in questions], later)
        gate_only = choose_threshold(conf, [q.answerable for q in questions])
        thresholds[mode.value] = round(choice.threshold, 6)
        details[mode.value] = {
            "balanced_accuracy": round(choice.balanced_accuracy, 6),
            "unanswerable_refused": round(choice.refuse_unanswerable, 6),
            "answerable_answered": round(choice.answer_answerable, 6),
            "model_alone_refused_unanswerable": round(sum(r for r, q in zip(later, questions) if not q.answerable) / sum(not q.answerable for q in questions), 6),
            "model_alone_answered_answerable": round(sum(not r for r, q in zip(later, questions) if q.answerable) / sum(q.answerable for q in questions), 6),
            "retrieval_only_threshold_for_comparison": round(gate_only.threshold, 6),
        }
        print(mode.value, thresholds[mode.value], details[mode.value], flush=True)
    body = {
        "generated_by": "scripts/tune_thresholds.py (do not edit by hand)",
        "tuning_split": cfg.tuning_path,
        "questions": len(questions),
        "tuning_model": model,
        "rule": ("refuse when retrieval confidence < threshold; threshold maximizes end-to-end balanced accuracy "
                 "(gate or downstream refusal) on the tuning split, ties to the lower value"),
        "thresholds": thresholds,
        "tuning_split_scores": details,
        "stamp": stamp_to_dict(stamp),
    }
    with open(cfg.thresholds_path, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(body, fh, sort_keys=False, width=120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
