"""Live pipeline check (not a reported number): python scripts/pipeline_check.py

Runs 5 tuning-split questions through the real hybrid+rerank pipeline with granite3.3:2b and
fails unless every Answer is a Refusal or cites only passages that were in its prompt.
Output: $GQA_RUNS/pipeline_check.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from gqa.config import load_config
from gqa.evalset import load_questions
from gqa.runtime import load_backend, make_pipeline


def main() -> int:
    cfg = load_config()
    if "hybrid_rerank" not in cfg.thresholds:
        print("config/thresholds.yaml has no tuned hybrid_rerank threshold; run scripts/tune_thresholds.py", file=sys.stderr)
        return 1
    pipe = make_pipeline(cfg, load_backend(cfg), "hybrid_rerank", cfg.models["small"])
    rows, bad = [], 0
    for q in load_questions(cfg.tuning_path)[:5]:
        t = pipe.run(q.question)
        context = {r.passage_id for r in t.context}
        ok = t.answer.refused or (bool(t.answer.cited_ids) and set(t.answer.cited_ids) <= context)
        bad += not ok
        rows.append({"question": q.question, "answerable": q.answerable, "confidence": round(t.confidence, 4),
                     "answer": t.answer.text, "cited": list(t.answer.cited_ids), "refusal_source": t.refusal_source,
                     "raw_output": t.raw_output, "ok": ok})
        print(("ok  " if ok else "BAD ") + f"{q.question!r} -> {t.answer.text!r} ({t.refusal_source or 'answered'})")
    out = Path(os.environ.get("GQA_RUNS", Path.home() / "gqa-runs")) / "pipeline_check.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{5 - bad}/5 answers are refusals or cite only retrieved passages; details: {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
