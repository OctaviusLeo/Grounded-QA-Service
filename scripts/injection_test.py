"""Injection_Test: python scripts/injection_test.py

Plants the instructions of data/injection_cases.yaml in their host passages, runs each host
question through the hybrid+rerank pipeline with the large model, classifies each case
(docs/INJECTION.md), and writes results/injection.json with the stamped followed fraction.
"""

from __future__ import annotations

import sys

from gqa.config import load_config
from gqa.evalset import load_questions
from gqa.injection import build_cases, injected_corpus, load_specs, report_to_dict, run_injection_test
from gqa.ingest import read_passages
from gqa.model_client import ModelClientError
from gqa.repro import make_stamp, write_json
from gqa.retrieval import RagLiteBackend
from gqa.runtime import make_pipeline

MODE = "hybrid_rerank"


def main() -> int:
    cfg = load_config()
    stamp = make_stamp(seed=cfg.seed)
    passages = read_passages(cfg.corpus_path)
    cases = build_cases(load_specs("data/injection_cases.yaml"), load_questions(cfg.evalset_path), {p.passage_id: p for p in passages})
    backend = RagLiteBackend(injected_corpus(passages, cases), weights=cfg.hybrid_weights, candidates=cfg.candidates)
    model = cfg.models["large"]
    pipe = make_pipeline(cfg, backend, MODE, model)
    report = run_injection_test(cases, pipe.run, cfg.seed, stamp.command, stamp.commit, errors=(ModelClientError,))
    body = {"mode": MODE, "model": model, "threshold": pipe.threshold, **report_to_dict(report),
            "host_questions": {c.case_id: c.question_id for c in cases}}
    write_json("results/injection.json", body, stamp)
    for o in report.outcomes:
        print(f"{o.case_id} {o.label:>13}: {o.reason}")
    print(f"followed fraction {report.followed_fraction:.3f} ({report.followed} followed, {report.not_followed} not followed, "
          f"{report.unclassified_count} unclassified); canary in raw output for {report.raw_followed} classified cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
