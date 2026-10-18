"""Run the configuration experiments: python scripts/run_experiments.py

Reads config/experiments.yaml, answers every Evaluation_Set question with each Configuration,
judges the answers, and writes results/runs/<config>.records.jsonl and results/experiments.json.
Exits non-zero if any Configuration failed (the others are still written).
"""

from __future__ import annotations

import sys

from gqa.config import MODES, load_config
from gqa.evalset import load_questions
from gqa.experiments import load_experiments, results_to_json, run_experiments
from gqa.gpu import GpuSampler
from gqa.judge import LLMJudge
from gqa.metrics import MIN_RESAMPLES
from gqa.repro import make_stamp, write_json
from gqa.runtime import load_backend, make_client, make_pipeline


def main() -> int:
    cfg = load_config()
    stamp = make_stamp(seed=cfg.seed)
    missing = [m for m in MODES if m not in cfg.thresholds]
    if missing:
        print(f"thresholds not tuned for {missing}; run scripts/tune_thresholds.py first", file=sys.stderr)
        return 1
    configs, cost, raw = load_experiments("config/experiments.yaml")
    questions = load_questions(raw["questions"])
    backend = load_backend(cfg)
    results = run_experiments(
        configs, questions, cfg.seed,
        make_pipeline=lambda c, client: make_pipeline(cfg, backend, c.mode.value, c.model, client=client),
        make_client=lambda m: make_client(cfg, m),
        make_judge=lambda: LLMJudge(make_client(cfg, raw["judge_model"])),
        make_sampler=lambda: GpuSampler(interval_s=0.5),
        passages_by_id=backend.by_id, cost=cost, out_dir="results/runs",
        log=lambda s: print(s, flush=True),
    )
    write_json("results/experiments.json", {
        "questions": len(questions),
        "answerable": sum(q.answerable for q in questions),
        "unanswerable": sum(not q.answerable for q in questions),
        "resamples": MIN_RESAMPLES,
        "judge_model": raw["judge_model"],
        "thresholds": cfg.thresholds,
        "electricity_usd_per_kwh": cost.usd_per_kwh,
        "electricity_source": " ".join(cost.source.split()),
        "energy_scope": raw["cost"]["energy_scope"],
        "configurations": results_to_json(results),
    }, stamp)
    failed = [r.config for r in results if r.failed]
    print("failed configurations: " + (", ".join(failed) if failed else "none"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
