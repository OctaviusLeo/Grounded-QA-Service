"""Evaluation harness (Req 7.1-7.9, 0.1, 0.3).

One non-interactive command evaluates one Configuration on the Evaluation_Set (or its CI subset):

    python -m gqa.eval --config config/default.yaml --seed 1234 --mode hybrid_rerank \
        --model granite3.3:2b --subset ci --out results/eval/ci_run1.json

Answers are generated first and judged afterwards, so the Model_Server holds one model at a time.
Every metric is a mean of per-question values with a 95% bootstrap CI (1,000 resamples, seeded).
A metric whose inputs are missing or malformed is reported as ``not_computed`` with its cause,
and the remaining metrics are still computed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from gqa.answer import Answer, strip_citations
from gqa.config import Config, load_config
from gqa.evalset import EvalQuestion, EvalSetError, _from_json, ci_subset
from gqa.ingest import Passage
from gqa.judge import LLMJudge
from gqa.metrics import (
    bootstrap_ci,
    citation_precision,
    citation_recall,
    exact_match,
    ndcg_at_k,
    percentile,
    recall_at_k,
    token_f1,
)
from gqa.model_client import ModelClientError, ModelTimeoutError
from gqa.pipeline import AnsweringPipeline
from gqa.repro import make_stamp, write_json

KS = (5, 10, 20)
METRIC_ORDER = (
    [f"recall@{k}" for k in KS] + [f"ndcg@{k}" for k in KS]
    + ["exact_match", "f1", "citation_precision", "citation_recall", "refusal_accuracy", "answerable_refused", "faithfulness"]
)


@dataclass
class Record:
    question_id: str
    question: str
    answerable: bool
    answer: str
    cited_ids: list[str]
    refused: bool
    refusal_source: str | None
    candidate_ids: list[str]
    context_ids: list[str]
    confidence: float
    latency_ms: float
    retrieval_ms: float
    generation_ms: float
    prompt_tokens: int
    completion_tokens: int
    raw_output: str | None
    validation_accepted: bool | None = None
    judge_label: int | None = None
    judge_raw: str | None = None

    @property
    def answer_obj(self) -> Answer:
        return Answer(self.answer, tuple(self.cited_ids), self.refused)


def load_questions_lenient(path: str | Path) -> tuple[list[EvalQuestion], list[str]]:
    """Valid questions plus one problem line per malformed record (the run continues)."""
    p = Path(path)
    if not p.exists():
        return [], [f"{p} does not exist"]
    questions, problems = [], []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(_from_json(line, lineno))
        except EvalSetError as exc:
            problems.append(str(exc))
    return questions, problems


def generate(pipeline: AnsweringPipeline, questions: Sequence[EvalQuestion], retries: int = 1,
             progress: Callable[[int, int], None] | None = None) -> list[Record]:
    """Run the pipeline on every question. A timeout is retried; any other model error, or a
    timeout that persists, aborts the run (the caller reports the Configuration as failed)."""
    records = []
    for i, q in enumerate(questions, start=1):
        for attempt in range(retries + 1):
            try:
                t = pipeline.run(q.question)
                break
            except ModelTimeoutError:
                if attempt == retries:
                    raise
        records.append(Record(
            question_id=q.question_id, question=q.question, answerable=q.answerable,
            answer=t.answer.text, cited_ids=list(t.answer.cited_ids), refused=t.answer.refused,
            refusal_source=t.refusal_source, candidate_ids=[r.passage_id for r in t.candidates],
            context_ids=[r.passage_id for r in t.context], confidence=t.confidence,
            latency_ms=t.total_ms, retrieval_ms=t.retrieval_ms, generation_ms=t.generation_ms,
            prompt_tokens=t.prompt_tokens, completion_tokens=t.completion_tokens, raw_output=t.raw_output,
            validation_accepted=None if t.validation is None else t.validation.accepted,
        ))
        if progress:
            progress(i, len(questions))
    return records


def judge_records(judge: LLMJudge, records: Iterable[Record], passages_by_id: dict[str, Passage]) -> None:
    for r in records:
        if r.refused:
            continue
        verdict = judge.judge(r.answer_obj, [passages_by_id[pid] for pid in r.cited_ids if pid in passages_by_id], r.question)
        r.judge_label, r.judge_raw = verdict.label, verdict.raw


def _ci(values: Sequence[float], seed: int) -> dict[str, Any]:
    return bootstrap_ci(values, seed=seed).as_dict()


def _nc(cause: str) -> dict[str, str]:
    return {"not_computed": cause}


def compute_metrics(records: Sequence[Record], questions: Sequence[EvalQuestion], seed: int,
                    judged: bool = True) -> dict[str, dict[str, Any]]:
    """Every metric as a bootstrap CI over per-question values, or ``not_computed`` with its cause."""
    by_id = {q.question_id: q for q in questions}
    rows = [(r, by_id[r.question_id]) for r in records if r.question_id in by_id]
    out: dict[str, dict[str, Any]] = {}
    gold_rows = [(r, q) for r, q in rows if q.answerable and q.gold_passage_ids]
    for k in KS:
        if gold_rows:
            out[f"recall@{k}"] = _ci([recall_at_k(r.candidate_ids, set(q.gold_passage_ids), k) for r, q in gold_rows], seed)
            out[f"ndcg@{k}"] = _ci([ndcg_at_k(r.candidate_ids, set(q.gold_passage_ids), k) for r, q in gold_rows], seed)
        else:
            out[f"recall@{k}"] = out[f"ndcg@{k}"] = _nc("no answerable questions with gold Passage_IDs")
    answer_rows = [(r, q) for r, q in rows if q.answerable and q.gold_answers]
    if answer_rows:
        preds = [("" if r.refused else strip_citations(r.answer), q.gold_answers) for r, q in answer_rows]
        out["exact_match"] = _ci([exact_match(p, g) for p, g in preds], seed)
        out["f1"] = _ci([token_f1(p, g) for p, g in preds], seed)
    else:
        out["exact_match"] = out["f1"] = _nc("no answerable questions with gold answers")
    answered_gold = [(r, q) for r, q in gold_rows if not r.refused]
    out["citation_precision"] = (_ci([citation_precision(set(r.cited_ids), set(q.gold_passage_ids)) for r, q in answered_gold], seed)
                                 if answered_gold else _nc("no answered questions with gold Passage_IDs"))
    out["citation_recall"] = (_ci([citation_recall(set(r.cited_ids), set(q.gold_passage_ids)) for r, q in gold_rows], seed)
                              if gold_rows else _nc("no answerable questions with gold Passage_IDs"))
    una = [r for r, q in rows if not q.answerable]
    out["refusal_accuracy"] = _ci([float(r.refused) for r in una], seed) if una else _nc("no unanswerable questions")
    ans = [r for r, q in rows if q.answerable]
    out["answerable_refused"] = _ci([float(r.refused) for r in ans], seed) if ans else _nc("no answerable questions")
    labels = [r.judge_label for r, _ in rows if not r.refused and r.judge_label is not None]
    if not judged:
        out["faithfulness"] = _nc("judge disabled for this run")
    elif labels:
        out["faithfulness"] = _ci([float(x) for x in labels], seed)
    else:
        out["faithfulness"] = _nc("no answered question received a parseable judge verdict")
    return {name: out[name] for name in METRIC_ORDER}


def latency_summary(records: Sequence[Record]) -> dict[str, float | int]:
    """Pipeline latency only (the judge's calls are not included)."""
    values = [r.latency_ms for r in records]
    if not values:
        return {"n": 0}
    return {"n": len(values), "p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95),
            "mean_ms": sum(values) / len(values)}


def counts(records: Sequence[Record]) -> dict[str, int]:
    answered = [r for r in records if not r.refused]
    return {
        "questions": len(records),
        "answerable": sum(r.answerable for r in records),
        "unanswerable": sum(not r.answerable for r in records),
        "answered": len(answered),
        "refused_by_confidence": sum(r.refusal_source == "confidence" for r in records),
        "refused_by_model": sum(r.refusal_source == "model" for r in records),
        "refused_by_validation": sum(r.refusal_source == "validation" for r in records),
        "validation_rejected": sum(r.validation_accepted is False for r in records),
        "judged": sum(r.judge_label is not None for r in answered),
        "judge_unparseable": sum(r.judge_label is None and r.judge_raw is not None for r in answered),
        "prompt_tokens": sum(r.prompt_tokens for r in records),
        "completion_tokens": sum(r.completion_tokens for r in records),
    }


def write_records(path: str | Path, records: Sequence[Record]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(asdict(r), ensure_ascii=False, sort_keys=True) + "\n" for r in records),
                 encoding="utf-8", newline="\n")


def read_records(path: str | Path) -> list[Record]:
    return [Record(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_questions(cfg: Config, path: str, subset: str | None, ci_config: str = "config/ci.yaml") -> tuple[list[EvalQuestion], list[str]]:
    questions, problems = load_questions_lenient(path)
    if subset == "ci" and questions:
        sizes = yaml.safe_load(Path(ci_config).read_text(encoding="utf-8"))["subset"]
        try:
            questions = ci_subset(questions, int(sizes["answerable"]), int(sizes["unanswerable"]))
        except EvalSetError as exc:
            problems.append(str(exc))
    return questions, problems


def main(argv: list[str] | None = None, backend_factory=None, client_factory=None) -> int:
    from gqa.runtime import load_backend, make_client, make_pipeline

    parser = argparse.ArgumentParser(description="Evaluate one Configuration (Req 7.7).")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--seed", type=int, default=None, help="overrides the config seed")
    parser.add_argument("--mode", default="hybrid_rerank", choices=["bm25", "hybrid", "hybrid_rerank"])
    parser.add_argument("--model", default=None, help="answering model (default: models.large)")
    parser.add_argument("--judge-model", default=None, help="judge model (default: models.judge)")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--questions", default=None, help="question file (default: paths.evalset)")
    parser.add_argument("--subset", choices=["ci"], default=None)
    parser.add_argument("--out", required=True, help="output JSON; per-question records go beside it")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    seed = cfg.seed
    stamp = make_stamp(seed=seed)
    model = args.model or cfg.models["large"]
    judge_model = args.judge_model or cfg.models["judge"]
    questions, problems = select_questions(cfg, args.questions or cfg.evalset_path, args.subset)
    for p in problems:
        print(f"input problem: {p}", file=sys.stderr)
    out = Path(args.out)
    body: dict[str, Any] = {"mode": args.mode, "model": model, "judge_model": None if args.no_judge else judge_model,
                            "questions_file": args.questions or cfg.evalset_path, "subset": args.subset,
                            "input_problems": problems}
    if not questions:
        body["metrics"] = {name: _nc("no valid questions: " + "; ".join(problems)) for name in METRIC_ORDER}
        write_json(out, body, stamp)
        print("no valid questions; every metric not computed", file=sys.stderr)
        return 1
    make = client_factory or (lambda m: make_client(cfg, m))
    backend = (backend_factory or (lambda: load_backend(cfg)))()
    pipe = make_pipeline(cfg, backend, args.mode, model, client=make(model))
    try:
        records = generate(pipe, questions, progress=lambda i, n: print(f"generated {i}/{n}", flush=True) if i % 10 == 0 or i == n else None)
        judged = not args.no_judge
        if judged:
            judge_records(LLMJudge(make(judge_model)), records, backend.by_id)
    except ModelClientError as exc:
        body["failed"] = f"{type(exc).__name__}: {exc}"
        body["metrics"] = {name: _nc(body["failed"]) for name in METRIC_ORDER}
        write_json(out, body, stamp)
        print(f"FAILED: {body['failed']}", file=sys.stderr)
        return 1
    body["metrics"] = compute_metrics(records, questions, seed, judged=judged)
    body["latency"] = latency_summary(records)
    body["counts"] = counts(records)
    body["threshold"] = pipe.threshold
    write_records(out.with_suffix(".records.jsonl"), records)
    write_json(out, body, stamp)
    for name, value in body["metrics"].items():
        print(f"{name:>20}: " + (f"{value['point']:.3f} [{value['lower']:.3f}, {value['upper']:.3f}]" if "point" in value else f"not computed ({value['not_computed']})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
