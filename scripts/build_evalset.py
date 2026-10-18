"""Build the Evaluation_Set and the tuning split: python scripts/build_evalset.py

From the SQuAD 2.0 dev questions, sample (seeded) a reported Evaluation_Set of 150 answerable +
50 unanswerable questions and a disjoint tuning split of 70 + 30. Gold Passage_IDs are the
corpus passages of the question's paragraph that contain a gold answer span. Both files are
validated against the chunked Corpus before they are written.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Sequence

from gqa.config import load_config
from gqa.evalset import EvalQuestion, validate, write_questions
from gqa.ingest import Passage, read_passages
from gqa.repro import make_stamp, write_json
from gqa.squad import SQUAD_LICENSE, SQUAD_SHA256, SquadQuestion, ensure_downloaded, load_squad

EVAL_ANSWERABLE, EVAL_UNANSWERABLE = 150, 50
TUNE_ANSWERABLE, TUNE_UNANSWERABLE = 70, 30
MIN_WORDS = 3


def gold_passages(q: SquadQuestion, by_doc: dict[str, list[Passage]]) -> tuple[str, ...]:
    """Passages that fully contain some gold answer span; failing that, any that overlap one."""
    passages = by_doc.get(q.doc_id, [])
    full, partial = [], []
    for text, start in q.answers:
        end = start + len(text)
        for p in passages:
            if p.start_char <= start and end <= p.end_char:
                full.append(p.passage_id)
            elif p.start_char < end and start < p.end_char:
                partial.append(p.passage_id)
    chosen = full or partial
    return tuple(dict.fromkeys(chosen))


def to_eval(q: SquadQuestion, by_doc: dict[str, list[Passage]]) -> EvalQuestion:
    if q.is_impossible:
        return EvalQuestion(q.question_id, q.question, None, (), False, (), q.doc_id)
    answers = tuple(dict.fromkeys(t for t, _ in q.answers))
    return EvalQuestion(q.question_id, q.question, answers[0], gold_passages(q, by_doc), True, answers, q.doc_id)


def split(pool: Sequence[SquadQuestion], rng: random.Random, n_eval: int, n_tune: int):
    items = sorted(pool, key=lambda q: q.question_id)
    rng.shuffle(items)
    return items[:n_eval], items[n_eval:n_eval + n_tune]


def main() -> int:
    cfg = load_config()
    stamp = make_stamp(seed=cfg.seed)
    passages = read_passages(cfg.corpus_path)
    by_doc: dict[str, list[Passage]] = {}
    for p in passages:
        by_doc.setdefault(p.doc_id, []).append(p)
    data = load_squad(ensure_downloaded())
    usable = [q for q in data.questions if len(q.question.split()) >= MIN_WORDS]
    rng = random.Random(cfg.seed)
    ans_eval, ans_tune = split([q for q in usable if not q.is_impossible], rng, EVAL_ANSWERABLE, TUNE_ANSWERABLE)
    una_eval, una_tune = split([q for q in usable if q.is_impossible], rng, EVAL_UNANSWERABLE, TUNE_UNANSWERABLE)
    evalset = [to_eval(q, by_doc) for q in ans_eval + una_eval]
    tuning = [to_eval(q, by_doc) for q in ans_tune + una_tune]
    rng.shuffle(evalset)
    rng.shuffle(tuning)
    ids = {p.passage_id for p in passages}
    ok = True
    for name, qs, mins in (("evalset", evalset, (150, 20)), ("tuning", tuning, (100, 20))):
        report = validate(qs, ids, *mins)
        print(f"{name}: " + "; ".join(report.lines()))
        ok = ok and report.ok
    overlap = {q.question_id for q in evalset} & {q.question_id for q in tuning}
    if overlap:
        print(f"FAILED: {len(overlap)} questions are in both splits", file=sys.stderr)
        return 1
    if not ok:
        return 1
    write_questions(cfg.evalset_path, evalset)
    write_questions(cfg.tuning_path, tuning)
    write_json("data/evalset_manifest.json", {
        "source_sha256": SQUAD_SHA256,
        "license": SQUAD_LICENSE,
        "min_question_words": MIN_WORDS,
        "evalset": {"path": cfg.evalset_path, "answerable": EVAL_ANSWERABLE, "unanswerable": EVAL_UNANSWERABLE},
        "tuning": {"path": cfg.tuning_path, "answerable": TUNE_ANSWERABLE, "unanswerable": TUNE_UNANSWERABLE},
        "multi_gold_questions": sum(len(q.gold_passage_ids) > 1 for q in evalset),
        "disjoint": True,
    }, stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
