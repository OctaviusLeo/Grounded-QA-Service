"""Evaluation_Set records, loading, and validation (Req 2.3, 2.4, 2.6, 7.9).

Each line of ``data/evalset.jsonl`` is one :class:`EvalQuestion`. An answerable question carries
at least one gold Passage_ID and its gold answers; an Unanswerable_Question carries none, and the
correct behavior for it is a Refusal.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

MIN_QUESTIONS = 150
MIN_UNANSWERABLE = 20


class EvalSetError(ValueError):
    """A line of the Evaluation_Set file is missing or malformed."""


@dataclass(frozen=True)
class EvalQuestion:
    question_id: str
    question: str
    gold_answer: str | None
    gold_passage_ids: tuple[str, ...]
    answerable: bool
    gold_answers: tuple[str, ...] = ()
    source_doc_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "gold_passage_ids", tuple(self.gold_passage_ids))
        object.__setattr__(self, "gold_answers", tuple(self.gold_answers))


def _from_json(line: str, lineno: int) -> EvalQuestion:
    try:
        raw = json.loads(line)
        return EvalQuestion(
            question_id=str(raw["question_id"]),
            question=str(raw["question"]),
            gold_answer=raw.get("gold_answer"),
            gold_passage_ids=tuple(raw.get("gold_passage_ids", ())),
            answerable=bool(raw["answerable"]),
            gold_answers=tuple(raw.get("gold_answers", ())),
            source_doc_id=str(raw.get("source_doc_id", "")),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise EvalSetError(f"line {lineno}: malformed question record ({exc})") from exc


def load_questions(path: str | Path) -> list[EvalQuestion]:
    p = Path(path)
    if not p.exists():
        raise EvalSetError(f"{p} does not exist")
    out = []
    with p.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.strip():
                out.append(_from_json(line, lineno))
    return out


def write_questions(path: str | Path, questions: Iterable[EvalQuestion]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for q in questions:
            fh.write(json.dumps(asdict(q), ensure_ascii=False, sort_keys=True) + "\n")


@dataclass
class ValidationReport:
    total: int
    answerable: int
    unanswerable: int
    errors: list[str] = field(default_factory=list)
    unresolved: dict[str, list[str]] = field(default_factory=dict)  # question_id -> gold ids not in corpus

    @property
    def unresolved_count(self) -> int:
        return sum(len(v) for v in self.unresolved.values())

    @property
    def ok(self) -> bool:
        return not self.errors and not self.unresolved

    def lines(self) -> list[str]:
        out = [f"{self.total} questions: {self.answerable} answerable, {self.unanswerable} unanswerable"]
        out += [f"error: {e}" for e in self.errors]
        for qid, ids in self.unresolved.items():
            out += [f"unresolved gold Passage_ID {pid} (question {qid})" for pid in ids]
        if self.unresolved:
            out.append(f"total unresolved gold Passage_IDs: {self.unresolved_count}")
        return out


def validate(questions: Sequence[EvalQuestion], passage_ids: set[str],
             min_questions: int = MIN_QUESTIONS, min_unanswerable: int = MIN_UNANSWERABLE) -> ValidationReport:
    unanswerable = [q for q in questions if not q.answerable]
    report = ValidationReport(total=len(questions), answerable=len(questions) - len(unanswerable),
                              unanswerable=len(unanswerable))
    if len(questions) < min_questions:
        report.errors.append(f"{len(questions)} questions, need at least {min_questions}")
    if len(unanswerable) < min_unanswerable:
        report.errors.append(f"{len(unanswerable)} unanswerable questions, need at least {min_unanswerable}")
    seen: set[str] = set()
    for q in questions:
        if q.question_id in seen:
            report.errors.append(f"duplicate question_id {q.question_id}")
        seen.add(q.question_id)
        if not q.question.strip():
            report.errors.append(f"question {q.question_id} has empty text")
        if q.answerable and not q.gold_passage_ids:
            report.errors.append(f"answerable question {q.question_id} has no gold Passage_ID")
        if not q.answerable and q.gold_passage_ids:
            report.errors.append(f"unanswerable question {q.question_id} carries gold Passage_IDs")
        missing = [pid for pid in q.gold_passage_ids if pid not in passage_ids]
        if missing:
            report.unresolved[q.question_id] = missing
    return report


def ci_subset(questions: Sequence[EvalQuestion], n_answerable: int, n_unanswerable: int) -> list[EvalQuestion]:
    """The documented CI subset: the first ``n`` answerable and unanswerable questions in file order.

    ``build_evalset.py`` writes the Evaluation_Set in a seeded random order, so file order is
    already a fixed random sample.
    """
    ans = [q for q in questions if q.answerable][:n_answerable]
    una = [q for q in questions if not q.answerable][:n_unanswerable]
    if len(ans) < n_answerable or len(una) < n_unanswerable:
        raise EvalSetError(f"evaluation set too small for a {n_answerable}+{n_unanswerable} subset")
    chosen = {q.question_id for q in ans + una}
    return [q for q in questions if q.question_id in chosen]
