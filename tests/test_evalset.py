import json

import pytest

from gqa.evalset import (
    EvalQuestion,
    EvalSetError,
    ci_subset,
    load_questions,
    validate,
    write_questions,
)

IDS = {f"a00p{i:03d}:0:0000000{i % 10}" for i in range(200)}
ID_LIST = sorted(IDS)


def _answerable(i):
    return EvalQuestion(f"q{i}", f"Question {i}?", "gold", (ID_LIST[i % len(ID_LIST)],), True, ("gold",), "a00p000")


def _unanswerable(i):
    return EvalQuestion(f"u{i}", f"Unanswerable {i}?", None, (), False)


def _set(n_ans=130, n_una=20):
    return [_answerable(i) for i in range(n_ans)] + [_unanswerable(i) for i in range(n_una)]


def test_valid_set_passes():
    report = validate(_set(), IDS)
    assert report.ok and report.total == 150 and report.unanswerable == 20
    assert report.lines()[0] == "150 questions: 130 answerable, 20 unanswerable"


def test_count_thresholds():
    assert any("need at least 150" in e for e in validate(_set(129, 20), IDS).errors)
    assert any("need at least 20" in e for e in validate(_set(140, 19), IDS).errors)
    assert validate(_set(10, 2), IDS, min_questions=12, min_unanswerable=2).ok


def test_unanswerable_labeling_rules():
    qs = _set()
    qs[0] = EvalQuestion("q0", "Q?", "g", (), True)
    qs[-1] = EvalQuestion("u19", "U?", None, (ID_LIST[0],), False)
    errors = validate(qs, IDS).errors
    assert "answerable question q0 has no gold Passage_ID" in errors
    assert "unanswerable question u19 carries gold Passage_IDs" in errors


def test_unresolved_gold_ids_are_each_reported_with_total():
    qs = _set()
    qs[0] = EvalQuestion("q0", "Q?", "g", ("missing:0:aaaaaaaa", "missing:1:bbbbbbbb"), True)
    qs[1] = EvalQuestion("q1", "Q?", "g", (ID_LIST[1], "gone:0:cccccccc"), True)
    report = validate(qs, IDS)
    assert not report.ok
    assert report.unresolved == {"q0": ["missing:0:aaaaaaaa", "missing:1:bbbbbbbb"], "q1": ["gone:0:cccccccc"]}
    assert report.unresolved_count == 3
    lines = report.lines()
    assert "unresolved gold Passage_ID gone:0:cccccccc (question q1)" in lines
    assert lines[-1] == "total unresolved gold Passage_IDs: 3"


def test_duplicate_and_empty_questions():
    qs = _set()
    qs.append(_answerable(0))
    qs.append(EvalQuestion("blank", "  ", "g", (ID_LIST[0],), True))
    errors = validate(qs, IDS).errors
    assert "duplicate question_id q0" in errors and "question blank has empty text" in errors


def test_round_trip_and_malformed_lines(tmp_path):
    path = tmp_path / "e.jsonl"
    write_questions(path, _set(3, 1))
    assert load_questions(path) == _set(3, 1)
    with path.open("a") as fh:
        fh.write(json.dumps({"question": "no id"}) + "\n")
    with pytest.raises(EvalSetError, match="line 5"):
        load_questions(path)
    with pytest.raises(EvalSetError, match="does not exist"):
        load_questions(tmp_path / "nope.jsonl")


def test_ci_subset_takes_first_in_file_order():
    qs = _set(30, 10)
    sub = ci_subset(qs, 4, 2)
    assert [q.question_id for q in sub] == ["q0", "q1", "q2", "q3", "u0", "u1"]
    with pytest.raises(EvalSetError):
        ci_subset(qs, 40, 2)
