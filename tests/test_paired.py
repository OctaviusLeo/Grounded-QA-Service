import pytest

from gqa import results_table as rt
from gqa.eval import Record, compute_metrics
from gqa.evalset import EvalQuestion
from gqa.paired import compare, paired_difference, per_question
from gqa.repro import Stamp, write_json

QUESTIONS = [EvalQuestion(f"q{i}", f"question {i}", "a", (f"g{i}",), True, ("a",)) for i in range(6)] + [
    EvalQuestion("u0", "unanswerable", None, (), False)]


def _record(qid, candidates, cited, refused=False, answerable=True):
    return Record(qid, "question", answerable, "" if refused else "a", cited, refused, "model" if refused else None,
                  candidates, candidates[:5], 0.9, 1.0, 0.5, 0.5, 10, 2, None)


def _records(hits, answered):
    out = [_record(f"q{i}", ([f"g{i}"] if hit else []) + [f"x{j}" for j in range(5)],
                   [] if not ans else [f"g{i}" if hit else "x0"], refused=not ans)
           for i, (hit, ans) in enumerate(zip(hits, answered, strict=True))]
    return out + [_record("u0", ["x0"], [], refused=True, answerable=False)]


A = _records([1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 1])
B = _records([1, 0, 0, 1, 0, 0], [1, 1, 1, 0, 1, 1])


def test_per_question_skips_unanswerable_and_marks_refusals():
    values = per_question(A, QUESTIONS)
    assert values["recall@5"] == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert values["citation_precision"] == [1.0, 1.0, 1.0, 1.0, None, 0.0]


def test_point_estimate_is_the_difference_of_reported_metrics():
    out = compare(A, B, QUESTIONS, seed=1234)
    ma, mb = compute_metrics(A, QUESTIONS, 1234, judged=False), compute_metrics(B, QUESTIONS, 1234, judged=False)
    for m in ("recall@5", "citation_precision"):
        assert out[m]["point"] == pytest.approx(ma[m]["point"] - mb[m]["point"])
        assert out[m]["lower"] <= out[m]["point"] <= out[m]["upper"]
    assert (out["citation_precision"]["n_a"], out["citation_precision"]["n_b"]) == (5, 5)
    assert compare(A, B, QUESTIONS, seed=1234) == out  # deterministic for a seed


def test_identical_configurations_differ_by_exactly_zero():
    ci = paired_difference([1.0, 0.0, None, 0.5], [1.0, 0.0, None, 0.5], seed=7)
    assert (ci.point, ci.lower, ci.upper) == (0.0, 0.0, 0.0)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="differ in length"):
        paired_difference([1.0], [1.0, 0.0], seed=1)


def test_results_table_section(tmp_path):
    assert "Not computed: " in "\n".join(rt.paired_section(tmp_path / "paired.json"))
    write_json(tmp_path / "paired.json", {"reference": "r", "comparisons": {"o": compare(A, B, QUESTIONS, seed=1234)}},
               Stamp("python scripts/paired_comparison.py", 1234, "a" * 40))
    text = "\n".join(rt.paired_section(tmp_path / "paired.json"))
    assert "| r − o | Recall@5 | +0.500 [" in text and "| 5 / 5 |" in text
