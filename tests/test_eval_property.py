"""Feature: grounded-qa-service, Property 9: Determinism under fixed seed."""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from gqa import eval as harness
from gqa.evalset import EvalQuestion

IDS = [f"p{i}:0:{i:08x}" for i in range(8)]


def _record(i, answerable, refused, cited, candidates, label):
    return harness.Record(
        question_id=f"q{i}", question="q", answerable=answerable,
        answer="not found in the documents" if refused else "Some answer text [" + IDS[0] + "].",
        cited_ids=[] if refused else cited, refused=refused, refusal_source="model" if refused else None,
        candidate_ids=candidates, context_ids=candidates[:5], confidence=0.5, latency_ms=10.0 + i,
        retrieval_ms=1.0, generation_ms=9.0, prompt_tokens=10, completion_tokens=3, raw_output="x",
        judge_label=None if refused else label,
    )


rows = st.lists(st.tuples(st.booleans(), st.booleans(), st.lists(st.sampled_from(IDS), max_size=3, unique=True),
                          st.lists(st.sampled_from(IDS), max_size=8, unique=True), st.sampled_from([0, 1, None]),
                          st.frozensets(st.sampled_from(IDS), min_size=1, max_size=2)), min_size=1, max_size=12)


@settings(max_examples=100, derandomize=True, deadline=None)
@given(rows, st.integers(0, 2**31 - 1))
def test_property_9_determinism_under_fixed_seed(data, seed):
    """Feature: grounded-qa-service, Property 9: Determinism under fixed seed."""
    records, questions = [], []
    for i, (answerable, refused, cited, candidates, label, gold) in enumerate(data):
        records.append(_record(i, answerable, refused, cited, candidates, label))
        questions.append(EvalQuestion(f"q{i}", "q", "Some answer" if answerable else None,
                                      tuple(sorted(gold)) if answerable else (), answerable,
                                      ("Some answer",) if answerable else ()))
    first = harness.compute_metrics(records, questions, seed)
    second = harness.compute_metrics(records, questions, seed)
    assert first.keys() == second.keys()
    for name in first:
        a, b = first[name], second[name]
        assert a.keys() == b.keys()
        for key in a:
            if isinstance(a[key], float):
                assert math.isclose(a[key], b[key], abs_tol=1e-6)
            else:
                assert a[key] == b[key]  # ints and categoricals exactly equal
    assert harness.counts(records) == harness.counts(records)
    assert harness.latency_summary(records) == harness.latency_summary(records)
