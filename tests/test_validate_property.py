"""Feature: grounded-qa-service, Properties 1 and 2 (citation soundness, no uncited claims)."""

from hypothesis import given, settings
from hypothesis import strategies as st

from gqa.answer import REFUSAL_TEXT
from gqa.parser import parse_answer
from gqa.validate import CitationValidator

ALPHABET = [f"a0{i}p00{i}:{i % 3}:{i:08x}" for i in range(8)]
WORDS = ["Paris", "is", "the", "capital", "rivers", "flow", "north", "in", "1066"]

claim = st.tuples(
    st.lists(st.sampled_from(WORDS), min_size=1, max_size=6).map(" ".join),
    st.frozensets(st.sampled_from(ALPHABET), max_size=3),
)
answers = st.lists(claim, min_size=1, max_size=4)


def render(claims):
    out = []
    for words, ids in claims:
        cites = "".join(f" [{i}]" for i in sorted(ids))
        out.append(f"{words}{cites}.")
    return " ".join(out)


@settings(max_examples=100, derandomize=True)
@given(answers, st.frozensets(st.sampled_from(ALPHABET), max_size=8))
def test_property_1_citation_soundness(claims, retrieved):
    """Feature: grounded-qa-service, Property 1: Citation soundness."""
    parsed = parse_answer(render(claims))
    cited = frozenset().union(*(ids for _, ids in claims))
    no_uncited_claim = all(ids for _, ids in claims)
    result = CitationValidator().validate(parsed, retrieved)
    assert parsed.cited_ids == cited
    assert result.accepted == (cited <= retrieved and no_uncited_claim)
    assert result.offending_ids == cited - retrieved


@settings(max_examples=100, derandomize=True)
@given(answers, st.frozensets(st.sampled_from(ALPHABET), max_size=8), st.sampled_from(ALPHABET))
def test_property_2_no_uncited_claims_after_validation(claims, retrieved, outside):
    """Feature: grounded-qa-service, Property 2: No uncited claims after validation."""
    # Force a failing answer: one claim cites an ID outside the retrieved set.
    retrieved = retrieved - {outside}
    claims = claims + [("this cites something unretrieved", frozenset({outside}))]
    result = CitationValidator().validate(parse_answer(render(claims)), retrieved)
    assert not result.accepted
    final = result.answer
    if final.refused:
        assert final.text == REFUSAL_TEXT and final.cited_ids == ()
    else:
        assert set(final.cited_ids) <= retrieved and final.cited_ids
        for sentence in parse_answer(final.text).sentences:
            assert sentence.cited_ids and sentence.cited_ids <= retrieved
