from gqa.answer import REFUSAL
from gqa.parser import parse_answer
from gqa.validate import CitationValidator

A, B, C = "a01p002:0:1a2b3c4d", "a01p003:0:00ff00ff", "a09p009:0:deadbeef"
V = CitationValidator()


def test_accepts_when_all_cited_ids_retrieved():
    r = V.validate(parse_answer(f"Paris is the capital [{A}]. It has rivers [{B}]."), {A, B, C})
    assert r.accepted and not r.corrected and r.offending_ids == frozenset()
    assert r.answer.cited_ids == (A, B) and not r.answer.refused


def test_rejects_when_a_cited_id_was_not_retrieved():
    r = V.validate(parse_answer(f"Paris is the capital [{A}]. It is huge [{C}]."), {A, B})
    assert not r.accepted and r.offending_ids == {C}
    assert C in r.reason
    assert r.corrected and r.answer.text == f"Paris is the capital [{A}]." and r.answer.cited_ids == (A,)


def test_rejects_uncited_claim_and_refuses_when_nothing_valid_remains():
    r = V.validate(parse_answer("Paris is the capital."), {A})
    assert not r.accepted and r.uncited_sentences == ("Paris is the capital.",)
    assert r.answer == REFUSAL
    r2 = V.validate(parse_answer(f"Paris [{C}]."), {A})
    assert not r2.accepted and r2.answer == REFUSAL and r2.offending_ids == {C}


def test_refusal_and_empty_answers():
    r = V.validate(parse_answer("not found in the documents"), {A})
    assert r.accepted and r.answer == REFUSAL
    e = V.validate(parse_answer("   "), {A})
    assert not e.accepted and e.answer == REFUSAL and e.reason == "empty answer"
