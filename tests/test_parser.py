from gqa.parser import citations_in, is_refusal, parse_answer, split_sentences

A, B, C = "a01p002:0:1a2b3c4d", "a01p003:1:00ff00ff", "x_y-z:12:abcdef01"


def test_zero_one_and_many_citations():
    assert parse_answer("Paris is in France.").cited_ids == frozenset()
    one = parse_answer(f"Paris is in France [{A}].")
    assert one.cited_ids == {A} and len(one.sentences) == 1
    many = parse_answer(f"Paris [{A}, {B}]. It is old [{C}; {A}].")
    assert many.cited_ids == {A, B, C}
    assert [s.cited_ids for s in many.sentences] == [frozenset({A, B}), frozenset({C, A})]


def test_malformed_markers_are_not_citations():
    assert citations_in("[a01p002] [doc 1] [a01p002:0:XYZ] [a01p002:0:1a2b3c4] []") == frozenset()
    parsed = parse_answer("Paris is the capital [1].")
    assert parsed.cited_ids == frozenset() and parsed.sentences[0].cited_ids == frozenset()


def test_citation_after_period_attaches_to_sentence():
    assert split_sentences(f"Paris is the capital. [{A}] It is big. [{B}]") == [f"Paris is the capital. [{A}]", f"It is big. [{B}]"]
    assert split_sentences(f"One. Two! Three? [{A}]") == ["One.", "Two!", f"Three? [{A}]"]
    assert split_sentences("  ") == []
    assert split_sentences(f"The U.S. Army won in St. Louis [{A}]. Then it left. [{B}]") == [f"The U.S. Army won in St. Louis [{A}].", f"Then it left. [{B}]"]
    assert split_sentences(f"[{A}] Leading citation.") == [f"[{A}] Leading citation."]
    assert split_sentences(f"Yes. [{A}] [{B}]") == [f"Yes. [{A}] [{B}]"]
    assert split_sentences("Done. ...") == ["Done. ..."]


def test_refusal_detection():
    assert parse_answer("not found in the documents").refused
    assert parse_answer("Not found in the documents.").refused
    assert parse_answer("Sorry, the answer is not found in the documents.").refused
    assert not parse_answer(f"Paris [{A}]. Other details are not found in the documents.").refused
    assert not parse_answer("").refused
    assert is_refusal("NOT FOUND IN THE DOCUMENTS!")
    assert parse_answer(f"not found in the documents\n\n[Explanation] The passage [{A}] does not say.").refused
