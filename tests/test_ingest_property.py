"""Feature: grounded-qa-service, Property 4: Passage_ID stability and uniqueness."""

from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from gqa.ingest import Chunker, Document, find_duplicate_ids

words = st.sampled_from(["alpha", "beta", "gamma", "delta", "river", "castle", "Norman", "1066", "é", "\t", "\n"])
texts = st.lists(words, min_size=0, max_size=120).map(" ".join)
docs = st.lists(
    st.builds(Document, doc_id=st.sampled_from(["a00p000", "a00p001", "a01p000", "b"]), title=st.just("t"), text=texts),
    min_size=0,
    max_size=6,
)


@settings(max_examples=100, derandomize=True)
@given(docs, st.sampled_from([(40, 10), (80, 0), (200, 50)]))
def test_property_4_passage_id_stability_and_uniqueness(documents, sizes):
    """Feature: grounded-qa-service, Property 4: Passage_ID stability and uniqueness."""
    chunker = Chunker(*sizes)
    first = chunker.chunk(documents)
    second = Chunker(*sizes).chunk(list(documents))
    # Stability: chunking twice gives identical id -> text assignments.
    assert [(p.passage_id, p.text) for p in first] == [(p.passage_id, p.text) for p in second]
    # Uniqueness: distinct passages get distinct ids, or every collision is reported with its count.
    counts = Counter(p.passage_id for p in first)
    assert find_duplicate_ids(first) == {pid: n for pid, n in counts.items() if n > 1}
    by_id = {}
    for p in first:
        key = (p.doc_id, p.chunk_index, p.text)
        if p.passage_id in by_id and by_id[p.passage_id] != key:
            assert p.passage_id in find_duplicate_ids(first)
        by_id.setdefault(p.passage_id, key)
