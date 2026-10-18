"""Feature: grounded-qa-service, Properties 5 and 6 (retrieval ranking and empty retrieval)."""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gqa.ingest import Passage
from gqa.retrieval import RagLiteBackend, RagLiteRetriever, RetrievalMode

VOCAB = ["river", "castle", "norman", "bowl", "energy", "plant", "france", "england", "game", "light", "king", "city"]
FOREIGN = ["zzqx", "blorp", "quux", "vrrk", "snarf"]

passage_texts = st.lists(st.lists(st.sampled_from(VOCAB), min_size=1, max_size=12).map(" ".join), min_size=1, max_size=10)
queries = st.lists(st.sampled_from(VOCAB + FOREIGN), min_size=1, max_size=5).map(" ".join)
modes = st.sampled_from([RetrievalMode.BM25, RetrievalMode.HYBRID])


def _backend(texts):
    passages = [Passage(f"d{i}:0:{i:08x}", f"d{i}", 0, t) for i, t in enumerate(texts)]
    # Duplicate one passage's id on purpose so de-duplication is exercised.
    if len(passages) > 2:
        passages[-1] = Passage(passages[0].passage_id, "dup", 0, passages[-1].text)
    return RagLiteBackend(passages, use_embeddings=False, use_reranker=False, candidates=10, check_pins=False)


@settings(max_examples=100, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(passage_texts, queries, modes, st.integers(1, 12))
def test_property_5_ranking_order_and_no_duplicates(texts, query, mode, k):
    """Feature: grounded-qa-service, Property 5: Retrieval ranking order and no duplicates."""
    hits = RagLiteRetriever(mode, _backend(texts)).retrieve(query, k)
    scores = [h.score for h in hits]
    ids = [h.passage_id for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert len(ids) == len(set(ids))
    assert len(hits) <= k


@settings(max_examples=100, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(passage_texts, st.lists(st.sampled_from(FOREIGN), min_size=1, max_size=4).map(" ".join), modes, st.integers(1, 12))
def test_property_6_empty_retrieval_on_no_match(texts, query, mode, k):
    """Feature: grounded-qa-service, Property 6: Empty retrieval on no match."""
    assert RagLiteRetriever(mode, _backend(texts)).retrieve(query, k) == []
