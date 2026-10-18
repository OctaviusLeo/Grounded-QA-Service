import pytest

from gqa import hf_models
from gqa.ingest import Passage
from gqa.retrieval import (
    RagLiteBackend,
    RagLiteRetriever,
    RetrievalError,
    RetrievalMode,
    content_terms,
)

TEXTS = [
    "The Normans were a people who gave their name to Normandy in France.",
    "Super Bowl 50 was an American football game played in Santa Clara.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The Norman conquest of England began in 1066 under William.",
]
PASSAGES = [Passage(f"d{i}:0:{i:08x}", f"d{i}", 0, t) for i, t in enumerate(TEXTS)]


@pytest.fixture(scope="module")
def backend():
    return RagLiteBackend(PASSAGES, use_embeddings=False, use_reranker=False, candidates=4)


def test_content_terms_drop_stop_words():
    assert content_terms("Where was THE game played?") == {"game", "played"}


@pytest.mark.parametrize("mode", [RetrievalMode.BM25, RetrievalMode.HYBRID])
def test_relevant_passage_ranked_first(backend, mode):
    r = RagLiteRetriever(mode, backend)
    hits = r.retrieve("Where was Super Bowl 50 played?", k=3)
    assert hits[0].passage_id == "d1:0:00000001"
    assert [h.rank for h in hits] == list(range(len(hits)))
    assert r.passages_for(hits)[0].text == TEXTS[1]


def test_scoring_floor_drops_passages_without_shared_words(backend):
    hits = RagLiteRetriever(RetrievalMode.HYBRID, backend).retrieve("Normans Normandy", k=4)
    assert {h.passage_id for h in hits} == {"d0:0:00000000"}  # d3 says "Norman", not "Normans"


def test_no_match_and_stopword_only_queries_return_empty(backend):
    r = RagLiteRetriever(RetrievalMode.BM25, backend)
    assert r.retrieve("zzqx blorp", k=3) == []
    assert r.retrieve("what is the", k=3) == []
    assert r.retrieve("Normandy", k=0) == []


def test_rerank_requires_cross_encoder(backend):
    with pytest.raises(RetrievalError):
        RagLiteRetriever(RetrievalMode.HYBRID_RERANK, backend).retrieve("Normandy", k=2)
    with pytest.raises(RetrievalError):
        RagLiteBackend([], use_embeddings=False, use_reranker=False)


def test_dense_backend_checks_pins(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(hf_models.PinError):
        RagLiteBackend(PASSAGES, use_embeddings=True, use_reranker=False)


def test_pin_helpers(tmp_path):
    assert len(hf_models.verify_pins(tmp_path)) == 2
    calls = []

    def fake_download(repo_id, revision, cache_dir):
        calls.append((repo_id, revision))
        snap = tmp_path / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
        snap.mkdir(parents=True)

    hf_models.fetch_pinned(tmp_path, download=fake_download)
    assert calls == list(hf_models.PINNED.items())
    assert hf_models.verify_pins(tmp_path) == []
    hf_models.require_pins(tmp_path)
    repo = next(iter(hf_models.PINNED))
    (tmp_path / ("models--" + repo.replace("/", "--")) / "refs" / "main").write_text("other")
    assert "cached main is other" in hf_models.verify_pins(tmp_path)[0]


def test_hub_cache_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert hf_models.hub_cache() == tmp_path / "hub"


@pytest.mark.hf
def test_hybrid_rerank_with_real_models():
    b = RagLiteBackend(PASSAGES, use_embeddings=True, use_reranker=True, candidates=4)
    hits = RagLiteRetriever(RetrievalMode.HYBRID_RERANK, b).retrieve("Where was Super Bowl 50 played?", k=2)
    # Only d1 shares a content word with the query; the cross-encoder logit is strongly positive.
    assert [h.passage_id for h in hits] == ["d1:0:00000001"] and hits[0].score > 0
