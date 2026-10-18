import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gqa.answer import REFUSAL, REFUSAL_TEXT
from gqa.config import load_config
from gqa.model_client import ModelTimeoutError
from gqa.parser import parse_answer
from gqa.pipeline import AnsweringPipeline
from gqa.prompt import PromptBuilder
from gqa.retrieval import RagLiteRetriever, RetrievalMode
from gqa.runtime import make_client, make_pipeline
from gqa.validate import CitationValidator
from tests import fakes

Q = "Where was Super Bowl 50 played?"


def _pipe(model, threshold=0.0, mode=RetrievalMode.HYBRID):
    return AnsweringPipeline(RagLiteRetriever(mode, fakes.backend()), PromptBuilder(), model,
                             CitationValidator(), threshold=threshold, k=2, seed=1, candidates=4)


def test_answered_path_returns_cited_answer_and_trace():
    model = fakes.FakeModel()
    trace = _pipe(model).run(Q)
    assert trace.answer.text == "Santa Clara [d1:0:00000001]." and trace.answer.cited_ids == ("d1:0:00000001",)
    assert trace.validation.accepted and trace.refusal_source is None
    assert trace.context[0].passage_id == "d1:0:00000001" and len(trace.context) <= 2
    assert trace.prompt_tokens == 100 and trace.total_ms >= trace.retrieval_ms >= 0
    assert "[d1:0:00000001]" in model.calls[0].user


def test_unretrieved_citation_is_corrected_or_refused():
    model = fakes.FakeModel("Santa Clara [d1:0:00000001]. Also Normandy [d0:0:00000000].")
    trace = _pipe(model).run(Q)  # d0 shares no word with the query, so it is not retrieved
    assert not trace.validation.accepted and trace.validation.offending_ids == {"d0:0:00000000"}
    assert trace.answer.cited_ids == ("d1:0:00000001",)
    bad = _pipe(fakes.FakeModel("Normandy [d0:0:00000000].")).run(Q)
    assert bad.answer == REFUSAL and bad.refusal_source == "validation"


def test_model_refusal_and_confidence_refusal():
    t = _pipe(fakes.FakeModel(REFUSAL_TEXT)).run(Q)
    assert t.answer == REFUSAL and t.refusal_source == "model"
    model = fakes.FakeModel()
    t2 = _pipe(model, threshold=1.0).run("zzqx blorp")
    assert t2.answer == REFUSAL and t2.refused_before_model and t2.refusal_source == "confidence" and model.calls == []
    assert t2.confidence == 0.0


def test_model_errors_propagate():
    with pytest.raises(ModelTimeoutError):
        _pipe(fakes.FakeModel(error=ModelTimeoutError("slow"))).answer(Q)


def _collect(events):
    events = list(events)
    chunks = [e.text for e in events if e.kind == "chunk"]
    done = [e for e in events if e.kind == "done"]
    assert len(done) == 1 and events[-1] is done[0]
    return chunks, done[0]


def test_stream_emits_only_validated_sentences():
    reply = "Santa Clara hosted it. [d1:0:00000001] It was in 2016 [d9:0:99999999]. It was a game [d1:0:00000001]."
    chunks, done = _collect(_pipe(fakes.FakeModel(reply)).stream(Q))
    assert chunks == ["Santa Clara hosted it. [d1:0:00000001]", " It was a game [d1:0:00000001]."]
    assert done.answer.text == "".join(chunks) and done.meta["dropped_sentences"] == 1
    # Streaming and non-streaming agree on the final answer.
    assert done.answer == _pipe(fakes.FakeModel(reply)).answer(Q)


def test_stream_refusals():
    chunks, done = _collect(_pipe(fakes.FakeModel(), threshold=1.0).stream("zzqx"))
    assert chunks == [REFUSAL_TEXT] and done.answer == REFUSAL and done.meta["refusal_source"] == "confidence"
    chunks, done = _collect(_pipe(fakes.FakeModel(REFUSAL_TEXT + ".")).stream(Q))
    assert chunks == [REFUSAL_TEXT] and done.meta["refusal_source"] == "model"
    chunks, done = _collect(_pipe(fakes.FakeModel("Uncited claim.")).stream(Q))
    assert done.answer == REFUSAL and done.meta["refusal_source"] == "validation"


def test_stream_waits_for_complete_citations():
    chunks = ["Santa Clara. [d1:0", ":00000001] Next", " sentence [d1:0:00000001]."]
    out, done = _collect(_pipe(fakes.FakeModel(chunks=chunks)).stream(Q))
    assert out == ["Santa Clara. [d1:0:00000001]", " Next sentence [d1:0:00000001]."]


replies = st.lists(st.tuples(st.sampled_from(["Santa Clara", "A game", "It was 2016", "Normandy"]),
                             st.sampled_from(["", " [d1:0:00000001]", " [d0:0:00000000]", " [d1:0:00000001, d3:0:00000003]"]),
                             st.sampled_from([".", "!", ". "])), min_size=1, max_size=4)


@settings(max_examples=100, derandomize=True, deadline=None)
@given(replies, st.integers(1, 7))
def test_stream_matches_non_streaming_answer(parts, step):
    reply = " ".join(f"{w}{c}{p}".strip() for w, c, p in parts)
    chunks = [reply[i:i + step] for i in range(0, len(reply), step)]
    _, done = _collect(_pipe(fakes.FakeModel(reply, chunks=chunks)).stream(Q))
    expected = _pipe(fakes.FakeModel(reply)).answer(Q)
    assert done.answer == expected or (done.answer.refused and expected.refused)
    assert done.answer.refused or set(done.answer.cited_ids) <= {"d1:0:00000001", "d3:0:00000003"}


def test_runtime_factory_uses_config():
    cfg = load_config("config/default.yaml", env={})
    p = make_pipeline(cfg, fakes.backend(), "bm25", "granite3.3:2b", client=fakes.FakeModel())
    assert p.k == cfg.k and p.threshold == cfg.threshold_for("bm25") and p.mode == "bm25"
    client = make_client(cfg, "granite3.3:2b")
    assert client.model == "granite3.3:2b" and client.seed == cfg.seed
    client.close()
    assert parse_answer(p.answer(Q).text).cited_ids <= {"d1:0:00000001"}
