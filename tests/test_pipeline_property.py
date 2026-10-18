"""Feature: grounded-qa-service, Property 3: Refusal below confidence threshold."""

from hypothesis import given, settings
from hypothesis import strategies as st

from gqa.answer import REFUSAL_TEXT
from gqa.confidence import estimate_confidence
from gqa.pipeline import AnsweringPipeline
from gqa.prompt import PromptBuilder
from gqa.retrieval import RetrievalMode, RetrievedPassage
from gqa.validate import CitationValidator
from tests import fakes


class StubRetriever:
    def __init__(self, mode, ranked):
        self.mode = mode
        self.ranked = ranked

    def retrieve(self, query, k):
        return self.ranked[:k]

    def passages_for(self, retrieved):
        return [fakes.PASSAGES[int(r.passage_id[1])] for r in retrieved]


scores = st.lists(st.floats(min_value=-15, max_value=15, allow_nan=False), max_size=4).map(lambda xs: sorted(xs, reverse=True))


@settings(max_examples=100, derandomize=True)
@given(scores, st.floats(min_value=0.0, max_value=1.0), st.sampled_from(list(RetrievalMode)))
def test_property_3_refusal_below_confidence_threshold(score_list, threshold, mode):
    """Feature: grounded-qa-service, Property 3: Refusal below confidence threshold."""
    ranked = [RetrievedPassage(fakes.PASSAGES[i].passage_id, s, i) for i, s in enumerate(score_list)]
    model = fakes.FakeModel()
    pipe = AnsweringPipeline(StubRetriever(mode, ranked), PromptBuilder(), model, CitationValidator(),
                             threshold=threshold, k=4, seed=0)
    answer = pipe.answer("question")
    if estimate_confidence(ranked, mode) < threshold:
        assert answer.refused and answer.text == REFUSAL_TEXT and answer.cited_ids == ()
        assert model.calls == []
    else:
        assert len(model.calls) == 1
