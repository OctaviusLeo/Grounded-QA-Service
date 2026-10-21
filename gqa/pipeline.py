"""AnsweringPipeline: retrieve -> confidence -> (refuse | prompt -> model -> parse -> validate).

Used unchanged by the service, the evaluation harness, the experiments, the Injection_Test, and
the CI gate (Req 4.1-4.5, 5.3). ``run`` returns a :class:`Trace` with everything the evaluation
needs; ``answer`` returns only the final :class:`Answer`. ``stream`` emits the answer one
validated sentence at a time, so a sentence that fails citation validation is never sent.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from gqa.answer import REFUSAL, REFUSAL_TEXT, Answer
from gqa.confidence import estimate_confidence, should_refuse
from gqa.parser import Sentence, citations_in, is_refusal, parse_answer, split_sentences
from gqa.prompt import PromptBuilder, PromptMessages
from gqa.retrieval import RagLiteRetriever, RetrievedPassage
from gqa.validate import CitationValidator, ValidationResult, sentence_is_valid

__all__ = ["REFUSAL_TEXT", "Answer", "AnsweringPipeline", "StreamEvent", "Trace"]


class ChatModel(Protocol):
    model: str

    def complete(self, messages: Any) -> Any: ...

    def stream(self, messages: Any) -> Iterator[str]: ...


@dataclass
class Trace:
    query: str
    mode: str
    candidates: list[RetrievedPassage]
    context: list[RetrievedPassage]
    confidence: float
    threshold: float
    answer: Answer = REFUSAL
    refused_before_model: bool = False
    raw_output: str | None = None
    validation: ValidationResult | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    model: str = ""

    @property
    def refusal_source(self) -> str | None:
        if not self.answer.refused:
            return None
        if self.refused_before_model:
            return "confidence"
        if self.validation is not None and self.validation.accepted:
            return "model"
        return "validation"


@dataclass(frozen=True)
class StreamEvent:
    kind: str  # "chunk" or "done"
    text: str = ""
    answer: Answer | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class AnsweringPipeline:
    def __init__(self, retriever: RagLiteRetriever, prompt_builder: PromptBuilder, model_client: ChatModel,
                 validator: CitationValidator, threshold: float, k: int, seed: int, candidates: int | None = None) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.model_client = model_client
        self.validator = validator
        self.threshold = threshold
        self.k = k
        self.seed = seed
        self.candidates = max(k, candidates or k)

    @property
    def mode(self) -> str:
        return self.retriever.mode.value

    def _retrieve(self, query: str) -> tuple[list[RetrievedPassage], list[RetrievedPassage], float]:
        candidates = self.retriever.retrieve(query, self.candidates)
        context = candidates[: self.k]
        return candidates, context, estimate_confidence(context, self.retriever.mode)

    def _messages(self, query: str, context: Sequence[RetrievedPassage]) -> PromptMessages:
        return self.prompt_builder.build(query, self.retriever.passages_for(context))

    def run(self, query: str) -> Trace:
        t0 = time.perf_counter()
        candidates, context, confidence = self._retrieve(query)
        t1 = time.perf_counter()
        trace = Trace(query, self.mode, candidates, context, confidence, self.threshold,
                      retrieval_ms=(t1 - t0) * 1000.0, model=self.model_client.model)
        if should_refuse(confidence, self.threshold):
            trace.refused_before_model = True
            trace.total_ms = (time.perf_counter() - t0) * 1000.0
            return trace
        completion = self.model_client.complete(self._messages(query, context))
        t2 = time.perf_counter()
        trace.raw_output = completion.text
        trace.prompt_tokens = completion.prompt_tokens
        trace.completion_tokens = completion.completion_tokens
        trace.validation = self.validator.validate(parse_answer(completion.text), {r.passage_id for r in context})
        trace.answer = trace.validation.answer
        trace.generation_ms = (t2 - t1) * 1000.0
        trace.total_ms = (time.perf_counter() - t0) * 1000.0
        return trace

    def answer(self, query: str) -> Answer:
        return self.run(query).answer

    def stream(self, query: str) -> Iterator[StreamEvent]:
        """Yield validated sentences as ``chunk`` events, then one ``done`` event with the Answer."""
        candidates, context, confidence = self._retrieve(query)
        meta = {"confidence": confidence, "threshold": self.threshold, "mode": self.mode}
        if should_refuse(confidence, self.threshold):
            yield StreamEvent("chunk", REFUSAL_TEXT)
            yield StreamEvent("done", answer=REFUSAL, meta={**meta, "refusal_source": "confidence"})
            return
        retrieved = {r.passage_id for r in context}
        kept: list[Sentence] = []
        dropped = 0
        buffer, done_count = "", 0

        def settle(text: str) -> Iterator[StreamEvent]:
            nonlocal dropped
            sentence = Sentence(text, citations_in(text))
            # Before the first chunk, refuse exactly when the non-streaming path would: a reply that
            # opens with the refusal phrase is a refusal even if a later sentence cites a passage.
            if sentence_is_valid(sentence, retrieved) and (kept or not is_refusal(buffer)):
                kept.append(sentence)
                yield StreamEvent("chunk", text if len(kept) == 1 else " " + text)
            else:
                dropped += 1

        for delta in self.model_client.stream(self._messages(query, context)):
            buffer += delta
            if buffer.count("[") != buffer.count("]"):
                continue  # a citation is still arriving
            sentences = split_sentences(buffer)
            for text in sentences[done_count:-1]:
                yield from settle(text)
            done_count = max(done_count, len(sentences) - 1)
        for text in split_sentences(buffer)[done_count:]:
            yield from settle(text)
        if not kept:
            yield StreamEvent("chunk", REFUSAL_TEXT)
            yield StreamEvent("done", answer=REFUSAL, meta={**meta, "refusal_source": "model" if parse_answer(buffer).refused else "validation"})
            return
        cited = tuple(sorted(frozenset().union(*(s.cited_ids for s in kept))))
        answer = Answer(" ".join(s.text for s in kept), cited, False)
        yield StreamEvent("done", answer=answer, meta={**meta, "dropped_sentences": dropped})
