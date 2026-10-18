"""Test doubles shared by the pipeline, harness, experiments, and service tests."""

from __future__ import annotations

from gqa.ingest import Passage
from gqa.model_client import Completion
from gqa.retrieval import RagLiteBackend

TEXTS = [
    "The Normans were a people who gave their name to Normandy in France.",
    "Super Bowl 50 was an American football game played in Santa Clara.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The Norman conquest of England began in 1066 under William.",
]
PASSAGES = [Passage(f"d{i}:0:{i:08x}", f"d{i}", 0, t, 0, f"Title {i}") for i, t in enumerate(TEXTS)]


def backend(**kw):
    kw.setdefault("candidates", 4)
    return RagLiteBackend(PASSAGES, use_embeddings=False, use_reranker=False, check_pins=False, **kw)


class FakeModel:
    """Answers with a fixed reply (or a function of the prompt) and records every call."""

    def __init__(self, reply="Santa Clara [d1:0:00000001].", model="fake-model", chunks=None, error=None):
        self.reply = reply
        self.model = model
        self.chunks = chunks
        self.error = error
        self.calls = []

    def _text(self, messages):
        return self.reply(messages) if callable(self.reply) else self.reply

    def complete(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        text = self._text(messages)
        return Completion(text, prompt_tokens=100, completion_tokens=len(text.split()), elapsed_s=0.01)

    def stream(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        text = self._text(messages)
        yield from (self.chunks if self.chunks is not None else [text[i:i + 3] for i in range(0, len(text), 3)])
