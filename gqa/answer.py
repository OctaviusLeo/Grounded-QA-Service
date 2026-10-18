"""The Answer record shared by the validator, the pipeline, and the service (Req 4.3-4.5, 5.3)."""

from __future__ import annotations

import re
from dataclasses import dataclass

REFUSAL_TEXT = "not found in the documents"

_CITATION = re.compile(r"\s*\[[^\[\]]*\]")


@dataclass(frozen=True)
class Answer:
    text: str
    cited_ids: tuple[str, ...]
    refused: bool  # True => text == REFUSAL_TEXT and cited_ids == ()


REFUSAL = Answer(text=REFUSAL_TEXT, cited_ids=(), refused=True)


def strip_citations(text: str) -> str:
    """Answer text without its ``[passage_id]`` markers (used for exact match and F1)."""
    return _CITATION.sub("", text).strip()
