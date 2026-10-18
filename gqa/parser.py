"""AnswerParser: split model output into sentences and cited Passage_IDs (Req 4.3).

Citation syntax: a Passage_ID in square brackets, ``[a01p002:0:1a2b3c4d]``; several IDs may share
one bracket separated by commas or semicolons. Bracketed text that is not a well-formed
Passage_ID is not a citation. A citation written after a sentence's final period belongs to
that sentence. Each sentence is treated as one factual claim (Req 5.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from gqa.answer import REFUSAL_TEXT

PASSAGE_ID = re.compile(r"^[A-Za-z0-9_\-]+:\d+:[0-9a-f]{8}$")
_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_BOUNDARY = re.compile(r"[.!?]\s+")
_LEADING_CITATIONS = re.compile(r"^((?:\[[^\[\]]*\]\s*)+)(.*)$", re.S)
# A period after these does not end a sentence ("U.S. Army", "St. Louis").
_ABBREVIATION = re.compile(r"(?:\b(?:[A-Za-z]\.){1,3}|\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|No|Mt|Ft|Gen|Gov|Sen|Rep|Inc|Co|Corp|Ltd)\.)$")
_WORD = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class Sentence:
    text: str
    cited_ids: frozenset[str]


@dataclass(frozen=True)
class ParsedAnswer:
    text: str
    cited_ids: frozenset[str]
    sentences: tuple[Sentence, ...]
    refused: bool


def citations_in(text: str) -> frozenset[str]:
    ids = set()
    for inner in _BRACKET.findall(text):
        for part in re.split(r"[,;]", inner):
            part = part.strip()
            if PASSAGE_ID.match(part):
                ids.add(part)
    return frozenset(ids)


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    parts, start = [], 0
    for m in _BOUNDARY.finditer(text):
        head = text[start:m.start() + 1]
        if _ABBREVIATION.search(head):
            continue
        parts.append(head)
        start = m.end()
    parts.append(text[start:])
    sentences: list[str] = []
    for part in (p.strip() for p in parts):
        if not part:
            continue
        lead = _LEADING_CITATIONS.match(part)
        if lead and sentences:
            # Citations written after the period belong to the sentence before them.
            sentences[-1] = f"{sentences[-1]} {lead.group(1).strip()}"
            part = lead.group(2).strip()
            if not part:
                continue
        if sentences and not _WORD.search(_BRACKET.sub("", part)):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return sentences


def is_refusal(text: str) -> bool:
    """The reply starts with the refusal phrase, or contains it and cites nothing."""
    core = " ".join(re.sub(r"[^a-z ]", " ", text.lower()).split())
    return core.startswith(REFUSAL_TEXT) or (REFUSAL_TEXT in core and not citations_in(text))


def parse_answer(raw: str) -> ParsedAnswer:
    text = raw.strip()
    sentences = tuple(Sentence(s, citations_in(s)) for s in split_sentences(text))
    cited = frozenset().union(*(s.cited_ids for s in sentences)) if sentences else frozenset()
    return ParsedAnswer(text=text, cited_ids=cited, sentences=sentences, refused=bool(text) and is_refusal(text))
