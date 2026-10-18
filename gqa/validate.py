"""CitationValidator: no answer cites what was not retrieved, and no claim goes uncited (Req 5.1-5.4).

An answer is accepted only if every cited Passage_ID is in the retrieved set and every sentence
carries at least one citation. A rejected answer is corrected by keeping only the sentences
whose citations are non-empty and all retrieved; if none remain, the result is a Refusal. Either
way, the response that leaves the validator contains no uncited claim (Req 5.3).
"""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass

from gqa.answer import REFUSAL, Answer
from gqa.parser import ParsedAnswer, Sentence


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    answer: Answer
    offending_ids: frozenset[str]  # cited but not retrieved (Req 5.2)
    uncited_sentences: tuple[str, ...]
    reason: str

    @property
    def corrected(self) -> bool:
        return not self.accepted and not self.answer.refused


def sentence_is_valid(sentence: Sentence, retrieved_ids: Set[str]) -> bool:
    return bool(sentence.cited_ids) and sentence.cited_ids <= retrieved_ids


def _answer_from(sentences: list[Sentence]) -> Answer:
    cited = sorted(frozenset().union(*(s.cited_ids for s in sentences)))
    return Answer(text=" ".join(s.text for s in sentences), cited_ids=tuple(cited), refused=False)


class CitationValidator:
    def validate(self, parsed: ParsedAnswer, retrieved_ids: Set[str]) -> ValidationResult:
        retrieved = frozenset(retrieved_ids)
        if parsed.refused:
            return ValidationResult(True, REFUSAL, frozenset(), (), "model refused")
        offending = frozenset(parsed.cited_ids - retrieved)
        uncited = tuple(s.text for s in parsed.sentences if not s.cited_ids)
        if not parsed.sentences:
            return ValidationResult(False, REFUSAL, offending, uncited, "empty answer")
        if not offending and not uncited:
            return ValidationResult(True, _answer_from(list(parsed.sentences)), offending, uncited, "all claims cited from retrieved passages")
        reasons = []
        if offending:
            reasons.append(f"cites passages that were not retrieved: {', '.join(sorted(offending))}")
        if uncited:
            reasons.append(f"{len(uncited)} uncited claim(s)")
        keep = [s for s in parsed.sentences if sentence_is_valid(s, retrieved)]
        answer = _answer_from(keep) if keep else REFUSAL
        return ValidationResult(False, answer, offending, uncited, "; ".join(reasons))
