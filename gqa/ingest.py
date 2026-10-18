"""Corpus ingestion and chunking with stable Passage_IDs (Req 2.1, 2.2).

Window splitting is delegated to Rag_Lite's ``chunk_text`` (character windows with overlap,
preferring sentence boundaries). This module adds what Rag_Lite's sequential chunk ids lack:
a Passage_ID that depends only on the document id, the chunk's position, and the chunk's
normalized text, so the same corpus always yields the same IDs, plus duplicate detection.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from src.rag import chunk_text  # Rag_Lite

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Passage:
    passage_id: str
    doc_id: str
    chunk_index: int
    text: str
    start_char: int = 0  # offset of ``text`` in its document
    title: str = ""

    @property
    def end_char(self) -> int:
        return self.start_char + len(self.text)


def normalize(text: str) -> str:
    """Unicode NFC with every run of whitespace collapsed to one space."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def passage_id_for(doc_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}:{chunk_index}:{digest}"


class Chunker:
    """Split documents into Passages. Deterministic; ``seed`` is accepted for interface symmetry."""

    def __init__(self, chunk_size: int, overlap: int, seed: int = 0) -> None:
        if chunk_size <= 0 or not 0 <= overlap < chunk_size:
            raise ValueError(f"need chunk_size > 0 and 0 <= overlap < chunk_size, got {chunk_size}, {overlap}")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.seed = seed

    def chunk_document(self, doc: Document) -> list[Passage]:
        out = []
        for chunk in chunk_text(doc.text, chunk_size=self.chunk_size, overlap=self.overlap):
            # Rag_Lite strips each window; recover the stripped text's true offset.
            start = doc.text.find(chunk.text, chunk.start_char)
            out.append(Passage(
                passage_id=passage_id_for(doc.doc_id, chunk.chunk_index, chunk.text),
                doc_id=doc.doc_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                start_char=start if start >= 0 else chunk.start_char,
                title=doc.title,
            ))
        return out

    def chunk(self, documents: Iterable[Document]) -> list[Passage]:
        return [p for doc in documents for p in self.chunk_document(doc)]


def find_duplicate_ids(passages: Sequence[Passage]) -> dict[str, int]:
    """``{passage_id: count}`` for every id assigned to more than one Passage."""
    counts = Counter(p.passage_id for p in passages)
    return {pid: n for pid, n in sorted(counts.items()) if n > 1}


def write_passages(path: str | Path, passages: Sequence[Passage]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for passage in passages:
            fh.write(json.dumps(asdict(passage), ensure_ascii=False, sort_keys=True) + "\n")


def read_passages(path: str | Path) -> list[Passage]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Passage(**json.loads(line)))
    return out
