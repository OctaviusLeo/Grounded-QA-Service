"""The SQuAD 2.0 dev set as Corpus documents and candidate questions (Req 2.1, 2.5, 0.8).

Source: https://rajpurkar.github.io/SQuAD-explorer/ (Rajpurkar, Jia and Liang, 2018), license
CC BY-SA 4.0. The file is downloaded once into ``$GQA_DATA/squad`` and checked against a pinned
SHA-256, so every build reads the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from gqa.ingest import Document

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD_SHA256 = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"
SQUAD_LICENSE = "CC-BY-SA-4.0"


class DatasetError(RuntimeError):
    """The dataset file is missing, corrupt, or does not match the pinned checksum."""


@dataclass(frozen=True)
class SquadQuestion:
    question_id: str
    question: str
    doc_id: str
    answers: tuple[tuple[str, int], ...]  # (text, character offset in the context)
    is_impossible: bool


@dataclass(frozen=True)
class SquadData:
    documents: list[Document]
    questions: list[SquadQuestion]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_dir() -> Path:
    return Path(os.environ.get("GQA_DATA", Path.home() / "gqa-data"))


def ensure_downloaded(dest: Path | None = None, url: str = SQUAD_URL, expected: str = SQUAD_SHA256,
                      fetch=urllib.request.urlretrieve) -> Path:
    path = dest or data_dir() / "squad" / "dev-v2.0.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        fetch(url, tmp)
        tmp.replace(path)
    actual = sha256_of(path)
    if actual != expected:
        raise DatasetError(f"{path} has SHA-256 {actual}, expected {expected}")
    return path


def doc_id_for(article_index: int, paragraph_index: int) -> str:
    return f"a{article_index:02d}p{paragraph_index:03d}"


def load_squad(path: Path) -> SquadData:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        articles = raw["data"]
    except (OSError, ValueError, KeyError) as exc:
        raise DatasetError(f"cannot read SQuAD file {path}: {exc}") from exc
    documents, questions = [], []
    for ai, article in enumerate(articles):
        title = article["title"].replace("_", " ")
        for pi, para in enumerate(article["paragraphs"]):
            doc_id = doc_id_for(ai, pi)
            documents.append(Document(doc_id=doc_id, title=title, text=para["context"]))
            for qa in para["qas"]:
                answers = tuple((a["text"], int(a["answer_start"])) for a in qa.get("answers", []))
                questions.append(SquadQuestion(
                    question_id=qa["id"],
                    question=qa["question"].strip(),
                    doc_id=doc_id,
                    answers=answers,
                    is_impossible=bool(qa.get("is_impossible", False)),
                ))
    return SquadData(documents=documents, questions=questions)
