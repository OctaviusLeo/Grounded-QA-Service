import pytest

from gqa.ingest import (
    Chunker,
    Document,
    Passage,
    find_duplicate_ids,
    normalize,
    passage_id_for,
    read_passages,
    write_passages,
)

SENTENCES = " ".join(f"Sentence number {i} talks about the river." for i in range(40))


def test_passage_id_format_and_normalization():
    pid = passage_id_for("a01p002", 3, "Hello   world\n")
    assert pid.startswith("a01p002:3:") and len(pid.split(":")[2]) == 8
    assert pid == passage_id_for("a01p002", 3, " Hello world")
    assert pid != passage_id_for("a01p002", 4, "Hello world")
    assert normalize("é  x") == "é x"


def test_short_document_is_one_passage_with_offsets():
    doc = Document("d0", "Title", "  The Normans lived in Normandy.  ")
    (p,) = Chunker(1000, 200).chunk([doc])
    assert p.text == "The Normans lived in Normandy."
    assert doc.text[p.start_char:p.end_char] == p.text
    assert p.title == "Title" and p.chunk_index == 0


def test_long_document_overlaps_and_offsets_are_exact():
    doc = Document("d1", "T", SENTENCES)
    passages = Chunker(300, 80).chunk([doc])
    assert len(passages) > 3
    assert [p.chunk_index for p in passages] == list(range(len(passages)))
    for p in passages:
        assert doc.text[p.start_char:p.end_char] == p.text
        assert len(p.text) <= 300
    for a, b in zip(passages, passages[1:]):
        assert b.start_char < a.end_char  # windows overlap
    assert len({p.passage_id for p in passages}) == len(passages)


def test_empty_document_yields_no_passages():
    assert Chunker(100, 10).chunk([Document("e", "T", "   \n ")]) == []


def test_invalid_sizes_rejected():
    for size, overlap in [(0, 0), (100, 100), (100, -1)]:
        with pytest.raises(ValueError):
            Chunker(size, overlap)


def test_duplicate_ids_reported_with_counts():
    p = Passage("x:0:aaaaaaaa", "x", 0, "t")
    q = Passage("y:0:bbbbbbbb", "y", 0, "u")
    assert find_duplicate_ids([p, q, p, p]) == {"x:0:aaaaaaaa": 3}
    assert find_duplicate_ids([p, q]) == {}
    # The same document given twice collides on every passage, and the count says so.
    doc = Document("d", "T", SENTENCES)
    twice = Chunker(300, 80).chunk([doc, doc])
    dups = find_duplicate_ids(twice)
    assert dups and set(dups.values()) == {2}


def test_jsonl_round_trip(tmp_path):
    passages = Chunker(300, 80).chunk([Document("d", "T", SENTENCES)])
    path = tmp_path / "c" / "passages.jsonl"
    write_passages(path, passages)
    assert read_passages(path) == passages
    assert b"\r\n" not in path.read_bytes()
