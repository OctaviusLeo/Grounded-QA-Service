import json

import pytest

from gqa import squad

SAMPLE = {
    "version": "v2.0",
    "data": [
        {"title": "Super_Bowl_50", "paragraphs": [
            {"context": "Super Bowl 50 was played in Santa Clara.", "qas": [
                {"id": "q1", "question": " Where was Super Bowl 50 played? ", "answers": [{"text": "Santa Clara", "answer_start": 28}], "is_impossible": False},
                {"id": "q2", "question": "Who sang the anthem?", "answers": [], "plausible_answers": [{"text": "x", "answer_start": 0}], "is_impossible": True},
            ]},
        ]},
    ],
}


def test_load_squad_documents_and_questions(tmp_path):
    p = tmp_path / "dev.json"
    p.write_text(json.dumps(SAMPLE))
    data = squad.load_squad(p)
    (doc,) = data.documents
    assert doc.doc_id == "a00p000" and doc.title == "Super Bowl 50"
    q1, q2 = data.questions
    assert q1.question == "Where was Super Bowl 50 played?" and q1.answers == (("Santa Clara", 28),)
    assert doc.text[28:39] == "Santa Clara"
    assert q2.is_impossible and q2.answers == ()


def test_load_squad_rejects_bad_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{}")
    with pytest.raises(squad.DatasetError):
        squad.load_squad(p)


def test_ensure_downloaded_checks_sha(tmp_path):
    payload = json.dumps(SAMPLE).encode()

    def fetch(url, dest):
        dest.write_bytes(payload)

    dest = tmp_path / "squad" / "dev.json"
    expected = __import__("hashlib").sha256(payload).hexdigest()
    assert squad.ensure_downloaded(dest, url="u", expected=expected, fetch=fetch) == dest
    assert squad.ensure_downloaded(dest, url="u", expected=expected, fetch=None) == dest  # cached, no fetch
    with pytest.raises(squad.DatasetError, match="SHA-256"):
        squad.ensure_downloaded(dest, url="u", expected="0" * 64, fetch=None)


def test_data_dir_and_doc_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("GQA_DATA", str(tmp_path))
    assert squad.data_dir() == tmp_path
    assert squad.doc_id_for(3, 12) == "a03p012"
