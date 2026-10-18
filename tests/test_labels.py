import json

import pytest

from gqa import labels
from gqa.eval import Record
from tests import fakes


def _records(n, refused_every=4):
    out = []
    for i in range(n):
        refused = i % refused_every == 0
        out.append(Record(
            question_id=f"q{i:03d}", question=f"Question {i}?", answerable=True,
            answer="not found in the documents" if refused else f"Answer {i} [d1:0:00000001].",
            cited_ids=[] if refused else ["d1:0:00000001"], refused=refused, refusal_source="model" if refused else None,
            candidate_ids=[], context_ids=[], confidence=0.5, latency_ms=1, retrieval_ms=1, generation_ms=0,
            prompt_tokens=0, completion_tokens=0, raw_output=None, judge_label=None if refused else i % 2,
        ))
    return out


BY_ID = {p.passage_id: p for p in fakes.PASSAGES}


def test_packet_written_without_judge_verdicts(tmp_path):
    n = labels.make_packet(_records(80), BY_ID, tmp_path, seed=1, source="run.jsonl")
    assert n == 50
    packet = (tmp_path / "label_packet.md").read_text()
    assert packet.count("## S") == 50 and fakes.TEXTS[1] in packet and "SUPPORTED" not in packet
    assert "judge_label" not in packet
    samples = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 50 and all("judge_label" in s for s in samples)
    csv_lines = (tmp_path / "owner_labels.csv").read_text().splitlines()
    assert csv_lines[0] == "sample_id,label" and csv_lines[1] == "S01," and len(csv_lines) == 51
    # Same seed, same sample.
    again = tmp_path / "again"
    labels.make_packet(_records(80), BY_ID, again, seed=1, source="run.jsonl")
    assert (again / "samples.jsonl").read_text() == (tmp_path / "samples.jsonl").read_text()


def test_packet_refuses_to_overwrite_labels(tmp_path):
    labels.make_packet(_records(80), BY_ID, tmp_path, seed=1, source="r")
    (tmp_path / "owner_labels.csv").write_text("sample_id,label\nS01,1\n")
    with pytest.raises(labels.LabelError, match="refusing to overwrite"):
        labels.make_packet(_records(80), BY_ID, tmp_path, seed=1, source="r")


def _write(tmp_path, body):
    p = tmp_path / "owner_labels.csv"
    p.write_text(body, encoding="utf-8")
    return p


IDS = [f"S{i:02d}" for i in range(1, 51)]


def test_pending_labels_are_not_computed(tmp_path):
    p = _write(tmp_path, "sample_id,label\n" + "".join(f"{s},\n" for s in IDS))
    owner = labels.read_owner_labels(p, IDS)
    assert set(owner.values()) == {None}
    samples = [{"sample_id": s, "judge_label": 1} for s in IDS]
    result = labels.agreement(samples, owner, seed=0)
    assert result.kappa is None and result.cause == labels.PENDING


def test_complete_labels_give_kappa_with_ci(tmp_path):
    judge = [1, 1, 0, 1, 0] * 10
    owner_vals = [1, 0, 0, 1, 0] * 10
    # Notepad may add a BOM and CRLF line endings.
    p = tmp_path / "owner_labels.csv"
    p.write_bytes(("﻿sample_id,label\r\n" + "".join(f"{s},{v}\r\n" for s, v in zip(IDS, owner_vals))).encode("utf-8"))
    owner = labels.read_owner_labels(p, IDS)
    samples = [{"sample_id": s, "judge_label": j} for s, j in zip(IDS, judge)]
    result = labels.agreement(samples, owner, seed=3)
    assert result.n == 50 and result.kappa.point == pytest.approx(0.32 / 0.52)
    assert result.kappa.lower <= result.kappa.point <= result.kappa.upper
    assert result.owner_faithful_rate == 0.4 and result.judge_faithful_rate == 0.6
    short = labels.agreement(samples[:49] + [{"sample_id": "S50", "judge_label": None}], owner, seed=3)
    assert short.kappa is None and "need at least 50" in short.cause


@pytest.mark.parametrize("body, message", [
    ("id,value\n", "line 1"),
    ("sample_id,label\nS99,1\n", "line 2: unknown sample id"),
    ("sample_id,label\nS01,1\nS01,0\n", "line 3: sample S01 labeled twice"),
    ("sample_id,label\nS01,yes\n", "line 2: label must be 0 or 1"),
    ("sample_id,label\nS01,1,extra\n", "line 2: label must be 0 or 1"),
    ("sample_id,label\nS01,1\n", "missing from the file"),
])
def test_malformed_labels_rejected_with_line_number(tmp_path, body, message):
    with pytest.raises(labels.LabelError, match=message):
        labels.read_owner_labels(_write(tmp_path, body), IDS)


def test_existing_labels_reader(tmp_path):
    assert labels.existing_labels(tmp_path / "none.csv") == {}
    p = _write(tmp_path, "sample_id,label\nS01,1\nS02\n")
    assert labels.existing_labels(p) == {"S01": "1", "S02": ""}
