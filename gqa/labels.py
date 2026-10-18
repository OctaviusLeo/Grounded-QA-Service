"""The owner's faithfulness labels and the judge-agreement computation (Req 7.5).

``make_packet`` samples 50 answered questions from one experiment run and writes a reading packet
(question, answer, cited passages; the judge's verdict is withheld so it cannot anchor the owner),
the hidden sample file with the judge labels, and an empty CSV for the owner's labels. An
existing CSV that already holds labels is never overwritten. ``agreement`` reads the CSV,
validates it line by line, and computes Cohen's kappa with a bootstrap CI.
"""

from __future__ import annotations

import csv
import io
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gqa.eval import Record
from gqa.ingest import Passage
from gqa.judge import MIN_AGREEMENT_SAMPLES
from gqa.metrics import BootstrapCI, bootstrap_ci, cohen_kappa

PENDING = "owner labels pending"


class LabelError(ValueError):
    """The owner's label file is malformed."""


def sample_records(records: Sequence[Record], n: int, seed: int) -> list[Record]:
    answered = sorted((r for r in records if not r.refused), key=lambda r: r.question_id)
    rng = random.Random(seed)
    return sorted(rng.sample(answered, min(n, len(answered))), key=lambda r: r.question_id)


def existing_labels(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    rows = list(csv.reader(io.StringIO(csv_path.read_text(encoding="utf-8-sig"))))
    return {r[0].strip(): (r[1].strip() if len(r) > 1 else "") for r in rows[1:] if r and r[0].strip()}


def make_packet(records: Sequence[Record], passages_by_id: dict[str, Passage], out_dir: str | Path,
                seed: int, source: str, n: int = MIN_AGREEMENT_SAMPLES) -> int:
    out = Path(out_dir)
    csv_path = out / "owner_labels.csv"
    if any(v for v in existing_labels(csv_path).values()):
        raise LabelError(f"{csv_path} already contains owner labels; refusing to overwrite them")
    chosen = sample_records(records, n, seed)
    out.mkdir(parents=True, exist_ok=True)
    samples, lines = [], [
        "# Faithfulness labeling packet",
        "",
        f"{len(chosen)} answers sampled (seed {seed}) from `{source}`. For each sample, read the answer and the passages it cites.",
        "",
        "Label **1** if every claim in the answer is supported by the cited passages, otherwise **0**. "
        "Judge support only, not whether the answer is correct for the question. The model's own verdict is not shown.",
        "",
        "Write each label after the comma in `results/judge/owner_labels.csv` (one line per sample id), save, then run:",
        "",
        "```powershell",
        "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\finalize_owner_labels.ps1",
        "```",
        "",
    ]
    for i, r in enumerate(chosen, start=1):
        sid = f"S{i:02d}"
        cited = [passages_by_id[p] for p in r.cited_ids if p in passages_by_id]
        samples.append({"sample_id": sid, "question_id": r.question_id, "question": r.question, "answer": r.answer,
                        "cited_ids": r.cited_ids, "judge_label": r.judge_label})
        lines += [f"## {sid}", "", f"**Question:** {r.question}", "", f"**Answer:** {r.answer}", ""]
        for p in cited:
            lines += [f"**Passage `{p.passage_id}`** ({p.title})", "", "> " + p.text.replace("\n", "\n> "), ""]
    (out / "label_packet.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (out / "samples.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8", newline="\n")
    csv_path.write_text("sample_id,label\n" + "".join(f"{s['sample_id']},\n" for s in samples), encoding="utf-8", newline="\n")
    return len(chosen)


def read_owner_labels(csv_path: str | Path, sample_ids: Sequence[str]) -> dict[str, int | None]:
    """``{sample_id: 0 | 1 | None}``; raises :class:`LabelError` naming the line of any problem."""
    text = Path(csv_path).read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or [c.strip().lower() for c in rows[0][:2]] != ["sample_id", "label"]:
        raise LabelError("line 1: expected the header 'sample_id,label'")
    labels: dict[str, int | None] = {}
    expected = set(sample_ids)
    for lineno, row in enumerate(rows[1:], start=2):
        if not row or not "".join(row).strip():
            continue
        sid = row[0].strip()
        value = row[1].strip() if len(row) > 1 else ""
        if sid not in expected:
            raise LabelError(f"line {lineno}: unknown sample id {sid!r}")
        if sid in labels:
            raise LabelError(f"line {lineno}: sample {sid} labeled twice")
        if value not in ("", "0", "1") or len([c for c in row[2:] if c.strip()]):
            raise LabelError(f"line {lineno}: label must be 0 or 1, got {','.join(row[1:])!r}")
        labels[sid] = int(value) if value else None
    missing = expected - set(labels)
    if missing:
        raise LabelError(f"sample ids missing from the file: {', '.join(sorted(missing))}")
    return labels


def _kappa_stat(codes: np.ndarray) -> np.ndarray:
    """Kappa per row of ``codes`` where code = 2 * judge + owner."""
    rows = np.atleast_2d(codes)
    out = []
    for row in rows:
        j, o = (row // 2).tolist(), (row % 2).tolist()
        out.append(cohen_kappa(j, o))
    return np.asarray(out)


@dataclass(frozen=True)
class AgreementResult:
    kappa: BootstrapCI | None
    n: int
    cause: str | None
    owner_faithful_rate: float | None = None
    judge_faithful_rate: float | None = None


def agreement(samples: Sequence[dict], owner: dict[str, int | None], seed: int) -> AgreementResult:
    if all(v is None for v in owner.values()):
        return AgreementResult(None, 0, PENDING)
    pairs = [(s["judge_label"], owner[s["sample_id"]]) for s in samples
             if s.get("judge_label") is not None and owner.get(s["sample_id"]) is not None]
    if len(pairs) < MIN_AGREEMENT_SAMPLES:
        return AgreementResult(None, len(pairs), f"only {len(pairs)} labeled pairs with a judge verdict, need at least {MIN_AGREEMENT_SAMPLES}")
    codes = [2 * j + o for j, o in pairs]
    ci = bootstrap_ci(codes, seed=seed, statistic=_kappa_stat)
    return AgreementResult(ci, len(pairs), None, sum(o for _, o in pairs) / len(pairs), sum(j for j, _ in pairs) / len(pairs))
