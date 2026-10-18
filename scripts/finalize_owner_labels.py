"""Compute judge-vs-owner agreement from results/judge/owner_labels.csv.

python scripts/finalize_owner_labels.py   (run by scripts/finalize_owner_labels.ps1 after the labels are committed)
Writes results/judge/agreement.json; exits non-zero if the labels are malformed or incomplete.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gqa.config import load_config
from gqa.labels import LabelError, agreement, read_owner_labels
from gqa.repro import make_stamp, write_json

DIR = Path("results/judge")


def main() -> int:
    cfg = load_config()
    stamp = make_stamp(seed=cfg.seed)
    samples = [json.loads(line) for line in (DIR / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        owner = read_owner_labels(DIR / "owner_labels.csv", [s["sample_id"] for s in samples])
    except LabelError as exc:
        print(f"owner_labels.csv: {exc}", file=sys.stderr)
        return 1
    missing = [sid for sid, v in owner.items() if v is None]
    if missing:
        print(f"owner_labels.csv: {len(missing)} sample(s) still unlabeled: {', '.join(missing)}", file=sys.stderr)
        return 1
    result = agreement(samples, owner, cfg.seed)
    body = {"n": result.n, "samples": len(samples), "owner_faithful_rate": result.owner_faithful_rate,
            "judge_faithful_rate": result.judge_faithful_rate}
    body["kappa"] = result.kappa.as_dict() if result.kappa else {"not_computed": result.cause}
    write_json(DIR / "agreement.json", body, stamp)
    print("Cohen's kappa:", body["kappa"])
    return 0 if result.kappa else 1


if __name__ == "__main__":
    sys.exit(main())
