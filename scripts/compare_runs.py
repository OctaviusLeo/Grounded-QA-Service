"""Compare two harness outputs for Req 7.8: python scripts/compare_runs.py a.json b.json

Floats must agree within 1e-6 and everything else exactly, excluding timing (latency) values,
which are wall-clock measurements. Writes results/eval/determinism.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gqa.config import load_config
from gqa.repro import make_stamp, write_json

EXCLUDED = {"latency", "stamp"}
RECORD_TIMING = {"latency_ms", "retrieval_ms", "generation_ms"}


def diffs(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k in EXCLUDED or k in RECORD_TIMING:
                continue
            out += diffs(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diffs(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if abs(a - b) <= 1e-6 else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def main(argv: list[str]) -> int:
    a, b = (Path(p) for p in argv[1:3])
    found = diffs(json.loads(a.read_text()), json.loads(b.read_text()))
    ra = [json.loads(line) for line in a.with_suffix(".records.jsonl").read_text().splitlines()]
    rb = [json.loads(line) for line in b.with_suffix(".records.jsonl").read_text().splitlines()]
    record_diffs = diffs(ra, rb, "records")
    stamp = make_stamp(seed=load_config().seed)
    write_json("results/eval/determinism.json", {
        "runs": [a.as_posix(), b.as_posix()], "metric_differences": found, "record_differences": record_diffs[:50],
        "record_difference_count": len(record_diffs), "identical": not found and not record_diffs,
        "excluded": sorted(EXCLUDED | RECORD_TIMING),
    }, stamp)
    print("identical within 1e-6 (timings excluded)" if not found and not record_diffs else
          f"{len(found)} metric and {len(record_diffs)} record differences:\n" + "\n".join((found + record_diffs)[:20]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
