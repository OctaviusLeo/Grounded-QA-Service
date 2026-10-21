"""Paired bootstrap of configuration differences: python scripts/paired_comparison.py

CPU only. Reads the committed per-question records (results/runs/*.records.jsonl) and the
evaluation questions, and writes results/paired.json: hybrid_rerank_8b minus bm25_8b and minus
hybrid_8b, for Recall@5 and citation precision, each with a 95% paired bootstrap CI.
"""

import sys

from gqa.config import load_config
from gqa.eval import read_records
from gqa.evalset import load_questions
from gqa.paired import compare
from gqa.repro import make_stamp, write_json

REFERENCE = "hybrid_rerank_8b"
OTHERS = ("bm25_8b", "hybrid_8b")


def main() -> int:
    cfg = load_config()
    stamp = make_stamp(seed=cfg.seed)
    questions = load_questions(cfg.evalset_path)
    ref = read_records(f"results/runs/{REFERENCE}.records.jsonl")
    comparisons = {o: compare(ref, read_records(f"results/runs/{o}.records.jsonl"), questions, cfg.seed) for o in OTHERS}
    write_json("results/paired.json", {"reference": REFERENCE, "comparisons": comparisons}, stamp)
    for other, metrics in comparisons.items():
        for m, ci in metrics.items():
            print(f"{REFERENCE} - {other} {m}: {ci['point']:+.3f} [{ci['lower']:+.3f}, {ci['upper']:+.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
