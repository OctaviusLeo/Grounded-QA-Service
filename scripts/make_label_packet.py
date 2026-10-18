"""Write the owner's faithfulness labeling packet: python scripts/make_label_packet.py

Samples 50 answered questions (seeded) from the large-model hybrid+rerank experiment run and
writes results/judge/{label_packet.md,samples.jsonl,owner_labels.csv}. Never overwrites labels.
"""

from __future__ import annotations

import argparse
import sys

from gqa.config import load_config
from gqa.eval import read_records
from gqa.ingest import read_passages
from gqa.labels import LabelError, make_packet

DEFAULT_RUN = "results/runs/hybrid_rerank_8b.records.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=DEFAULT_RUN)
    parser.add_argument("--out", default="results/judge")
    args = parser.parse_args(argv)
    cfg = load_config()
    passages = {p.passage_id: p for p in read_passages(cfg.corpus_path)}
    try:
        n = make_packet(read_records(args.records), passages, args.out, seed=cfg.seed, source=args.records)
    except LabelError as exc:
        print(f"not written: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {n} samples to {args.out}/label_packet.md and {args.out}/owner_labels.csv")
    return 0 if n >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
