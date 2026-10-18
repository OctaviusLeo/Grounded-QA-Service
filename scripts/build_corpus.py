"""Build the Corpus: python scripts/build_corpus.py

Downloads the SQuAD 2.0 dev set (checksum-verified), chunks every context paragraph into
data/corpus/passages.jsonl, refuses to write a corpus with duplicate Passage_IDs (reporting
each duplicated ID and the count of affected passages), and writes a stamped manifest.
"""

from __future__ import annotations

import sys

from gqa.config import load_config
from gqa.docblock import update_file
from gqa.ingest import Chunker, find_duplicate_ids, write_passages
from gqa.repro import make_stamp, stamp_markdown, write_json
from gqa.squad import SQUAD_LICENSE, SQUAD_SHA256, SQUAD_URL, ensure_downloaded, load_squad, sha256_of
from pathlib import Path


def main() -> int:
    cfg = load_config()
    stamp = make_stamp(seed=cfg.seed)
    data = load_squad(ensure_downloaded())
    passages = Chunker(cfg.chunk_size, cfg.overlap, cfg.seed).chunk(data.documents)
    dups = find_duplicate_ids(passages)
    if dups:
        for pid, n in dups.items():
            print(f"duplicate Passage_ID {pid}: {n} passages", file=sys.stderr)
        print(f"FAILED: {len(dups)} duplicated Passage_IDs affecting {sum(dups.values())} passages", file=sys.stderr)
        return 1
    write_passages(cfg.corpus_path, passages)
    manifest = {
        "source_url": SQUAD_URL,
        "source_sha256": SQUAD_SHA256,
        "license": SQUAD_LICENSE,
        "documents": len(data.documents),
        "passages": len(passages),
        "multi_passage_documents": len({p.doc_id for p in passages if p.chunk_index > 0}),
        "chunk_size_chars": cfg.chunk_size,
        "overlap_chars": cfg.overlap,
        "max_passage_chars": max(len(p.text) for p in passages),
        "duplicate_passage_ids": 0,
        "passages_sha256": sha256_of(Path(cfg.corpus_path)),
    }
    write_json("data/corpus/manifest.json", manifest, stamp)
    update_file("THIRD_PARTY.md", "dataset-squad", "\n".join([
        stamp_markdown(stamp), "",
        "- **SQuAD 2.0 dev set** (`dev-v2.0.json`, version v2.0), Rajpurkar, Jia and Liang (2018), "
        "\"Know What You Don't Know: Unanswerable Questions for SQuAD\". Source: "
        f"{SQUAD_URL}, SHA-256 `{SQUAD_SHA256}`. License: **{SQUAD_LICENSE}** (the paragraphs are from Wikipedia, also CC BY-SA).",
        f"- Used for both the Corpus ({manifest['documents']} context paragraphs chunked into {manifest['passages']} passages, "
        "`data/corpus/passages.jsonl`) and the Evaluation_Set and tuning questions (`data/evalset.jsonl`, `data/tuning.jsonl`).",
        "- Derived files under `data/` are adaptations of SQuAD 2.0 and carry the same CC BY-SA 4.0 license with attribution as above.",
    ]), after_heading="## Datasets")
    print(f"corpus: {manifest['documents']} documents -> {manifest['passages']} passages, no duplicate Passage_IDs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
