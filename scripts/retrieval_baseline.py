"""Retrieval baseline: python scripts/retrieval_baseline.py

Recall@K and nDCG@K (K = 5, 10, 20) with 95% bootstrap CIs (1,000 resamples) over the answerable
Evaluation_Set questions, for BM25, hybrid, and hybrid + cross-encoder reranking, all scored by
Rag_Lite. Writes results/retrieval_baseline.json and results/retrieval_baseline.md.
"""

from __future__ import annotations

import sys
import time

from gqa.config import load_config
from gqa.evalset import load_questions
from gqa.ingest import read_passages
from gqa.metrics import bootstrap_ci, ndcg_at_k, recall_at_k
from gqa.repro import make_stamp, stamp_markdown, write_json
from gqa.retrieval import RagLiteBackend, RagLiteRetriever, RetrievalMode

KS = (5, 10, 20)
MODES = (RetrievalMode.BM25, RetrievalMode.HYBRID, RetrievalMode.HYBRID_RERANK)


def main() -> int:
    cfg = load_config()
    stamp = make_stamp(seed=cfg.seed)
    passages = read_passages(cfg.corpus_path)
    questions = [q for q in load_questions(cfg.evalset_path) if q.answerable]
    t0 = time.perf_counter()
    backend = RagLiteBackend(passages, weights=cfg.hybrid_weights, candidates=cfg.candidates)
    print(f"index built over {len(passages)} passages in {time.perf_counter() - t0:.1f} s")
    results: dict[str, dict] = {}
    for mode in MODES:
        retriever = RagLiteRetriever(mode, backend, seed=cfg.seed)
        per_q: dict[str, list[float]] = {f"{m}@{k}": [] for m in ("recall", "ndcg") for k in KS}
        for q in questions:
            ids = [h.passage_id for h in retriever.retrieve(q.question, max(KS))]
            gold = set(q.gold_passage_ids)
            for k in KS:
                per_q[f"recall@{k}"].append(recall_at_k(ids, gold, k))
                per_q[f"ndcg@{k}"].append(ndcg_at_k(ids, gold, k))
        results[mode.value] = {name: bootstrap_ci(vals, seed=cfg.seed).as_dict() for name, vals in per_q.items()}
        print(mode.value, {n: round(v["point"], 3) for n, v in results[mode.value].items()})
    write_json("results/retrieval_baseline.json", {"questions": len(questions), "modes": results, "ks": list(KS)}, stamp)
    header = "| Mode | " + " | ".join(f"Recall@{k}" for k in KS) + " | " + " | ".join(f"nDCG@{k}" for k in KS) + " |"
    lines = [
        "# Retrieval baseline (Rag_Lite)", "",
        stamp_markdown(stamp), "",
        f"Answerable Evaluation_Set questions: {len(questions)}. Each cell: point estimate [95% bootstrap CI, 1,000 resamples].", "",
        header, "|" + " --- |" * (1 + 2 * len(KS)),
    ]
    for mode, metrics in results.items():
        cells = [f"{metrics[f'{m}@{k}']['point']:.3f} [{metrics[f'{m}@{k}']['lower']:.3f}, {metrics[f'{m}@{k}']['upper']:.3f}]"
                 for m in ("recall", "ndcg") for k in KS]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    with open("results/retrieval_baseline.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
