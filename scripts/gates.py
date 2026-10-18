"""Milestone gate checks (replace the owner review stops). Usage: python scripts/gates.py m0|m1|...

Each check prints PASS or FAIL with the reason; the exit status is non-zero if any check fails.
The unit suite itself is run by scripts/gate_m<N>.sh before this script.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from gqa.docblock import read_block

FAILS: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + what + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(what)


def commit_exists(commit: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode == 0


def stamped(path: str) -> dict:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    st = body.get("stamp", {})
    check(bool(re.fullmatch(r"[0-9a-f]{40}", st.get("commit", ""))) and commit_exists(st["commit"]),
          f"{path} stamped with an existing 40-char commit", st.get("commit", "missing"))
    check(isinstance(st.get("seed"), int) and bool(st.get("command")), f"{path} records seed and command")
    return body


def gate_m0() -> None:
    env = Path("docs/ENVIRONMENT.md").read_text(encoding="utf-8")
    for model in ("granite3.3:8b", "granite3.3:2b"):
        block = read_block(env, f"smoke:{model}") or ""
        check("Status: passed" in block, f"smoke test passed for {model}")
        m = re.search(r"peak during generation: \*\*(\d+) MiB\*\* of (\d+) MiB", block)
        check(bool(m) and int(m.group(1)) <= int(m.group(2)), f"{model} peak GPU memory <= total", m.group(0) if m else "not recorded")
        check(bool(re.search(r"commit `[0-9a-f]{40}`", block)), f"{model} evidence stamped")
    for name in ("model-server", "tools", "python-packages", "machine", "hosted-model"):
        check(read_block(env, name) is not None, f"ENVIRONMENT.md block {name}")
    tp = Path("THIRD_PARTY.md").read_text(encoding="utf-8")
    for needle in ("Ollama", "granite3.3:2b", "granite3.3:8b", "Apache-2.0"):
        check(needle in tp, f"THIRD_PARTY.md lists {needle}")


def gate_m1() -> None:
    from gqa.evalset import load_questions, validate
    from gqa.ingest import find_duplicate_ids, read_passages
    from gqa.squad import sha256_of

    manifest = stamped("data/corpus/manifest.json")
    passages = read_passages("data/corpus/passages.jsonl")
    check(not find_duplicate_ids(passages), "corpus has no duplicate Passage_IDs", f"{len(passages)} passages")
    check(sha256_of(Path("data/corpus/passages.jsonl")) == manifest["passages_sha256"], "corpus file matches its manifest")
    ids = {p.passage_id for p in passages}
    evalset, tuning = load_questions("data/evalset.jsonl"), load_questions("data/tuning.jsonl")
    for name, qs, mins in (("Evaluation_Set", evalset, (150, 20)), ("tuning split", tuning, (100, 20))):
        report = validate(qs, ids, *mins)
        check(report.ok, f"{name} validates", "; ".join(report.lines()))
    check(not ({q.question_id for q in evalset} & {q.question_id for q in tuning}), "tuning split is disjoint from the Evaluation_Set")
    stamped("data/evalset_manifest.json")
    base = stamped("results/retrieval_baseline.json")
    values = [v for mode in base["modes"].values() for ci in mode.values() for v in (ci["point"], ci["lower"], ci["upper"])]
    check(bool(values) and all(0.0 <= v <= 1.0 for v in values), "every retrieval baseline value is in [0, 1]", f"{len(values)} values")
    check(set(base["modes"]) == {"bm25", "hybrid", "hybrid_rerank"} and base["ks"] == [5, 10, 20], "baseline covers 3 modes at K = 5, 10, 20")
    check(all(ci["resamples"] >= 1000 for mode in base["modes"].values() for ci in mode.values()), "baseline CIs use >= 1000 resamples")


def gate_m2() -> None:
    import os

    import yaml

    data = yaml.safe_load(Path("config/thresholds.yaml").read_text(encoding="utf-8"))
    st = data.get("stamp", {})
    check(bool(re.fullmatch(r"[0-9a-f]{40}", st.get("commit", ""))) and commit_exists(st["commit"]), "thresholds stamped with an existing commit")
    th = data.get("thresholds", {})
    check(set(th) == {"bm25", "hybrid", "hybrid_rerank"} and all(0.0 <= float(v) <= 1.0 for v in th.values()),
          "a tuned threshold in [0, 1] for every mode", str(th))
    check(data.get("tuning_split") == "data/tuning.jsonl", "thresholds tuned on the tuning split")
    runs = Path(os.environ.get("GQA_RUNS", Path.home() / "gqa-runs"))
    live = json.loads((runs / "pipeline_check.json").read_text(encoding="utf-8")) if (runs / "pipeline_check.json").exists() else []
    check(len(live) == 5 and all(r["ok"] for r in live), "live pipeline check: 5 answers, each a refusal or citing only retrieved passages")


GATES = {"m0": gate_m0, "m1": gate_m1, "m2": gate_m2}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        GATES[name]()
    print(f"gate {' '.join(sys.argv[1:])}: {'FAILED' if FAILS else 'passed'}")
    sys.exit(1 if FAILS else 0)
