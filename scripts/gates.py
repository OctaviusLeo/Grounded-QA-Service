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


def gate_m3() -> None:
    runs = [stamped(f"results/eval/ci_run{i}.json") for i in (1, 2)]
    for i, body in enumerate(runs, 1):
        computed = [m for m, v in body["metrics"].items() if "point" in v]
        check(len(computed) == len(body["metrics"]), f"CI-subset run {i} computed every metric", f"{len(computed)}/{len(body['metrics'])}")
        check(body["counts"]["questions"] == 30, f"CI-subset run {i} covered the 30-question subset")
    det = stamped("results/eval/determinism.json")
    check(det["identical"], "the two runs agree within 1e-6 (timings excluded)",
          f"{len(det['metric_differences'])} metric / {det['record_difference_count']} record differences")
    for path in ("scripts/make_label_packet.py", "scripts/finalize_owner_labels.ps1", "scripts/finalize_owner_labels.py"):
        check(Path(path).exists(), f"{path} exists")


def gate_m4() -> None:
    summary = Path("results/summary.md").read_text(encoding="utf-8")
    exp = stamped("results/experiments.json")
    rows = exp["configurations"]
    check(len(rows) >= 4 and not any(r["failed"] for r in rows), "4 configurations completed", ", ".join(r["config"] for r in rows))
    modes = {(r["mode"], r["model"]) for r in rows}
    check({m for m, _ in modes} == {"bm25", "hybrid", "hybrid_rerank"}
          and {"granite3.3:2b", "granite3.3:8b"} <= {mod for m, mod in modes if m == "hybrid_rerank"},
          "BM25, hybrid, and hybrid+rerank with a smaller and a larger model")
    quality = summary.split("### Answer quality, latency, and cost")[1].split("### Retrieval")[0]
    table_rows = [line for line in quality.splitlines() if line.startswith("| ") and "---" not in line][1:]
    check(len(table_rows) == len(rows) and all(line.count("[") >= 6 for line in table_rows),
          "Results_Table has one row per configuration with CIs", f"{len(table_rows)} rows")
    for r in rows:
        for group in ("quality", "retrieval"):
            for name, v in r[group].items():
                ok = "point" in v and 0.0 <= v["lower"] <= v["point"] <= v["upper"] <= 1.0 and v["resamples"] >= 1000
                if not ok and not (name == "faithfulness" and "not_computed" in v):
                    check(False, f"{r['config']} {name} has a 95% CI in [0, 1]", str(v))
        check(r["p50_latency_ms"] <= r["p95_latency_ms"], f"{r['config']} P50 <= P95")
        check(r["cost_per_1k_usd"] is not None, f"{r['config']} cost per 1,000 questions reported", r["cost_basis"])
    for heading in ("## Retrieval baseline", "## Injection test", "## Judge agreement"):
        check(heading in summary, f"summary has a section '{heading}'")
    agreement = summary.split("## Judge agreement")[1].split("##")[0]
    check("not computed: owner labels pending" in agreement or "Cohen's kappa over" in agreement, "judge agreement reported or pending owner labels")
    inj = stamped("results/injection.json")
    check(inj["cases"] >= 20 and 0.0 <= inj["followed_fraction"] <= 1.0, "injection test: >= 20 cases, fraction in [0, 1]",
          f"{inj['followed']} followed / {inj['not_followed']} not / {inj['unclassified']} unclassified")
    check(Path("results/judge/label_packet.md").exists(), "results/judge/label_packet.md exists")
    lines = Path("results/judge/owner_labels.csv").read_text(encoding="utf-8").splitlines()
    check(lines[0] == "sample_id,label" and len(lines) == 51, "owner_labels.csv has 50 sample lines")


def gate_m5() -> None:
    from gqa.gate import load_baseline
    from gqa.readme_check import run

    findings = run()
    check(not findings, "every README number traces to a generated file", "; ".join(f"{f.number} (line {f.line})" for f in findings[:5]))
    b = load_baseline("results/baseline.json")
    check(commit_exists(b.commit), "results/baseline.json is stamped with an existing commit", b.commit)
    check(Path("docs/media/demo.gif").exists() and Path("docs/media/demo.gif").stat().st_size > 10_000, "docs/media/demo.gif exists")
    readme = Path("README.md").read_text(encoding="utf-8")
    for needle in ("```mermaid", "docker compose up", "bash scripts/test_unit.sh", "## Known limitations", "docs/media/demo.gif", "## Cost and latency trade-offs"):
        check(needle in readme, f"README contains {needle!r}")
    check(Path(".github/workflows/ci.yml").exists(), "CI workflow exists")


GATES = {"m0": gate_m0, "m1": gate_m1, "m2": gate_m2, "m3": gate_m3, "m4": gate_m4, "m5": gate_m5}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        GATES[name]()
    print(f"gate {' '.join(sys.argv[1:])}: {'FAILED' if FAILS else 'passed'}")
    sys.exit(1 if FAILS else 0)
