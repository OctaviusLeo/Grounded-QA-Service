"""The generated Results_Table, ``results/summary.md`` (Req 8.3, 8.4, 0.1).

Rendered only from stamped result files; each section carries its source's command, seed, and
commit. The only way a number in it changes is by regenerating it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from gqa.experiments import QUALITY, RETRIEVAL, format_usd
from gqa.repro import Stamp, stamp_from_dict, stamp_markdown

PENDING = "owner labels pending"

QUALITY_LABELS = {
    "exact_match": "Exact match", "f1": "Token F1", "citation_precision": "Citation precision",
    "citation_recall": "Citation recall", "refusal_accuracy": "Refusal accuracy",
    "answerable_refused": "Answerable refused", "faithfulness": "Faithfulness (judge)",
}
RETRIEVAL_LABELS = {m: m.replace("recall", "Recall").replace("ndcg", "nDCG") for m in RETRIEVAL}


def cell(metric: dict[str, Any] | None) -> str:
    if not metric:
        return "not computed (missing)"
    if "not_computed" in metric:
        return f"not computed ({metric['not_computed']})"
    return f"{metric['point']:.3f} [{metric['lower']:.3f}, {metric['upper']:.3f}]"


def _load(path: Path) -> tuple[dict[str, Any] | None, Stamp | None, str | None]:
    if not path.exists():
        return None, None, f"{path.as_posix()} not found"
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        return body, stamp_from_dict(body["stamp"]), None
    except (ValueError, KeyError) as exc:
        return None, None, f"{path.as_posix()} unreadable: {exc}"


def experiments_section(path: Path) -> list[str]:
    body, stamp, err = _load(path)
    lines = ["## Configuration experiments (Results_Table)", ""]
    if err:
        return lines + [f"Not computed: {err}.", ""]
    rows = body["configurations"]
    lines += [stamp_markdown(stamp), "",
              f"{body['questions']} Evaluation_Set questions ({body['answerable']} answerable, {body['unanswerable']} unanswerable). "
              f"Each metric: point estimate [95% bootstrap CI, {body['resamples']} resamples]. Latency is the pipeline only "
              f"(retrieval, reranking, generation, validation), not the judge. Judge: `{body['judge_model']}`.", "",
              "### Answer quality, latency, and cost", "",
              "| Configuration | Retrieval | Model | " + " | ".join(QUALITY_LABELS[m] for m in QUALITY) + " | P50 latency (ms) | P95 latency (ms) | Cost per 1,000 questions (USD) |",
              "|" + " --- |" * (3 + len(QUALITY) + 3)]
    for r in rows:
        if r["failed"]:
            lines.append(f"| {r['config']} | {r['mode']} | `{r['model']}` | " + " | ".join(["failed"] * (len(QUALITY) + 3)) + " |")
            continue
        lat = [f"{r['p50_latency_ms']:.0f}", f"{r['p95_latency_ms']:.0f}"]
        cost = format_usd(r["cost_per_1k_usd"]) if r["cost_per_1k_usd"] is not None else "not computed"
        lines.append(f"| {r['config']} | {r['mode']} | `{r['model']}` | " + " | ".join(cell(r["quality"].get(m)) for m in QUALITY)
                     + f" | {lat[0]} | {lat[1]} | {cost} |")
    lines += ["", "### Retrieval", "",
              "| Configuration | " + " | ".join(RETRIEVAL_LABELS[m] for m in RETRIEVAL) + " |",
              "|" + " --- |" * (1 + len(RETRIEVAL))]
    for r in rows:
        if r["failed"]:
            lines.append(f"| {r['config']} | " + " | ".join(["failed"] * len(RETRIEVAL)) + " |")
        else:
            lines.append(f"| {r['config']} | " + " | ".join(cell(r["retrieval"].get(m)) for m in RETRIEVAL) + " |")
    lines += ["", "Cost basis:", ""]
    for r in rows:
        lines.append(f"- {r['config']}: " + (f"FAILED: {r['cause']}" if r["failed"] else r["cost_basis"]))
    lines += ["", f"Electricity price: {body['electricity_source']}", ""]
    return lines


REFERENCE = "hybrid_rerank_8b"
TRADEOFF_METRICS = ("f1", "citation_precision", "refusal_accuracy", "faithfulness")


def tradeoffs_section(path: Path) -> list[str]:
    """Each configuration relative to the reference (differences of point estimates, ratios of latency and cost)."""
    body, stamp, err = _load(path)
    lines = ["## Trade-offs", ""]
    if err:
        return lines + [f"Not computed: {err}.", ""]
    rows = [r for r in body["configurations"] if not r["failed"]]
    ref = next((r for r in rows if r["config"] == REFERENCE), rows[0] if rows else None)
    if ref is None:
        return lines + ["Not computed: every configuration failed.", ""]

    def point(r, m):
        v = r["quality"].get(m, {})
        return v.get("point")

    def diff(r, m):
        a, b = point(r, m), point(ref, m)
        return "n/a" if a is None or b is None else f"{a - b:+.3f}"

    def ratio(a, b):
        return "n/a" if a is None or not b else f"{a / b:.2f}x"

    lines += [stamp_markdown(stamp), "",
              f"Relative to `{ref['config']}`: differences of point estimates (compare the CIs above before reading much into small ones) and ratios of latency and cost.", "",
              "| Configuration | " + " | ".join(f"{QUALITY_LABELS[m]} vs ref" for m in TRADEOFF_METRICS) + " | P50 latency vs ref | P95 latency vs ref | Cost vs ref |",
              "|" + " --- |" * (1 + len(TRADEOFF_METRICS) + 3)]
    for r in rows:
        lines.append(f"| {r['config']} | " + " | ".join(diff(r, m) for m in TRADEOFF_METRICS) + " | "
                     + " | ".join([ratio(r["p50_latency_ms"], ref["p50_latency_ms"]), ratio(r["p95_latency_ms"], ref["p95_latency_ms"]),
                                   ratio(r["cost_per_1k_usd"], ref["cost_per_1k_usd"])]) + " |")
    return lines + [""]


def baseline_section(path: Path) -> list[str]:
    body, stamp, err = _load(path)
    lines = ["## Retrieval baseline (Rag_Lite, answerable questions)", ""]
    if err:
        return lines + [f"Not computed: {err}.", ""]
    ks = body["ks"]
    lines += [stamp_markdown(stamp), "", f"{body['questions']} answerable questions.", "",
              "| Mode | " + " | ".join(f"Recall@{k}" for k in ks) + " | " + " | ".join(f"nDCG@{k}" for k in ks) + " |",
              "|" + " --- |" * (1 + 2 * len(ks))]
    for mode, m in body["modes"].items():
        lines.append(f"| {mode} | " + " | ".join(cell(m[f"recall@{k}"]) for k in ks) + " | " + " | ".join(cell(m[f"ndcg@{k}"]) for k in ks) + " |")
    return lines + [""]


def injection_section(path: Path) -> list[str]:
    body, stamp, err = _load(path)
    lines = ["## Injection test", ""]
    if err:
        return lines + [f"Not computed: {err}.", ""]
    lines += [stamp_markdown(stamp), "",
              f"Configuration: {body['mode']} with `{body['model']}`. Rule: see `docs/INJECTION.md`.", "",
              "| Cases | Followed | Not followed | Unclassified | Followed fraction |",
              "| --- | --- | --- | --- | --- |",
              f"| {body['cases']} | {body['followed']} | {body['not_followed']} | {body['unclassified']} | {body['followed_fraction']:.3f} |", ""]
    classified = body["followed"] + body["not_followed"]
    if "raw_followed" in body:
        lines += [f"Before citation validation, the canary appeared in the model's raw output in {body['raw_followed']} of the "
                  f"{classified} classified cases (validation dropped those uncited sentences).", ""]
    return lines


def agreement_section(path: Path) -> list[str]:
    lines = ["## Judge agreement (Cohen's kappa vs owner labels)", ""]
    body, stamp, err = _load(path)
    if err:
        return lines + [f"not computed: {PENDING} (`results/judge/owner_labels.csv`; run `scripts/finalize_owner_labels.ps1` after labeling).", ""]
    kappa = body["kappa"]
    if "not_computed" in kappa:
        return lines + [stamp_markdown(stamp), "", f"not computed: {kappa['not_computed']}", ""]
    return lines + [stamp_markdown(stamp), "",
                    f"Cohen's kappa over {body['n']} samples: {cell(kappa)}. Owner labeled {body['owner_faithful_rate']:.3f} "
                    f"of the samples faithful; the judge {body['judge_faithful_rate']:.3f}.", ""]


def thresholds_section(path: Path) -> list[str]:
    lines = ["## Refusal thresholds (tuned on the tuning split)", ""]
    if not path.exists():
        return lines + ["Not tuned: config/thresholds.yaml not found.", ""]
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    lines += [stamp_markdown(stamp_from_dict(data["stamp"])), "",
              f"{data['questions']} tuning questions (disjoint from the Evaluation_Set), model `{data['tuning_model']}`. {data['rule']}.", "",
              "| Mode | Threshold | Tuning balanced accuracy | Unanswerable refused | Answerable answered |", "| --- | --- | --- | --- | --- |"]
    for mode, t in data["thresholds"].items():
        s = data["tuning_split_scores"][mode]
        lines.append(f"| {mode} | {t:.4f} | {s['balanced_accuracy']:.3f} | {s['unanswerable_refused']:.3f} | {s['answerable_answered']:.3f} |")
    return lines + [""]


PAIRED_LABELS = {"recall@5": "Recall@5", "citation_precision": "Citation precision"}


def paired_section(path: Path) -> list[str]:
    body, stamp, err = _load(path)
    lines = ["## Paired comparison (same questions)", ""]
    if err:
        return lines + [f"Not computed: {err}.", ""]
    ref = body["reference"]
    lines += [stamp_markdown(stamp), "",
              f"Difference of `{ref}` minus each other configuration over the same answerable questions, with a 95% paired "
              "bootstrap CI (each resample scores both configurations on the same draw of questions). Citation precision "
              "counts only the questions each configuration answered. An interval that excludes zero indicates a difference.", "",
              "| Comparison | Metric | Difference [95% CI] | Questions scored (reference / other) |", "| --- | --- | --- | --- |"]
    for other, metrics in body["comparisons"].items():
        for m, ci in metrics.items():
            lines.append(f"| {ref} − {other} | {PAIRED_LABELS[m]} | {ci['point']:+.3f} [{ci['lower']:+.3f}, {ci['upper']:+.3f}] "
                         f"| {ci['n_a']} / {ci['n_b']} |")
    return lines + [""]


def render(stamp: Stamp, results_dir: str | Path = "results", thresholds: str | Path = "config/thresholds.yaml") -> str:
    d = Path(results_dir)
    parts: Sequence[list[str]] = [
        ["# Results", "", "Generated file: do not edit. Regenerate with `python scripts/write_results_table.py`.", "",
         stamp_markdown(stamp), ""],
        experiments_section(d / "experiments.json"),
        tradeoffs_section(d / "experiments.json"),
        baseline_section(d / "retrieval_baseline.json"),
        injection_section(d / "injection.json"),
        agreement_section(d / "judge" / "agreement.json"),
        thresholds_section(Path(thresholds)),
        paired_section(d / "paired.json"),
    ]
    return "\n".join(line for part in parts for line in part).rstrip() + "\n"


def write_results_table(stamp: Stamp, path: str | Path = "results/summary.md", **kw: Any) -> str:
    text = render(stamp, **kw)
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    return text
