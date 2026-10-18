"""Configuration experiments: quality, retrieval, latency, and cost per Configuration (Req 8.1-8.6).

For each Configuration the runner sends one unscored warm-up request (so model loading is not
counted as question latency), then answers every Evaluation_Set question while sampling GPU
power. After all Configurations have answered, the judge scores every answer (one model swap).

Cost per 1,000 questions:
- local model: measured GPU energy during the answering pass x the recorded electricity price;
- hosted model (none is configured): recorded token counts x the recorded per-token prices.
A Configuration that fails is reported with its cause; the others still produce results.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gqa.eval import KS, Record, compute_metrics, counts, generate, judge_records, latency_summary, write_records
from gqa.evalset import EvalQuestion
from gqa.prompt import PromptMessages
from gqa.retrieval import RetrievalMode

QUALITY = ("exact_match", "f1", "citation_precision", "citation_recall", "refusal_accuracy", "answerable_refused", "faithfulness")
RETRIEVAL = tuple(f"recall@{k}" for k in KS) + tuple(f"ndcg@{k}" for k in KS)


@dataclass(frozen=True)
class Configuration:
    name: str
    mode: RetrievalMode
    model: str
    hosted: bool = False


@dataclass(frozen=True)
class CostSettings:
    usd_per_kwh: float
    source: str
    hosted_prices: dict[str, dict[str, float]] = field(default_factory=dict)  # model -> input/output USD per 1M tokens


@dataclass
class ConfigResult:
    config: str
    mode: str
    model: str
    quality: dict[str, dict[str, Any]] = field(default_factory=dict)
    retrieval: dict[str, dict[str, Any]] = field(default_factory=dict)
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    cost_per_1k_usd: float | None = None
    cost_basis: str = ""
    energy_kwh: float | None = None
    gpu_mean_power_w: float | None = None
    answering_seconds: float | None = None
    counts: dict[str, int] = field(default_factory=dict)
    failed: bool = False
    cause: str | None = None


def load_experiments(path: str | Path) -> tuple[list[Configuration], CostSettings, dict[str, Any]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    configs = [Configuration(c["name"], RetrievalMode(c["mode"]), c["model"], bool(c.get("hosted", False)))
               for c in data["configurations"]]
    if len({c.name for c in configs}) != len(configs):
        raise ValueError("configuration names must be unique")
    cost = data["cost"]
    settings = CostSettings(float(cost["electricity_usd_per_kwh"]), str(cost["electricity_source"]),
                            {m: {k: float(v) for k, v in p.items()} for m, p in (data.get("hosted_prices") or {}).items()})
    return configs, settings, data


def hosted_cost_per_1k(prompt_tokens: int, completion_tokens: int, questions: int,
                       input_usd_per_mtok: float, output_usd_per_mtok: float) -> float:
    """USD per 1,000 questions from token counts and per-million-token prices (Req 8.5)."""
    if questions <= 0:
        raise ValueError("need at least one question")
    total = (prompt_tokens * input_usd_per_mtok + completion_tokens * output_usd_per_mtok) / 1_000_000
    return total / questions * 1000


def local_cost_per_1k(energy_kwh: float, questions: int, usd_per_kwh: float) -> float:
    """USD per 1,000 questions from measured energy and the electricity price."""
    if questions <= 0:
        raise ValueError("need at least one question")
    return energy_kwh * usd_per_kwh / questions * 1000


def format_usd(value: float) -> str:
    """At least two decimal places; more when the value would otherwise round to 0.00."""
    if value != 0 and abs(value) < 0.01:
        return f"{value:.4f}"
    return f"{value:.2f}"


ClientFactory = Callable[[str], Any]
PipelineFactory = Callable[[Configuration, Any], Any]
SamplerFactory = Callable[[], Any]


def run_experiments(configs: Sequence[Configuration], questions: Sequence[EvalQuestion], seed: int, *,
                    make_pipeline: PipelineFactory, make_client: ClientFactory, make_judge: Callable[[], Any],
                    make_sampler: SamplerFactory, passages_by_id: dict, cost: CostSettings,
                    out_dir: str | Path | None = None, log: Callable[[str], None] = print) -> list[ConfigResult]:
    results: dict[str, ConfigResult] = {}
    records: dict[str, list[Record]] = {}
    for c in configs:
        res = ConfigResult(c.name, c.mode.value, c.model)
        results[c.name] = res
        try:
            client = make_client(c.model)
            client.complete(PromptMessages(system="Reply briefly.", user="Reply with the word OK."))  # warm-up, not timed
            pipe = make_pipeline(c, client)
            sampler = make_sampler().start()
            try:
                recs = generate(pipe, questions, progress=lambda i, n: log(f"{c.name}: {i}/{n}") if i % 25 == 0 else None)
            finally:
                usage = sampler.stop()
        except Exception as exc:  # any failure marks this Configuration failed (Req 8.6)
            res.failed, res.cause = True, f"{type(exc).__name__}: {exc}"
            log(f"{c.name} FAILED: {res.cause}")
            continue
        records[c.name] = recs
        res.answering_seconds = usage.duration_s
        res.gpu_mean_power_w = usage.mean_power_w
        res.energy_kwh = usage.energy_kwh
        lat = latency_summary(recs)
        res.p50_latency_ms, res.p95_latency_ms = lat.get("p50_ms"), lat.get("p95_ms")
        if c.hosted:
            price = cost.hosted_prices.get(c.model)
            if price is None:
                res.cost_basis = f"not computed: no recorded per-token price for {c.model}"
            else:
                pt, ct = sum(r.prompt_tokens for r in recs), sum(r.completion_tokens for r in recs)
                res.cost_per_1k_usd = hosted_cost_per_1k(pt, ct, len(recs), price["input_usd_per_mtok"], price["output_usd_per_mtok"])
                res.cost_basis = f"{pt} prompt + {ct} completion tokens at recorded prices"
        elif res.energy_kwh is None:
            res.cost_basis = "not computed: nvidia-smi reported no power draw"
        else:
            res.cost_per_1k_usd = local_cost_per_1k(res.energy_kwh, len(recs), cost.usd_per_kwh)
            res.cost_basis = (f"{res.energy_kwh:.5f} kWh GPU energy over {len(recs)} questions "
                              f"(mean {res.gpu_mean_power_w:.0f} W for {res.answering_seconds:.0f} s) at USD {cost.usd_per_kwh}/kWh")
        if out_dir:
            write_records(Path(out_dir) / f"{c.name}.records.jsonl", recs)
        log(f"{c.name}: answered {len(recs)} questions in {usage.duration_s:.0f} s")

    judge_error = None
    if records:
        try:
            judge = make_judge()
            for name, recs in records.items():
                judge_records(judge, recs, passages_by_id)
                log(f"{name}: judged")
                if out_dir:
                    write_records(Path(out_dir) / f"{name}.records.jsonl", recs)
        except Exception as exc:
            judge_error = f"judge failed: {type(exc).__name__}: {exc}"
            log(judge_error)
    for name, recs in records.items():
        metrics = compute_metrics(recs, questions, seed, judged=judge_error is None)
        if judge_error:
            metrics["faithfulness"] = {"not_computed": judge_error}
        res = results[name]
        res.quality = {m: metrics[m] for m in QUALITY}
        res.retrieval = {m: metrics[m] for m in RETRIEVAL}
        res.counts = counts(recs)
    return [results[c.name] for c in configs]


def results_to_json(results: Sequence[ConfigResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]


def results_from_json(rows: Sequence[dict[str, Any]]) -> list[ConfigResult]:
    return [ConfigResult(**row) for row in rows]


def dump(results: Sequence[ConfigResult]) -> str:
    return json.dumps(results_to_json(results), indent=2, sort_keys=True)
