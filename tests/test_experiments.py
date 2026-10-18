import pytest
from gqa import experiments as ex
from gqa.gpu import SamplerResult
from gqa.model_client import ModelServerError
from gqa.pipeline import AnsweringPipeline
from gqa.prompt import PromptBuilder
from gqa.retrieval import RagLiteRetriever, RetrievalMode
from gqa.validate import CitationValidator
from tests import fakes
from tests.test_eval import QUESTIONS, reply


class FakeSampler:
    def start(self):
        return self

    def stop(self):
        return SamplerResult(samples=3, duration_s=36.0, peak_memory_used_mib=5000, baseline_memory_used_mib=2000, mean_power_w=200.0)


def _run(configs, client_factory, judge_reply="SUPPORTED", cost=None, tmp_path=None):
    backend = fakes.backend()

    def make_pipeline(c, client):
        return AnsweringPipeline(RagLiteRetriever(c.mode, backend), PromptBuilder(), client, CitationValidator(),
                                 threshold=0.1, k=2, seed=1, candidates=4)

    return ex.run_experiments(
        configs, QUESTIONS, seed=1, make_pipeline=make_pipeline, make_client=client_factory,
        make_judge=lambda: __import__("gqa.judge", fromlist=["LLMJudge"]).LLMJudge(fakes.FakeModel(judge_reply)),
        make_sampler=FakeSampler, passages_by_id=backend.by_id,
        cost=cost or ex.CostSettings(0.20, "test price", {"hosted-m": {"input_usd_per_mtok": 3.0, "output_usd_per_mtok": 15.0}}),
        out_dir=tmp_path, log=lambda s: None,
    )


CONFIGS = [ex.Configuration("bm25_big", RetrievalMode.BM25, "big"), ex.Configuration("hybrid_small", RetrievalMode.HYBRID, "small")]


def test_runner_reports_quality_retrieval_latency_and_cost(tmp_path):
    results = _run(CONFIGS, lambda m: fakes.FakeModel(reply, model=m), tmp_path=tmp_path)
    assert [r.config for r in results] == ["bm25_big", "hybrid_small"]
    for r in results:
        assert not r.failed and set(r.quality) == set(ex.QUALITY) and set(r.retrieval) == set(ex.RETRIEVAL)
        assert all("point" in v for v in r.retrieval.values())
        assert r.p50_latency_ms <= r.p95_latency_ms
        # 200 W for 36 s = 0.002 kWh; at $0.20/kWh over 3 questions -> $0.1333 per 1,000.
        assert r.energy_kwh == pytest.approx(0.002) and r.cost_per_1k_usd == pytest.approx(0.002 * 0.2 / 3 * 1000)
        assert r.quality["faithfulness"]["point"] == 1.0 and r.counts["questions"] == 3
        assert (tmp_path / f"{r.config}.records.jsonl").exists()
    round_trip = ex.results_from_json(ex.results_to_json(results))
    assert round_trip == results and '"config": "bm25_big"' in ex.dump(results)


def test_failed_configuration_is_reported_and_others_continue():
    def factory(model):
        if model == "small":
            raise ModelServerError("model 'small' not found")
        return fakes.FakeModel(reply, model=model)

    results = _run(CONFIGS, factory)
    ok, bad = results
    assert not ok.failed and "point" in ok.quality["exact_match"]
    assert bad.failed and "model 'small' not found" in bad.cause and bad.quality == {}


def test_judge_failure_marks_faithfulness_not_computed():
    class BrokenJudge(fakes.FakeModel):
        def complete(self, messages):
            raise ModelServerError("judge down")

    results = ex.run_experiments(
        CONFIGS[:1], QUESTIONS, seed=1,
        make_pipeline=lambda c, client: AnsweringPipeline(RagLiteRetriever(c.mode, fakes.backend()), PromptBuilder(), client,
                                                           CitationValidator(), threshold=0.1, k=2, seed=1, candidates=4),
        make_client=lambda m: fakes.FakeModel(reply, model=m),
        make_judge=lambda: __import__("gqa.judge", fromlist=["LLMJudge"]).LLMJudge(BrokenJudge()),
        make_sampler=FakeSampler, passages_by_id=fakes.backend().by_id, cost=ex.CostSettings(0.2, "x"), log=lambda s: None)
    assert "judge down" in results[0].quality["faithfulness"]["not_computed"]
    assert "point" in results[0].quality["f1"]


def test_hosted_configuration_cost_from_token_prices():
    results = _run([ex.Configuration("hosted", RetrievalMode.BM25, "hosted-m", hosted=True)], lambda m: fakes.FakeModel(reply, model=m))
    r = results[0]
    c = r.counts
    expected = ex.hosted_cost_per_1k(c["prompt_tokens"], c["completion_tokens"], 3, 3.0, 15.0)
    assert r.cost_per_1k_usd == pytest.approx(expected) and "tokens at recorded prices" in r.cost_basis
    missing = _run([ex.Configuration("h2", RetrievalMode.BM25, "unpriced", hosted=True)], lambda m: fakes.FakeModel(reply, model=m))
    assert missing[0].cost_per_1k_usd is None and "no recorded per-token price" in missing[0].cost_basis


def test_no_power_reading_leaves_cost_not_computed():
    class NoPower(FakeSampler):
        def stop(self):
            return SamplerResult(1, 1.0, 1, 1, None)

    backend = fakes.backend()
    r = ex.run_experiments(CONFIGS[:1], QUESTIONS, 1,
                           make_pipeline=lambda c, client: AnsweringPipeline(RagLiteRetriever(c.mode, backend), PromptBuilder(), client, CitationValidator(), 0.1, 2, 1, 4),
                           make_client=lambda m: fakes.FakeModel(reply, model=m), make_judge=lambda: None,
                           make_sampler=NoPower, passages_by_id=backend.by_id, cost=ex.CostSettings(0.2, "x"), log=lambda s: None)[0]
    assert r.cost_per_1k_usd is None and "no power draw" in r.cost_basis
    assert "judge failed" in r.quality["faithfulness"]["not_computed"]


def test_load_experiments_and_formatting(tmp_path):
    configs, cost, raw = ex.load_experiments("config/experiments.yaml")
    assert len(configs) >= 4 and {c.mode for c in configs} == set(RetrievalMode)
    assert {c.model for c in configs if c.mode is RetrievalMode.HYBRID_RERANK} == {"granite3.3:2b", "granite3.3:8b"}
    assert 0 < cost.usd_per_kwh < 1 and "EIA" in cost.source
    p = tmp_path / "e.yaml"
    p.write_text("configurations:\n  - {name: a, mode: bm25, model: m}\n  - {name: a, mode: bm25, model: m}\ncost: {electricity_usd_per_kwh: 0.1, electricity_source: s}\n")
    with pytest.raises(ValueError, match="unique"):
        ex.load_experiments(p)
    assert ex.format_usd(1.5) == "1.50" and ex.format_usd(0.004321) == "0.0043" and ex.format_usd(0.0123) == "0.0123" and ex.format_usd(0) == "0.00"
    with pytest.raises(ValueError):
        ex.local_cost_per_1k(1.0, 0, 0.1)
    with pytest.raises(ValueError):
        ex.hosted_cost_per_1k(1, 1, 0, 1, 1)
