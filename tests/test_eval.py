import json

import pytest

from gqa import eval as harness
from gqa.evalset import EvalQuestion, write_questions
from gqa.judge import LLMJudge
from gqa.model_client import ModelServerError, ModelTimeoutError
from gqa.prompt import PromptBuilder
from gqa.retrieval import RagLiteRetriever, RetrievalMode
from gqa.pipeline import AnsweringPipeline
from gqa.validate import CitationValidator
from tests import fakes

QUESTIONS = [
    EvalQuestion("q1", "Where was Super Bowl 50 played?", "Santa Clara", ("d1:0:00000001",), True, ("Santa Clara",)),
    EvalQuestion("q2", "When did the Norman conquest of England begin?", "1066", ("d3:0:00000003",), True, ("1066",)),
    EvalQuestion("u1", "Who won the zzqx blorp prize?", None, (), False),
]


def reply(messages):
    if "Super Bowl" in messages.user.split("Question:")[-1]:
        return "Santa Clara [d1:0:00000001]."
    if "Norman" in messages.user.split("Question:")[-1]:
        return "It began in 1066 [d3:0:00000003]."
    return "not found in the documents"


def _pipe(model=None, threshold=0.1):
    return AnsweringPipeline(RagLiteRetriever(RetrievalMode.HYBRID, fakes.backend()), PromptBuilder(),
                             model or fakes.FakeModel(reply), CitationValidator(), threshold=threshold, k=2, seed=1, candidates=4)


def test_generate_judge_and_metrics():
    records = harness.generate(_pipe(), QUESTIONS)
    assert [r.refused for r in records] == [False, False, True]
    assert records[2].refusal_source == "confidence"
    harness.judge_records(LLMJudge(fakes.FakeModel("SUPPORTED")), records, fakes.backend().by_id)
    m = harness.compute_metrics(records, QUESTIONS, seed=3)
    assert list(m) == harness.METRIC_ORDER
    assert m["exact_match"]["point"] == pytest.approx(0.5)  # "It began in 1066" is not an exact match
    assert m["f1"]["point"] > 0.5 and m["recall@5"]["point"] == 1.0
    assert m["citation_precision"]["point"] == 1.0 and m["refusal_accuracy"]["point"] == 1.0
    assert m["faithfulness"]["point"] == 1.0 and m["answerable_refused"]["point"] == 0.0
    lat = harness.latency_summary(records)
    assert lat["n"] == 3 and lat["p50_ms"] <= lat["p95_ms"]
    c = harness.counts(records)
    assert c["answered"] == 2 and c["judged"] == 2 and c["refused_by_confidence"] == 1


def test_missing_inputs_are_not_computed_and_rest_continue():
    only_unanswerable = [QUESTIONS[2]]
    records = harness.generate(_pipe(), only_unanswerable)
    m = harness.compute_metrics(records, only_unanswerable, seed=0, judged=False)
    assert m["recall@5"] == {"not_computed": "no answerable questions with gold Passage_IDs"}
    assert m["exact_match"]["not_computed"] and m["faithfulness"] == {"not_computed": "judge disabled for this run"}
    assert m["refusal_accuracy"]["point"] == 1.0  # still computed
    no_gold_answers = [EvalQuestion("q1", QUESTIONS[0].question, None, ("d1:0:00000001",), True)]
    m2 = harness.compute_metrics(harness.generate(_pipe(), no_gold_answers), no_gold_answers, seed=0)
    assert "not_computed" in m2["exact_match"] and "point" in m2["recall@5"]
    assert m2["refusal_accuracy"] == {"not_computed": "no unanswerable questions"}
    assert harness.latency_summary([]) == {"n": 0}


def test_unparseable_judge_verdicts_and_errors():
    records = harness.generate(_pipe(), QUESTIONS[:1])
    harness.judge_records(LLMJudge(fakes.FakeModel("maybe")), records, fakes.backend().by_id)
    m = harness.compute_metrics(records, QUESTIONS[:1], seed=0)
    assert "not_computed" in m["faithfulness"] and harness.counts(records)["judge_unparseable"] == 1
    with pytest.raises(ModelServerError):
        harness.generate(_pipe(fakes.FakeModel(error=ModelServerError("down"))), QUESTIONS[:1])


def test_timeout_is_retried_once():
    class Flaky(fakes.FakeModel):
        def complete(self, messages):
            if not self.calls:
                self.calls.append(messages)
                raise ModelTimeoutError("slow")
            return super().complete(messages)

    records = harness.generate(_pipe(Flaky(reply)), QUESTIONS[:1])
    assert records[0].answer == "Santa Clara [d1:0:00000001]."
    with pytest.raises(ModelTimeoutError):
        harness.generate(_pipe(fakes.FakeModel(error=ModelTimeoutError("slow"))), QUESTIONS[:1])


def test_lenient_loader_reports_malformed_lines(tmp_path):
    path = tmp_path / "q.jsonl"
    write_questions(path, QUESTIONS)
    with path.open("a") as fh:
        fh.write('{"question": "missing fields"}\n\n')
    qs, problems = harness.load_questions_lenient(path)
    assert len(qs) == 3 and len(problems) == 1 and "line 4" in problems[0]
    assert harness.load_questions_lenient(tmp_path / "none.jsonl") == ([], [f"{tmp_path / 'none.jsonl'} does not exist"])


def test_records_round_trip(tmp_path):
    records = harness.generate(_pipe(), QUESTIONS)
    harness.write_records(tmp_path / "r.jsonl", records)
    assert harness.read_records(tmp_path / "r.jsonl") == records


@pytest.fixture
def repo_env(tmp_path, monkeypatch):
    """Run main() against a temporary question file without touching results/."""
    monkeypatch.setattr(harness, "make_stamp", lambda seed: __import__("gqa.repro", fromlist=["Stamp"]).Stamp("cmd", seed, "e" * 40))
    qpath = tmp_path / "q.jsonl"
    write_questions(qpath, QUESTIONS)
    return qpath


def test_main_single_command_end_to_end(repo_env, tmp_path):
    out = tmp_path / "out" / "run.json"
    rc = harness.main(["--questions", str(repo_env), "--mode", "hybrid", "--model", "m", "--seed", "5", "--out", str(out)],
                      backend_factory=fakes.backend,
                      client_factory=lambda m: fakes.FakeModel(reply if m == "m" else "SUPPORTED", model=m))
    assert rc == 0
    body = json.loads(out.read_text())
    assert body["stamp"]["seed"] == 5 and body["model"] == "m" and body["counts"]["questions"] == 3
    assert body["metrics"]["faithfulness"]["n"] == 2
    assert out.with_suffix(".records.jsonl").exists()


def test_main_reports_failure_and_missing_questions(repo_env, tmp_path):
    out = tmp_path / "fail.json"
    rc = harness.main(["--questions", str(repo_env), "--mode", "bm25", "--out", str(out), "--no-judge"],
                      backend_factory=fakes.backend, client_factory=lambda m: fakes.FakeModel(error=ModelServerError("down")))
    assert rc == 1 and "ModelServerError" in json.loads(out.read_text())["failed"]
    rc = harness.main(["--questions", str(tmp_path / "missing.jsonl"), "--out", str(out)], backend_factory=fakes.backend)
    body = json.loads(out.read_text())
    assert rc == 1 and all("not_computed" in v for v in body["metrics"].values())


def test_ci_subset_selection(tmp_path):
    from gqa.config import load_config

    ci = tmp_path / "ci.yaml"
    ci.write_text("subset: {answerable: 1, unanswerable: 1}\n")
    qpath = tmp_path / "q.jsonl"
    write_questions(qpath, QUESTIONS)
    qs, problems = harness.select_questions(load_config(env={}), str(qpath), "ci", ci_config=str(ci))
    assert [q.question_id for q in qs] == ["q1", "u1"] and problems == []
    ci.write_text("subset: {answerable: 5, unanswerable: 1}\n")
    qs, problems = harness.select_questions(load_config(env={}), str(qpath), "ci", ci_config=str(ci))
    assert problems and "too small" in problems[0]
