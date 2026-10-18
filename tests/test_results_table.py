import json
import re

import yaml

from gqa import results_table as rt
from gqa.experiments import QUALITY, RETRIEVAL
from gqa.repro import Stamp, stamp_to_dict

STAMP = Stamp("python scripts/x.py", 1234, "a" * 40)
CI = {"point": 0.5, "lower": 0.4, "upper": 0.6, "n": 10, "resamples": 1000, "confidence": 0.95}


def _write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    body["stamp"] = stamp_to_dict(STAMP)
    path.write_text(json.dumps(body))


def _config(name, failed=False):
    if failed:
        return {"config": name, "mode": "bm25", "model": "m", "quality": {}, "retrieval": {}, "p50_latency_ms": None,
                "p95_latency_ms": None, "cost_per_1k_usd": None, "cost_basis": "", "failed": True, "cause": "ModelServerError: down"}
    quality = {m: dict(CI) for m in QUALITY}
    quality["faithfulness"] = {"not_computed": "judge failed"}
    return {"config": name, "mode": "hybrid_rerank", "model": "granite3.3:8b", "quality": quality,
            "retrieval": {m: dict(CI) for m in RETRIEVAL}, "p50_latency_ms": 1234.4, "p95_latency_ms": 2345.6,
            "cost_per_1k_usd": 0.0123, "cost_basis": "0.001 kWh", "failed": False, "cause": None}


def _fixture(tmp_path):
    _write(tmp_path / "experiments.json", {"questions": 200, "answerable": 150, "unanswerable": 50, "resamples": 1000,
                                           "judge_model": "granite3.3:8b", "electricity_source": "EIA",
                                           "configurations": [_config("a"), _config("b"), _config("c"), _config("d", failed=True)]})
    _write(tmp_path / "retrieval_baseline.json", {"questions": 150, "ks": [5, 10, 20],
                                                  "modes": {"bm25": {f"{m}@{k}": CI for m in ("recall", "ndcg") for k in (5, 10, 20)}}})
    _write(tmp_path / "injection.json", {"mode": "hybrid_rerank", "model": "m", "cases": 20, "followed": 3, "not_followed": 15,
                                         "unclassified": 2, "followed_fraction": 3 / 18})
    th = tmp_path / "thresholds.yaml"
    th.write_text(yaml.safe_dump({"questions": 100, "tuning_model": "m", "rule": "r", "thresholds": {"bm25": 0.0},
                                  "tuning_split_scores": {"bm25": {"balanced_accuracy": 0.5, "unanswerable_refused": 0.1, "answerable_answered": 0.9}},
                                  "stamp": stamp_to_dict(STAMP)}))
    return th


def test_regenerating_is_deterministic_and_has_cis(tmp_path):
    th = _fixture(tmp_path)
    out = tmp_path / "summary.md"
    a = rt.write_results_table(STAMP, out, results_dir=tmp_path, thresholds=th)
    b = rt.write_results_table(STAMP, out, results_dir=tmp_path, thresholds=th)
    assert a == b == out.read_text()
    quality_rows = [line for line in a.splitlines() if line.startswith("| a |") or line.startswith("| b |") or line.startswith("| c |")]
    assert len(quality_rows) == 6  # three healthy configurations in each of the two tables
    for row in quality_rows:
        assert row.count("[0.400, 0.600]") >= 6
    assert "not computed (judge failed)" in a
    assert "| d | bm25 | `m` | failed" in a and "FAILED: ModelServerError: down" in a
    assert "0.0123" in a and "| 1234 | 2346 |" in a
    assert a.count("a" * 40) == 5  # every section carries its stamp
    assert "0.167" in a and "not computed: owner labels pending" in a


def test_missing_and_pending_sources(tmp_path):
    text = rt.render(STAMP, results_dir=tmp_path, thresholds=tmp_path / "none.yaml")
    assert "Not computed: " in text and "experiments.json not found" in text and "Not tuned" in text
    (tmp_path / "experiments.json").write_text("{bad")
    assert "unreadable" in rt.render(STAMP, results_dir=tmp_path, thresholds=tmp_path / "none.yaml")
    _write(tmp_path / "judge" / "agreement.json", {"n": 0, "kappa": {"not_computed": "owner labels pending"}})
    assert "not computed: owner labels pending" in rt.render(STAMP, results_dir=tmp_path)
    _write(tmp_path / "judge" / "agreement.json", {"n": 50, "owner_faithful_rate": 0.4, "judge_faithful_rate": 0.6,
                                                   "kappa": {"point": 0.61, "lower": 0.4, "upper": 0.8}})
    assert re.search(r"Cohen's kappa over 50 samples: 0\.610 \[0\.400, 0\.800\]", rt.render(STAMP, results_dir=tmp_path))


def test_cell_formats():
    assert rt.cell(None) == "not computed (missing)"
    assert rt.cell({"not_computed": "x"}) == "not computed (x)"
    assert rt.cell(CI) == "0.500 [0.400, 0.600]"
