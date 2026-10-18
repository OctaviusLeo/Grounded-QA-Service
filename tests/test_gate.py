import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gqa import gate

COMMIT = "c" * 40


def _baseline(tmp_path, **over):
    body = {"faithfulness": 0.8, "citation_precision": 0.7, "stamp": {"command": "python -m gqa.eval", "seed": 1, "commit": COMMIT}}
    body.update(over)
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(body))
    return p


def _subset(tmp_path, faith=0.8, cp=0.7, failed=None):
    metrics = {"faithfulness": {"point": faith} if faith is not None else {"not_computed": "judge disabled"},
               "citation_precision": {"point": cp}}
    p = tmp_path / "subset.json"
    p.write_text(json.dumps({"metrics": metrics, **({"failed": failed} if failed else {})}))
    return p


def test_missing_baseline_fails_with_cause(tmp_path):
    failed, report = gate.run_gate(_subset(tmp_path), tmp_path / "nope.json")
    assert failed and "Baseline_File missing" in report


@pytest.mark.parametrize("content", ["{not json", json.dumps({"faithfulness": 0.8}),
                                     json.dumps({"faithfulness": "x", "citation_precision": 1, "stamp": {"command": "c", "seed": 1, "commit": COMMIT}}),
                                     json.dumps({"faithfulness": 0.8, "citation_precision": 0.7, "stamp": {"command": "c", "seed": 1, "commit": "short"}})])
def test_unreadable_baseline_fails_with_cause(tmp_path, content):
    p = tmp_path / "baseline.json"
    p.write_text(content)
    failed, report = gate.run_gate(_subset(tmp_path), p)
    assert failed and "Baseline_File unreadable" in report


def test_pass_fail_and_not_computed(tmp_path):
    base = _baseline(tmp_path)
    assert gate.run_gate(_subset(tmp_path, 0.75, 0.65), base) == (False, gate.run_gate(_subset(tmp_path, 0.75, 0.65), base)[1])
    failed, report = gate.run_gate(_subset(tmp_path, 0.65, 0.7), base)
    assert failed and "FAIL faithfulness" in report and "PASS citation_precision" in report
    failed, report = gate.run_gate(_subset(tmp_path, None, 0.7), base)
    assert failed and "not computed" in report
    failed, report = gate.run_gate(_subset(tmp_path, failed="ModelServerError: down"), base)
    assert failed and "subset run failed" in report


def test_thresholds_recorded_in_repository():
    th = gate.load_thresholds("config/ci.yaml")
    assert set(th) == set(gate.GATED) and all(0 <= v <= 1 for v in th.values())


def test_negative_threshold_rejected(tmp_path):
    p = tmp_path / "ci.yaml"
    p.write_text("regression_threshold: {faithfulness: -0.1, citation_precision: 0.1}\n")
    with pytest.raises(ValueError):
        gate.load_thresholds(p)


unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@settings(max_examples=100, derandomize=True)
@given(unit, unit, unit, unit, st.floats(min_value=0.0, max_value=0.5, allow_nan=False))
def test_property_15_ci_regression_gate_decision(sub_f, base_f, sub_p, base_p, threshold):
    """Feature: grounded-qa-service, Property 15: CI regression gate decision."""
    baseline = gate.BaselineFile(base_f, base_p, "cmd", 1, COMMIT)
    failed, report = gate.check_regression({"faithfulness": sub_f, "citation_precision": sub_p}, baseline, threshold)
    expected_f = (base_f - sub_f) > threshold
    expected_p = (base_p - sub_p) > threshold
    assert failed == (expected_f or expected_p)
    assert ("FAIL faithfulness" in report) == expected_f
    assert ("FAIL citation_precision" in report) == expected_p
