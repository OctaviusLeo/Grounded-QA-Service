import math

import pytest
from src.evaluate import calculate_ndcg_at_k  # Rag_Lite

from gqa import metrics


def test_recall_and_ndcg_examples():
    gold = {"a", "b"}
    assert metrics.recall_at_k(["x", "a", "y", "b"], gold, 2) == 0.5
    assert metrics.recall_at_k(["x", "a", "y", "b"], gold, 5) == 1.0
    assert metrics.recall_at_k(["x"], set(), 5) == 0.0
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert metrics.ndcg_at_k(["x", "a"], gold, 2) == pytest.approx(expected)
    # The duplicate "a" earns nothing, so "b" sits at rank 3.
    assert metrics.ndcg_at_k(["a", "a", "b"], gold, 5) == pytest.approx((1 + 0.5) / (1 + 1 / math.log2(3)))
    assert metrics.ndcg_at_k(["a"], set(), 5) == 0.0


def test_ndcg_matches_rag_lite_on_full_lists_and_fixes_short_lists():
    gold = {"a", "c"}
    full = ["b", "a", "d", "c", "e"]
    assert metrics.ndcg_at_k(full, gold, 5) == pytest.approx(calculate_ndcg_at_k(sorted(gold), full))
    short = ["a"]  # the scoring floor returned one result
    assert calculate_ndcg_at_k(sorted(gold), short) == 1.0  # Rag_Lite's ideal DCG uses len(list)
    assert metrics.ndcg_at_k(short, gold, 5) == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_bad_k_rejected():
    with pytest.raises(ValueError):
        metrics.recall_at_k([], {"a"}, 0)
    with pytest.raises(ValueError):
        metrics.ndcg_at_k([], {"a"}, -1)


def test_bootstrap_ci_basics():
    ci = metrics.bootstrap_ci([0, 1] * 50, seed=1)
    assert ci.point == 0.5 and ci.lower < 0.5 < ci.upper and ci.n == 100 and ci.resamples == 1000
    assert metrics.BootstrapCI.from_dict(ci.as_dict()) == ci
    const = metrics.bootstrap_ci([0.3] * 10, seed=2)
    assert const.lower == const.point == const.upper == pytest.approx(0.3)


def test_bootstrap_ci_custom_statistic_and_errors():
    median = metrics.bootstrap_ci([1, 2, 3, 4, 100], seed=0, statistic=lambda a: __import__("numpy").median(a, axis=-1))
    assert median.point == 3.0
    with pytest.raises(ValueError, match="1000"):
        metrics.bootstrap_ci([1.0], seed=0, resamples=999)
    with pytest.raises(ValueError, match="empty"):
        metrics.bootstrap_ci([], seed=0)
    with pytest.raises(ValueError, match="confidence"):
        metrics.bootstrap_ci([1.0], seed=0, confidence=1.0)
