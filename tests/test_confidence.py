import math

import pytest

from gqa.config import ConfigError, load_config
from gqa.confidence import estimate_confidence, should_refuse
from gqa.retrieval import RetrievalMode, RetrievedPassage


def _r(*scores):
    return [RetrievedPassage(f"p{i}", s, i) for i, s in enumerate(scores)]


def test_empty_retrieval_is_zero_confidence():
    for mode in RetrievalMode:
        assert estimate_confidence([], mode) == 0.0


def test_rerank_uses_logistic_of_top_logit():
    assert estimate_confidence(_r(0.0, -3.0), "hybrid_rerank") == 0.5
    assert estimate_confidence(_r(10.0), "hybrid_rerank") == pytest.approx(1 / (1 + math.exp(-10)))
    assert 0.0 <= estimate_confidence(_r(-800.0), "hybrid_rerank") < 1e-300  # no overflow


@pytest.mark.parametrize("scores, expected", [((0.42, 0.1), 0.42), ((1.7,), 1.0), ((-0.2,), 0.0), ((float("nan"),), 0.0)])
def test_fused_scores_are_clipped(scores, expected):
    assert estimate_confidence(_r(*scores), RetrievalMode.HYBRID) == expected


def test_confidence_always_in_unit_interval():
    for mode in RetrievalMode:
        for s in (-1e9, -5, 0, 0.3, 5, 1e9):
            assert 0.0 <= estimate_confidence(_r(s), mode) <= 1.0


def test_should_refuse_and_threshold_validation():
    assert should_refuse(0.2, 0.3) and not should_refuse(0.3, 0.3)
    with pytest.raises(ConfigError):
        should_refuse(0.5, 1.2)


def test_threshold_comes_from_config_in_unit_interval():
    cfg = load_config("config/default.yaml", env={})
    for mode in RetrievalMode:
        t = cfg.threshold_for(mode.value)
        assert isinstance(t, float) and 0.0 <= t <= 1.0
