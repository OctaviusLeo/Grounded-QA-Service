"""Feature: grounded-qa-service, Properties 7 and 8 (metric ranges, bootstrap CIs)."""

from hypothesis import given, settings
from hypothesis import strategies as st

from gqa import metrics

IDS = [f"p{i}" for i in range(12)]
id_lists = st.lists(st.sampled_from(IDS), max_size=25)
id_sets = st.frozensets(st.sampled_from(IDS), max_size=6)


@settings(max_examples=100, derandomize=True)
@given(id_lists, id_sets, st.sampled_from([5, 10, 20]))
def test_property_7_metric_ranges(retrieved, gold, k):
    """Feature: grounded-qa-service, Property 7: Metric ranges."""
    for value in (metrics.recall_at_k(retrieved, gold, k), metrics.ndcg_at_k(retrieved, gold, k)):
        assert 0.0 <= value <= 1.0 + 1e-12
    # Identity case: a list that starts with every gold passage scores 1.0 when they fit in k.
    if gold and len(gold) <= k:
        perfect = sorted(gold) + [r for r in retrieved if r not in gold]
        assert metrics.recall_at_k(perfect, gold, k) == 1.0
        assert abs(metrics.ndcg_at_k(perfect, gold, k) - 1.0) < 1e-12


samples = st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=1, max_size=60)


@settings(max_examples=100, derandomize=True, deadline=None)
@given(samples, st.integers(0, 2**31 - 1))
def test_property_8_bootstrap_ci_brackets_point_and_is_deterministic(values, seed):
    """Feature: grounded-qa-service, Property 8: Bootstrap CI brackets the point estimate."""
    a = metrics.bootstrap_ci(values, seed=seed, resamples=1000)
    b = metrics.bootstrap_ci(values, seed=seed, resamples=1000)
    assert a.lower <= a.point <= a.upper
    assert abs(a.lower - b.lower) <= 1e-6 and abs(a.upper - b.upper) <= 1e-6 and abs(a.point - b.point) <= 1e-6
