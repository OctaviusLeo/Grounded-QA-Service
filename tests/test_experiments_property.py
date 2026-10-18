"""Feature: grounded-qa-service, Properties 12 and 13 (latency percentiles, hosted cost)."""

from hypothesis import given, settings
from hypothesis import strategies as st

from gqa import experiments as ex
from gqa.metrics import percentile


@settings(max_examples=100, derandomize=True)
@given(st.lists(st.floats(min_value=0, max_value=1e6, allow_nan=False), min_size=1, max_size=200))
def test_property_12_latency_percentile_ordering(latencies):
    """Feature: grounded-qa-service, Property 12: Latency percentile ordering."""
    assert percentile(latencies, 50) <= percentile(latencies, 95)


@settings(max_examples=100, derandomize=True)
@given(st.integers(0, 10**9), st.integers(0, 10**8), st.integers(1, 10**5),
       st.floats(min_value=0, max_value=100, allow_nan=False), st.floats(min_value=0, max_value=200, allow_nan=False))
def test_property_13_hosted_cost_formula(prompt_tokens, completion_tokens, questions, price_in, price_out):
    """Feature: grounded-qa-service, Property 13: Hosted cost computed from the defining formula."""
    cost = ex.hosted_cost_per_1k(prompt_tokens, completion_tokens, questions, price_in, price_out)
    formula = (prompt_tokens * price_in / 1e6 + completion_tokens * price_out / 1e6) * 1000 / questions
    assert abs(cost - formula) <= 1e-6 * max(1.0, abs(formula))
    text = ex.format_usd(cost)
    assert len(text.split(".")[1]) >= 2
