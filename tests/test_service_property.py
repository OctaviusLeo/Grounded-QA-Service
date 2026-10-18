"""Feature: grounded-qa-service, Property 14: Request identifier propagation."""

from collections import defaultdict

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gqa.config import load_config
from gqa.service import create_app
from tests.test_service import Q, _pipeline

LOGS: list[dict] = []
APP = create_app(load_config(env={}), pipeline_factory=lambda c: _pipeline(), log_sink=LOGS.append, env={})
CLIENT = TestClient(APP)
CLIENT.__enter__()  # run start-up once for the whole module

requests = st.lists(st.tuples(st.sampled_from(["/answer", "/answer/stream", "/health", "/metrics"]),
                              st.sampled_from([Q, "zzqx blorp", "", "Normans Normandy"])), min_size=1, max_size=6)


@settings(max_examples=100, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(requests)
def test_property_14_request_identifier_propagation(sequence):
    """Feature: grounded-qa-service, Property 14: Request identifier propagation."""
    LOGS.clear()
    ids = []
    for path, query in sequence:
        if path.startswith("/answer"):
            r = CLIENT.post(path, json={"query": query})
            r.read()
        else:
            r = CLIENT.get(path)
        ids.append(r.headers["x-request-id"])
    assert len(set(ids)) == len(ids)  # unique across requests
    by_id = defaultdict(list)
    for entry in LOGS:
        assert entry["request_id"] in ids  # no entry without (or with a foreign) request id
        by_id[entry["request_id"]].append(entry)
    for rid in ids:
        finals = [e for e in by_id[rid] if e["event"] == "request"]
        assert len(finals) == 1
        assert {"start_time", "outcome", "duration_ms"} <= finals[0].keys()
