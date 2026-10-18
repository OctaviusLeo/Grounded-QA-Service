import dataclasses
import io
import json
import threading

import pytest
from fastapi.testclient import TestClient

from gqa.config import load_config
from gqa.model_client import MissingSecretError, ModelServerError
from gqa.pipeline import AnsweringPipeline, StreamEvent
from gqa.prompt import PromptBuilder
from gqa.retrieval import RagLiteRetriever, RetrievalMode
from gqa.service import Logger, Metrics, create_app
from gqa.validate import CitationValidator
from tests import fakes

Q = "Where was Super Bowl 50 played?"


def _pipeline(model=None, threshold=0.0):
    return AnsweringPipeline(RagLiteRetriever(RetrievalMode.HYBRID, fakes.backend()), PromptBuilder(),
                             model or fakes.FakeModel(), CitationValidator(), threshold=threshold, k=2, seed=1, candidates=4)


def _app(pipe=None, logs=None, cfg=None, env=None):
    cfg = cfg or load_config(env={})
    return create_app(cfg, pipeline_factory=lambda c: pipe or _pipeline(), log_sink=(logs.append if logs is not None else None), env=env or {})


def test_answer_health_and_request_id():
    logs = []
    with TestClient(_app(logs=logs)) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/answer", json={"query": Q})
        assert r.status_code == 200
        body = r.json()
        assert body["text"] == "Santa Clara [d1:0:00000001]." and body["cited_ids"] == ["d1:0:00000001"] and not body["refused"]
        assert body["request_id"] == r.headers["x-request-id"]
    entries = [e for e in logs if e["request_id"] == body["request_id"]]
    final = [e for e in entries if e["event"] == "request"]
    assert len(final) == 1 and final[0]["outcome"] == "answered" and final[0]["duration_ms"] >= 0 and final[0]["start_time"]
    assert any(e["event"] == "answer" for e in entries)


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "   "}, {"query": 42}, {"query": "x" * 1001}, {}, {"query": Q, "extra": 1}])
def test_invalid_queries_rejected_without_pipeline_and_state_unchanged(payload):
    model = fakes.FakeModel()
    with TestClient(_app(_pipeline(model))) as client:
        before = client.get("/metrics").json()
        r = client.post("/answer", json=payload)
        assert r.status_code == 422 and r.json()["error"] == "invalid query"
        s = client.post("/answer/stream", json=payload)
        assert s.status_code == 422
        assert client.get("/metrics").json() == before
    assert model.calls == []


def test_non_json_body_is_invalid():
    with TestClient(_app()) as client:
        r = client.post("/answer", content=b"not json", headers={"content-type": "application/json"})
        assert r.status_code == 422 and r.json()["error"] == "invalid query"


def test_metrics_shape():
    with TestClient(_app()) as client:
        client.post("/answer", json={"query": Q})
        client.post("/answer", json={"query": "zzqx blorp"})
        m = client.get("/metrics").json()
    assert m["requests_total"] == 2 and m["by_endpoint"] == {"/answer": 2}
    assert set(m["by_outcome"]) <= {"answered", "refused"} and sum(m["by_outcome"].values()) == 2
    assert m["latency_ms"]["count"] == 2 and m["latency_ms"]["p50_ms"] <= m["latency_ms"]["p95_ms"] <= m["latency_ms"]["max_ms"]


def _events(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_stream_first_chunk_and_reassembly():
    reply = "Santa Clara [d1:0:00000001]. It was a game [d1:0:00000001]."
    logs = []
    with TestClient(_app(_pipeline(fakes.FakeModel(reply)), logs=logs)) as client:
        with client.stream("POST", "/answer/stream", json={"query": Q}) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            events = _events(r.read().decode())
        metrics = client.get("/metrics").json()
    chunks = [d["text"] for e, d in events if e == "chunk"]
    done = [d for e, d in events if e == "done"]
    assert len(chunks) == 2 and len(done) == 1 and "".join(chunks) == done[0]["text"] == reply
    rid = done[0]["request_id"]
    assert r.headers["x-request-id"] == rid
    assert any(e["event"] == "first_chunk" and e["request_id"] == rid for e in logs)
    final = [e for e in logs if e["event"] == "request" and e["request_id"] == rid]
    assert len(final) == 1 and final[0]["outcome"] == "answered"
    assert metrics["by_endpoint"] == {"/answer/stream": 1}


def test_stream_first_chunk_timeout_and_model_error():
    cfg = dataclasses.replace(load_config(env={}), first_chunk_timeout_s=0.2)
    gate = threading.Event()

    class Slow:
        threshold = 0.0

        def stream(self, query):
            gate.wait(5)
            yield StreamEvent("chunk", "late")

    with TestClient(create_app(cfg, pipeline_factory=lambda c: Slow(), env={})) as client:
        events = _events(client.post("/answer/stream", json={"query": Q}).text)
    gate.set()
    assert events[0][0] == "error" and "0.2 s" in events[0][1]["detail"]
    with TestClient(_app(_pipeline(fakes.FakeModel(error=ModelServerError("down"))))) as client:
        events = _events(client.post("/answer/stream", json={"query": Q}).text)
        assert events == [("error", {"detail": "down", "request_id": events[0][1]["request_id"]})]
        r = client.post("/answer", json={"query": Q})
        assert r.status_code == 502 and r.json()["error"] == "model server error"


def test_missing_secret_stops_startup_and_names_variable():
    cfg = load_config(env={})
    cfg = dataclasses.replace(cfg, model_server=dataclasses.replace(cfg.model_server, api_key_required=True))
    logs = []
    app = create_app(cfg, pipeline_factory=lambda c: pytest.fail("pipeline must not be built"), log_sink=logs.append, env={})
    with pytest.raises(MissingSecretError, match=cfg.model_server.api_key_env):
        with TestClient(app):
            pass
    assert logs[-1]["event"] == "startup_failed" and logs[-1]["missing_secret"] == cfg.model_server.api_key_env
    ok = create_app(cfg, pipeline_factory=lambda c: _pipeline(), env={cfg.model_server.api_key_env: "set"})
    with TestClient(ok) as client:
        assert client.get("/health").status_code == 200


def test_health_before_startup_and_helpers():
    client = TestClient(_app())  # no context manager: lifespan has not run
    assert client.get("/health").status_code == 503
    stream = io.StringIO()
    Logger(stream=stream).emit("x", request_id="r1", a=1)
    assert json.loads(stream.getvalue()) == {"event": "x", "request_id": "r1", "a": 1}
    assert Metrics().snapshot() == {"requests_total": 0, "by_endpoint": {}, "by_outcome": {}, "latency_ms": {"count": 0}}
