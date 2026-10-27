"""FastAPI service (Req 9.1-9.9).

Endpoints:
- ``POST /answer``          {"query": "..."} -> CitedAnswer
- ``POST /answer/stream``   server-sent events: ``chunk`` events with validated sentences, then ``done``
- ``GET  /health``          {"status": "ok"} once the index and model client are ready (503 before)
- ``GET  /metrics``         request counts and latency, as JSON

Every request gets a fresh request ID (returned in ``X-Request-ID``), and every log line written
while serving it is a JSON object carrying that ID; the final line records the start time, the
outcome, and the duration. An invalid query (empty, blank, not a string, or longer than
``service.max_query_len``) is rejected with 422 before the pipeline runs and leaves the service's
state, including ``/metrics``, unchanged. A required secret that is missing stops start-up with an
error naming the variable.

Run: ``uvicorn gqa.service:create_app --factory --host 0.0.0.0 --port 8000``
"""

import contextvars
import json
import queue
import sys
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gqa.config import Config, load_config
from gqa.metrics import percentile
from gqa.model_client import ModelClientError, resolve_api_key
from gqa.pipeline import AnsweringPipeline

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
MAX_QUERY_LEN = 1000  # default; the service uses config service.max_query_len


def request_model(max_len: int = MAX_QUERY_LEN) -> type[BaseModel]:
    class AnswerRequest(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")
        query: str = Field(min_length=1, max_length=max_len,
                           description=f"The question: a non-blank string of at most {max_len} characters.")

        @field_validator("query")
        @classmethod
        def not_blank(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("query must contain at least one non-whitespace character")
            return value

    return AnswerRequest


AnswerRequest = request_model()


class CitedAnswer(BaseModel):
    text: str = Field(description="The answer, every sentence citing a retrieved passage, or the refusal text.")
    cited_ids: list[str] = Field(description="IDs of the retrieved passages the answer cites (`doc:chunk:sha1-prefix`).")
    request_id: str = Field(description="The same value as the `X-Request-ID` response header.")
    refused: bool = Field(description="True when the service refused instead of answering.")


class InvalidQueryError(BaseModel):
    """Body of a 422 response: the query was rejected before the pipeline ran."""

    error: str = Field(description='Always "invalid query".')
    detail: list[dict[str, Any]] = Field(description="Each validation failure, as its location and message.")
    request_id: str


class ModelServerErrorBody(BaseModel):
    """Body of a 502 response from ``POST /answer``: the model server failed."""

    error: str = Field(description='Always "model server error".')
    detail: str
    request_id: str


_INVALID = {422: {"model": InvalidQueryError, "description": "Invalid query; the pipeline did not run."}}


class Logger:
    """One JSON object per line on stdout (and to an optional sink, for tests)."""

    def __init__(self, sink: Callable[[dict[str, Any]], None] | None = None, stream=None) -> None:
        self.sink = sink
        self.stream = stream or sys.stdout
        self.lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        entry = {"event": event, "request_id": fields.pop("request_id", None) or _request_id.get(), **fields}
        with self.lock:
            self.stream.write(json.dumps(entry, default=str) + "\n")
            self.stream.flush()
        if self.sink:
            self.sink(entry)


class Metrics:
    """Request counters and latency. Count, mean, and max cover every request; P50 and P95 cover the
    most recent ``window`` requests, so memory stays bounded however long the service runs."""

    def __init__(self, window: int = 10_000) -> None:
        self.lock = threading.Lock()
        self.by_outcome: Counter[str] = Counter()
        self.by_endpoint: Counter[str] = Counter()
        self.latencies: deque[float] = deque(maxlen=window)
        self.latency_count = 0
        self.latency_sum = 0.0
        self.latency_max = 0.0

    def record(self, endpoint: str, outcome: str, duration_ms: float) -> None:
        with self.lock:
            self.by_outcome[outcome] += 1
            self.by_endpoint[endpoint] += 1
            self.latencies.append(duration_ms)
            self.latency_count += 1
            self.latency_sum += duration_ms
            self.latency_max = max(self.latency_max, duration_ms)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            lat = list(self.latencies)
            latency = {"count": self.latency_count}
            if lat:
                latency.update(mean_ms=self.latency_sum / self.latency_count, p50_ms=percentile(lat, 50),
                               p95_ms=percentile(lat, 95), max_ms=self.latency_max)
            return {"requests_total": sum(self.by_endpoint.values()), "by_endpoint": dict(self.by_endpoint),
                    "by_outcome": dict(self.by_outcome), "latency_ms": latency}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(cfg: Config | None = None, pipeline_factory: Callable[[Config], AnsweringPipeline] | None = None,
               log_sink: Callable[[dict[str, Any]], None] | None = None, env: dict[str, str] | None = None) -> FastAPI:
    cfg = cfg or load_config()
    log = Logger(log_sink)
    metrics = Metrics()
    state: dict[str, Any] = {"pipeline": None}
    Query = request_model(cfg.max_query_len)

    def default_factory(c: Config) -> AnsweringPipeline:
        from gqa.runtime import load_backend, make_pipeline

        return make_pipeline(c, load_backend(c), c.service_mode, c.service_model)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ms = cfg.model_server
        try:
            resolve_api_key(ms.api_key_env, ms.api_key_required, env)  # Req 9.9: fail before serving
        except ModelClientError as exc:
            log.emit("startup_failed", error=str(exc), missing_secret=getattr(exc, "variable", None))
            raise
        state["pipeline"] = (pipeline_factory or default_factory)(cfg)
        log.emit("startup", mode=cfg.service_mode, model=cfg.service_model, threshold=state["pipeline"].threshold)
        yield

    app = FastAPI(title="Grounded QA Service", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = uuid.uuid4().hex
        token = _request_id.set(rid)
        request.state.request_id = rid
        request.state.outcome = None
        request.state.start_wall = datetime.now(timezone.utc).isoformat()
        request.state.start = start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            _request_id.reset(token)
        response.headers["X-Request-ID"] = rid
        if request.url.path == "/answer/stream" and response.status_code == 200:
            return response  # the stream logs and counts itself when it ends
        duration_ms = (time.perf_counter() - start) * 1000.0
        outcome = request.state.outcome or ("ok" if response.status_code < 400 else "error")
        if request.url.path.startswith("/answer") and outcome != "invalid":
            metrics.record(request.url.path, outcome, duration_ms)
        log.emit("request", request_id=rid, method=request.method, path=request.url.path, start_time=request.state.start_wall,
                 outcome=outcome, status=response.status_code, duration_ms=round(duration_ms, 3))
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_query(request: Request, exc: RequestValidationError):
        request.state.outcome = "invalid"
        errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": "invalid query", "detail": errors,
                                                      "request_id": request.state.request_id})

    def pipeline() -> AnsweringPipeline:
        return state["pipeline"]

    @app.get("/health", summary="Readiness: ok once the index and model client are loaded",
             responses={503: {"description": 'Still starting: `{"status": "starting"}`.'}})
    def health():
        if state["pipeline"] is None:
            return JSONResponse(status_code=503, content={"status": "starting"})
        return {"status": "ok", "mode": cfg.service_mode, "model": cfg.service_model}

    @app.get("/metrics", summary="Request counts and latency per answering endpoint")
    def get_metrics():
        return metrics.snapshot()

    @app.post("/answer", response_model=CitedAnswer, summary="Answer from retrieved passages, with citations, or refuse",
              responses={**_INVALID, 502: {"model": ModelServerErrorBody, "description": "The model server failed."}})
    def answer(body: Query, request: Request):  # type: ignore[valid-type]
        rid = request.state.request_id
        try:
            trace = pipeline().run(body.query)
        except ModelClientError as exc:
            request.state.outcome = "error"
            log.emit("model_error", request_id=rid, error=str(exc))
            return JSONResponse(status_code=502, content={"error": "model server error", "detail": str(exc), "request_id": rid})
        log.emit("answer", request_id=rid, confidence=round(trace.confidence, 4), refused=trace.answer.refused,
                 refusal_source=trace.refusal_source, cited_ids=list(trace.answer.cited_ids),
                 retrieval_ms=round(trace.retrieval_ms, 3), generation_ms=round(trace.generation_ms, 3))
        request.state.outcome = "refused" if trace.answer.refused else "answered"
        return CitedAnswer(text=trace.answer.text, cited_ids=list(trace.answer.cited_ids), request_id=rid, refused=trace.answer.refused)

    @app.post("/answer/stream", response_class=StreamingResponse, summary="Stream validated sentences as server-sent events",
              responses={**_INVALID, 200: {"content": {"text/event-stream": {"schema": {"type": "string"}}},
                                           "description": "`chunk` events, one validated sentence each, then one `done` event "
                                                          "with the full answer; or one `error` event."}})
    def answer_stream(body: Query, request: Request):  # type: ignore[valid-type]
        rid = request.state.request_id
        timeout = cfg.first_chunk_timeout_s
        events: queue.Queue = queue.Queue()
        started, start_wall = request.state.start, request.state.start_wall

        def produce() -> None:
            token = _request_id.set(rid)
            try:
                for event in pipeline().stream(body.query):
                    events.put(event)
            except ModelClientError as exc:
                events.put(exc)
            finally:
                events.put(None)
                _request_id.reset(token)

        threading.Thread(target=produce, daemon=True).start()

        def body_iter() -> Iterator[str]:
            outcome, first = "error", True
            try:
                while True:
                    try:
                        item = events.get(timeout=timeout if first else cfg.model_server.timeout_s)
                    except queue.Empty:
                        yield _sse("error", {"detail": f"no answer text within {timeout:g} s", "request_id": rid})
                        return
                    if item is None:
                        return
                    if isinstance(item, ModelClientError):
                        log.emit("model_error", request_id=rid, error=str(item))
                        yield _sse("error", {"detail": str(item), "request_id": rid})
                        return
                    if item.kind == "chunk":
                        if first:
                            log.emit("first_chunk", request_id=rid, ms=round((time.perf_counter() - started) * 1000.0, 3))
                        first = False
                        yield _sse("chunk", {"text": item.text})
                    else:
                        a = item.answer
                        outcome = "refused" if a.refused else "answered"
                        log.emit("answer", request_id=rid, refused=a.refused, cited_ids=list(a.cited_ids), stream=True, **item.meta)
                        yield _sse("done", {"text": a.text, "cited_ids": list(a.cited_ids), "refused": a.refused, "request_id": rid})
            finally:
                duration_ms = (time.perf_counter() - started) * 1000.0
                metrics.record("/answer/stream", outcome, duration_ms)
                log.emit("request", request_id=rid, method="POST", path="/answer/stream", start_time=start_wall,
                         outcome=outcome, status=200, duration_ms=round(duration_ms, 3))

        return StreamingResponse(body_iter(), media_type="text/event-stream", headers={"X-Request-ID": rid})

    return app
