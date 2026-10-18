import json

import httpx
import pytest

from gqa.model_client import (
    MissingSecretError,
    ModelClient,
    ModelServerError,
    ModelTimeoutError,
    resolve_api_key,
    to_messages,
)


class Msgs:
    system = "sys"
    user = "usr"


def _client(handler, **kw):
    kw.setdefault("env", {})
    return ModelClient(
        "http://server/v1", "m", "GQA_KEY", 5.0, 42, transport=httpx.MockTransport(handler), **kw
    )


def test_missing_required_secret_fails_before_any_request():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={})

    with pytest.raises(MissingSecretError) as err:
        _client(handler, api_key_required=True, env={"GQA_KEY": "  "})
    assert "GQA_KEY" in str(err.value)
    assert err.value.variable == "GQA_KEY"
    assert calls == []


def test_optional_secret_may_be_absent_and_present_key_is_sent():
    assert resolve_api_key("X", required=False, env={}) is None
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

    c = _client(handler, api_key_required=True, env={"GQA_KEY": "secret"})
    out = c.complete(Msgs())
    assert out.text == "hi" and out.prompt_tokens == 3 and out.completion_tokens == 1
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["seed"] == 42 and seen["body"]["temperature"] == 0.0
    assert seen["body"]["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    assert seen["body"]["stream"] is False


def test_timeout_surfaces_as_typed_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ModelTimeoutError, match="within 5 s"):
        _client(handler).complete(Msgs())


def test_connection_error_and_http_error_are_typed():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ModelServerError, match="cannot reach"):
        _client(refuse).complete(Msgs())

    with pytest.raises(ModelServerError, match="HTTP 500"):
        _client(lambda r: httpx.Response(500, text="boom")).complete(Msgs())

    with pytest.raises(ModelServerError, match="malformed"):
        _client(lambda r: httpx.Response(200, json={"choices": []})).complete(Msgs())


def _sse(*chunks):
    lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
    return "\n\n".join(lines) + "\n\n"


def test_stream_reassembles_deltas():
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
        text = _sse(
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [], "usage": {"prompt_tokens": 1}},
        )
        return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})

    assert list(_client(handler).stream([{"role": "user", "content": "x"}])) == ["Hel", "lo"]


def test_stream_errors_are_typed():
    with pytest.raises(ModelServerError, match="HTTP 404"):
        list(_client(lambda r: httpx.Response(404, text="nope")).stream(Msgs()))
    with pytest.raises(ModelServerError, match="malformed stream chunk"):
        list(_client(lambda r: httpx.Response(200, text="data: {bad\n\n")).stream(Msgs()))

    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ModelTimeoutError):
        list(_client(slow).stream(Msgs()))

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ModelServerError, match="cannot reach"):
        list(_client(refuse).stream(Msgs()))


def test_to_messages_accepts_dicts_and_context_manager_closes():
    assert to_messages([{"role": "user", "content": "q"}]) == [{"role": "user", "content": "q"}]
    with _client(lambda r: httpx.Response(200, json={})) as c:
        assert c.model == "m"
