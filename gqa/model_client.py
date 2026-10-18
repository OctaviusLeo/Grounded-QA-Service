"""OpenAI-compatible chat client for the Model_Server (Req 0.13, 1.6, 9.3, 9.9).

The client is server-agnostic: it speaks the ``/v1/chat/completions`` protocol that Ollama (and
llama.cpp, vLLM, or a hosted API) implement. Secrets come only from environment variables; a
required key that is unset or empty raises :class:`MissingSecretError` in the constructor, before
any request can be sent. Every request carries temperature 0 and a fixed seed.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class ModelClientError(RuntimeError):
    """Base class for Model_Server failures."""


class MissingSecretError(ModelClientError):
    """A required secret environment variable is unset or empty."""

    def __init__(self, variable: str) -> None:
        super().__init__(f"required secret environment variable {variable} is unset or empty")
        self.variable = variable


class ModelTimeoutError(ModelClientError):
    """The Model_Server did not answer within the timeout."""


class ModelServerError(ModelClientError):
    """The Model_Server could not be reached or returned an error."""


class HasMessages(Protocol):
    system: str
    user: str


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float


def resolve_api_key(variable: str, required: bool, env: Mapping[str, str] | None = None) -> str | None:
    """Read a secret from the environment; fail fast when it is required but missing."""
    value = (os.environ if env is None else env).get(variable, "")
    if not value.strip():
        if required:
            raise MissingSecretError(variable)
        return None
    return value


def to_messages(messages: HasMessages | Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if hasattr(messages, "system") and hasattr(messages, "user"):
        return [
            {"role": "system", "content": messages.system},  # type: ignore[union-attr]
            {"role": "user", "content": messages.user},  # type: ignore[union-attr]
        ]
    return [dict(m) for m in messages]  # type: ignore[union-attr]


class ModelClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str,
        timeout_s: float,
        seed: int,
        *,
        api_key_required: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.0,
        transport: httpx.BaseTransport | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        key = resolve_api_key(api_key_env, api_key_required, env)  # before any request (Req 0.13)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.model = model
        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
        )

    def _payload(self, messages: HasMessages | Sequence[Mapping[str, str]], stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_messages(messages),
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def complete(self, messages: HasMessages | Sequence[Mapping[str, str]]) -> Completion:
        start = time.perf_counter()
        try:
            response = self._client.post("/chat/completions", json=self._payload(messages, stream=False))
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(f"no answer from {self.model} within {self.timeout_s:g} s") from exc
        except httpx.HTTPError as exc:
            raise ModelServerError(f"cannot reach the model server: {exc}") from exc
        if response.status_code != 200:
            raise ModelServerError(f"model server returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelServerError(f"malformed completion response: {response.text[:300]}") from exc
        usage = body.get("usage") or {}
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            elapsed_s=time.perf_counter() - start,
        )

    def stream(self, messages: HasMessages | Sequence[Mapping[str, str]]) -> Iterator[str]:
        """Yield answer text deltas as the server produces them."""
        try:
            with self._client.stream("POST", "/chat/completions", json=self._payload(messages, stream=True)) as response:
                if response.status_code != 200:
                    response.read()
                    raise ModelServerError(f"model server returned HTTP {response.status_code}: {response.text[:300]}")
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except ValueError as exc:
                        raise ModelServerError(f"malformed stream chunk: {data[:200]}") from exc
                    for choice in chunk.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            yield delta
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(f"stream from {self.model} stalled for {self.timeout_s:g} s") from exc
        except httpx.HTTPError as exc:
            raise ModelServerError(f"cannot reach the model server: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ModelClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
