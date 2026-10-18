"""Environment and Model_Server smoke test (Req 1.1-1.8, 0.4).

One run: check Python >= 3.11 and Docker, capture ``nvidia-smi``, start the Model_Server, send
exactly one question, sample peak GPU memory while the answer is generated, and require a
non-empty answer within 120 seconds. The evidence (or the failed condition and its cause) is
rendered as a block for ``docs/ENVIRONMENT.md``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from gqa.gpu import GpuError, GpuReading, GpuSampler, nvidia_smi_report, query_gpus
from gqa.model_client import ModelClient, ModelClientError
from gqa.repro import Stamp, stamp_markdown

QUESTION = "In one sentence, what does a retrieval-augmented question-answering system do?"
TIMEOUT_S = 120.0


@dataclass
class SmokeDeps:
    run: Callable[[list[str]], subprocess.CompletedProcess]
    gpu_query: Callable[[], list[GpuReading]]
    gpu_report: Callable[[], str]
    http_get: Callable[[str], Any]
    make_client: Callable[[str, float], Any]
    http_post: Callable[[str, dict], Any] = lambda path, body: {}
    sleep: Callable[[float], None] = time.sleep
    python_version: tuple[int, int, int] = tuple(sys.version_info[:3])  # type: ignore[assignment]


@dataclass
class SmokeResult:
    model: str
    ok: bool = False
    failed_condition: str | None = None
    cause: str | None = None
    python_version: str = ""
    docker_version: str = ""
    server_version: str = ""
    nvidia_smi: str = ""
    gpu_name: str = ""
    total_mib: int = 0
    baseline_used_mib: int = 0
    peak_used_mib: int = 0
    model_vram_bytes: int = 0
    model_details: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    elapsed_s: float = 0.0


def default_deps(base_url: str, api_key_env: str, seed: int) -> SmokeDeps:
    root = base_url.rstrip("/").removesuffix("/v1")

    def http_get(path: str) -> Any:
        return httpx.get(root + path, timeout=10).json()

    def http_post(path: str, body: dict) -> Any:
        return httpx.post(root + path, json=body, timeout=60).json()

    def make_client(model: str, timeout_s: float) -> ModelClient:
        return ModelClient(base_url, model, api_key_env, timeout_s, seed, max_tokens=128)

    def run(args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, check=False, timeout=600)

    return SmokeDeps(run=run, gpu_query=query_gpus, gpu_report=nvidia_smi_report,
                     http_get=http_get, make_client=make_client, http_post=http_post)


def unload_all(deps: SmokeDeps, wait_s: float = 30.0) -> None:
    """Unload every resident model so the baseline reading excludes earlier models."""
    for loaded in deps.http_get("/api/ps").get("models", []):
        deps.http_post("/api/generate", {"model": loaded["name"], "keep_alive": 0})
    waited = 0.0
    while deps.http_get("/api/ps").get("models") and waited < wait_s:
        deps.sleep(0.5)
        waited += 0.5


def _fail(result: SmokeResult, condition: str, cause: str) -> SmokeResult:
    result.ok = False
    result.failed_condition = condition
    result.cause = cause
    return result


def run_smoke(model: str, deps: SmokeDeps, start_server: bool = True, timeout_s: float = TIMEOUT_S) -> SmokeResult:
    result = SmokeResult(model=model)
    major, minor, micro = deps.python_version
    result.python_version = f"{major}.{minor}.{micro}"
    if (major, minor) < (3, 11):
        return _fail(result, "python>=3.11", f"Python {result.python_version} is older than 3.11")

    proc = deps.run(["docker", "version", "--format", "{{.Server.Version}}"])
    if proc.returncode != 0:
        return _fail(result, "docker available", (proc.stderr or proc.stdout).strip() or "docker version failed")
    result.docker_version = proc.stdout.strip()

    try:
        gpu0 = deps.gpu_query()[0]
    except GpuError as exc:
        return _fail(result, "nvidia-smi available", str(exc))
    result.gpu_name, result.total_mib = gpu0.name, gpu0.memory_total_mib

    if start_server:
        proc = deps.run(["docker", "compose", "up", "-d", "--wait", "ollama"])
        if proc.returncode != 0:
            return _fail(result, "model server started", (proc.stderr or proc.stdout).strip()[-500:])
    try:
        result.server_version = str(deps.http_get("/api/version").get("version", ""))
        tags = deps.http_get("/api/tags").get("models", [])
    except (httpx.HTTPError, ValueError) as exc:
        return _fail(result, "model server reachable", str(exc))
    entry = next((m for m in tags if m.get("name") == model), None)
    if entry is None:
        return _fail(result, "model pulled", f"{model} is not in the server's model list; run scripts/pull_models.sh")
    result.model_details = {
        "digest": entry.get("digest", ""),
        "size_bytes": entry.get("size", 0),
        **{k: v for k, v in (entry.get("details") or {}).items() if k in ("family", "parameter_size", "quantization_level", "format")},
    }

    try:
        unload_all(deps)
    except (httpx.HTTPError, ValueError) as exc:
        return _fail(result, "model server reachable", f"cannot unload resident models: {exc}")

    try:
        result.nvidia_smi = deps.gpu_report()
    except GpuError as exc:
        return _fail(result, "nvidia-smi available", str(exc))
    client = deps.make_client(model, timeout_s)
    sampler = GpuSampler(interval_s=0.2, query=deps.gpu_query)
    sampler.start()
    started = time.perf_counter()
    try:
        completion = client.complete([{"role": "user", "content": QUESTION}])
    except ModelClientError as exc:
        sampler.stop()
        return _fail(result, f"answer within {timeout_s:g} s", f"{type(exc).__name__}: {exc}")
    finally:
        result.elapsed_s = time.perf_counter() - started
    usage = sampler.stop()
    result.baseline_used_mib = usage.baseline_memory_used_mib
    result.peak_used_mib = usage.peak_memory_used_mib
    result.answer = completion.text.strip()
    try:
        loaded = deps.http_get("/api/ps").get("models", [])
        result.model_vram_bytes = int(next((m.get("size_vram", 0) for m in loaded if m.get("name") == model), 0))
    except (httpx.HTTPError, ValueError):
        result.model_vram_bytes = 0

    if result.elapsed_s > timeout_s:
        return _fail(result, f"answer within {timeout_s:g} s", f"answer took {result.elapsed_s:.1f} s")
    if not result.answer:
        return _fail(result, "non-empty answer", "the model returned only whitespace")
    if result.peak_used_mib > result.total_mib:
        return _fail(result, "peak GPU memory <= total", f"peak {result.peak_used_mib} MiB > total {result.total_mib} MiB")
    result.ok = True
    return result


def render_block(result: SmokeResult, stamp: Stamp, when: str) -> str:
    lines = [f"### Smoke test: `{result.model}`", "", stamp_markdown(stamp) + f" · run at {when}", ""]
    if not result.ok:
        lines += [
            f"- **Status: FAILED** — condition `{result.failed_condition}` not met.",
            f"- Cause: {result.cause}",
        ]
    else:
        lines += ["- **Status: passed**"]
    lines += [
        f"- Python: {result.python_version}; Docker Engine: {result.docker_version or 'n/a'}; Ollama server: {result.server_version or 'n/a'}",
        f"- GPU: {result.gpu_name or 'n/a'}, total memory {result.total_mib} MiB (nvidia-smi)",
    ]
    if result.model_details:
        d = result.model_details
        lines.append(
            f"- Model: `{result.model}` digest `{d.get('digest', '')}`, {d.get('parameter_size', '?')} parameters, "
            f"quantization {d.get('quantization_level', '?')}, format {d.get('format', '?')}, {d.get('size_bytes', 0)} bytes"
        )
    if result.ok or result.peak_used_mib:
        lines += [
            f"- GPU memory in use before the request: {result.baseline_used_mib} MiB; peak during generation: "
            f"**{result.peak_used_mib} MiB** of {result.total_mib} MiB; model resident in VRAM per Ollama: "
            f"{result.model_vram_bytes / 2**20:.0f} MiB",
            f"- Time from request to answer: {result.elapsed_s:.1f} s (limit {TIMEOUT_S:g} s, includes loading the model)",
        ]
    if result.answer:
        lines += ["", f"Question: {QUESTION}", "", "Answer:", "", "> " + result.answer.replace("\n", "\n> ")]
    if result.nvidia_smi:
        lines += ["", "<details><summary>nvidia-smi before the request</summary>", "", "```text", result.nvidia_smi.rstrip(), "```", "", "</details>"]
    return "\n".join(lines)
