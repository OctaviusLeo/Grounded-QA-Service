import subprocess

import httpx
import pytest

from gqa import smoke
from gqa.docblock import read_block, replace_block, update_file
from gqa.gpu import GpuError, GpuReading
from gqa.model_client import Completion, ModelTimeoutError
from gqa.repro import Stamp

STAMP = Stamp(command="python scripts/smoke_test.py --model m", seed=1, commit="c" * 40)


class FakeClient:
    def __init__(self, text="A system that answers from retrieved passages.", exc=None):
        self.text, self.exc = text, exc

    def complete(self, messages):
        if self.exc:
            raise self.exc
        return Completion(self.text, 10, 5, 0.1)


def _deps(**over):
    def run(args):
        if args[:2] == ["docker", "version"]:
            return subprocess.CompletedProcess(args, 0, "29.9.1\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    def http_get(path):
        return {
            "/api/version": {"version": "0.40.6"},
            "/api/tags": {"models": [{"name": "m", "digest": "d" * 64, "size": 5, "details": {"parameter_size": "8.2B", "quantization_level": "Q4_K_M", "format": "gguf"}}]},
            "/api/ps": {"models": [{"name": "m", "size_vram": 6 * 2**30}]},
        }[path]

    values = dict(
        run=run,
        gpu_query=lambda: [GpuReading("RTX", 10240, 3000, 100.0)],
        gpu_report=lambda: "NVIDIA-SMI table",
        http_get=http_get,
        make_client=lambda model, timeout: FakeClient(),
        python_version=(3, 12, 15),
        sleep=lambda s: None,
    )
    values.update(over)
    return smoke.SmokeDeps(**values)


def test_smoke_passes_and_renders_evidence():
    r = smoke.run_smoke("m", _deps())
    assert r.ok and r.answer and r.peak_used_mib == 3000 and r.total_mib == 10240
    assert r.model_details["quantization_level"] == "Q4_K_M"
    block = smoke.render_block(r, STAMP, "now")
    assert "Status: passed" in block and "3000 MiB" in block and "c" * 40 in block and "NVIDIA-SMI table" in block


@pytest.mark.parametrize(
    "over, condition",
    [
        ({"python_version": (3, 10, 4)}, "python>=3.11"),
        ({"run": lambda a: subprocess.CompletedProcess(a, 1, "", "Cannot connect to the Docker daemon")}, "docker available"),
        ({"make_client": lambda m, t: FakeClient(exc=ModelTimeoutError("no answer within 120 s"))}, "answer within 120 s"),
        ({"make_client": lambda m, t: FakeClient(text="   ")}, "non-empty answer"),
        ({"gpu_query": lambda: [GpuReading("RTX", 100, 300, None)]}, "peak GPU memory <= total"),
    ],
)
def test_smoke_failures_record_condition(over, condition):
    r = smoke.run_smoke("m", _deps(**over))
    assert not r.ok and r.failed_condition == condition and r.cause
    block = smoke.render_block(r, STAMP, "now")
    assert "FAILED" in block and condition in block


def test_smoke_reports_missing_gpu_model_and_server():
    def no_gpu():
        raise GpuError("nvidia-smi not found on PATH")

    assert smoke.run_smoke("m", _deps(gpu_report=no_gpu)).failed_condition == "nvidia-smi available"
    assert smoke.run_smoke("m", _deps(gpu_query=no_gpu)).failed_condition == "nvidia-smi available"
    assert smoke.run_smoke("other", _deps()).failed_condition == "model pulled"

    def down(path):
        raise httpx.ConnectError("refused")

    assert smoke.run_smoke("m", _deps(http_get=down)).failed_condition == "model server reachable"

    def compose_fails(args):
        if args[:2] == ["docker", "version"]:
            return subprocess.CompletedProcess(args, 0, "29\n", "")
        return subprocess.CompletedProcess(args, 1, "", "no such service")

    assert smoke.run_smoke("m", _deps(run=compose_fails)).failed_condition == "model server started"
    assert smoke.run_smoke("m", _deps(run=compose_fails), start_server=False).ok


def test_docblock_insert_replace_and_read(tmp_path):
    text = "# Env\n\n## Tools\n\nprose\n\n## Smoke test evidence\n"
    once = replace_block(text, "x", "one", after_heading="## Tools")
    assert once.index("<!-- BEGIN x -->") < once.index("## Smoke test evidence")
    twice = replace_block(once, "x", "two")
    assert read_block(twice, "x") == "two" and "one" not in twice
    appended = replace_block("abc", "y", "z")
    assert appended.endswith("<!-- END y -->\n")
    assert read_block("nothing", "x") is None
    tail = replace_block(text, "s", "evidence", after_heading="## Smoke test evidence")
    assert tail.rstrip().endswith("<!-- END s -->")
    # A second block in the same section goes after the first, even if the first has headings.
    two = replace_block(replace_block(text, "a", "### A\nbody", after_heading="## Tools"), "b", "### B", after_heading="## Tools")
    assert two.index("<!-- END a -->") < two.index("<!-- BEGIN b -->") < two.index("## Smoke test evidence")
    p = tmp_path / "doc.md"
    update_file(p, "k", "v")
    assert read_block(p.read_text(), "k") == "v"


def test_default_deps_wiring():
    d = smoke.default_deps("http://127.0.0.1:8090/v1", "GQA_MODEL_API_KEY", 1)
    client = d.make_client("m", 5.0)
    assert client.model == "m" and client.max_tokens == 128
    client.close()
    assert d.run(["true"]).returncode == 0


def test_unload_all_waits_until_nothing_is_resident():
    state = {"loaded": [{"name": "big"}], "posts": [], "polls": 0}

    def http_get(path):
        state["polls"] += 1
        if state["polls"] > 2:
            state["loaded"] = []
        return {"models": state["loaded"]}

    d = _deps(http_get=http_get, http_post=lambda p, b: state["posts"].append((p, b)), sleep=lambda s: None)
    smoke.unload_all(d)
    assert state["posts"] == [("/api/generate", {"model": "big", "keep_alive": 0})]
    assert state["loaded"] == []


def test_unload_failure_is_reported():
    base = _deps().http_get

    def http_get(path):
        if path == "/api/ps":
            raise httpx.ConnectError("down")
        return base(path)

    result = smoke.run_smoke("m", _deps(http_get=http_get))
    assert result.failed_condition == "model server reachable"
    assert result.cause.startswith("cannot unload resident models")
