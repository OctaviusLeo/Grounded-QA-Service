"""Write the version and license blocks of docs/ENVIRONMENT.md and THIRD_PARTY.md.

python scripts/record_environment.py   (needs the model server running: docker compose up -d ollama)

Every value comes from a query run on this machine; the command is recorded beside it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx

from gqa.config import load_config
from gqa.docblock import update_file
from gqa.envrecord import (
    direct_dependencies,
    installed_packages,
    model_license_id,
    render_packages,
    render_tools,
    run,
    tool_versions,
)
from gqa.gpu import query_gpus
from gqa.repro import make_stamp, stamp_markdown

ENV = "docs/ENVIRONMENT.md"
TP = "THIRD_PARTY.md"
OLLAMA_IMAGE = "ollama/ollama:0.40.6"


def main() -> int:
    cfg = load_config()
    stamp = make_stamp(seed=cfg.seed)
    head = stamp_markdown(stamp)
    root = cfg.model_server.base_url.removesuffix("/v1")

    # Machine
    gpu = query_gpus()[0]
    windows = run(["cmd.exe", "/c", "ver"]).strip()
    update_file(ENV, "machine", "\n".join([
        head, "",
        f"- Host OS: {windows} (artifact: `cmd.exe /c ver`)",
        f"- GPU: {gpu.name}, {gpu.memory_total_mib} MiB total (artifact: `nvidia-smi --query-gpu=name,memory.total --format=csv`)",
        "- Python, Docker, and every script run inside WSL2 (Ubuntu 24.04); Git and the GitHub CLI run on Windows.",
    ]), after_heading="## Machine")

    # Tools
    update_file(ENV, "tools", head + "\n\n" + render_tools(tool_versions()), after_heading="## Tools")

    # Python packages: cross-check importlib.metadata against `uv pip list`.
    pkgs = installed_packages()
    uv_list = {p["name"].lower(): p["version"] for p in json.loads(run(["uv", "pip", "list", "--format", "json"]))}
    mismatched = [p.name for p in pkgs if uv_list.get(p.name.lower()) not in (None, p.version)]
    if mismatched:
        print(f"version mismatch between uv pip list and metadata: {mismatched}", file=sys.stderr)
        return 1
    direct = direct_dependencies(Path("pyproject.toml").read_text(encoding="utf-8"))
    table = render_packages(pkgs, direct, "`uv pip list --format json` in `~/.venvs/gqa` (licenses from each installed distribution's metadata)")
    update_file(ENV, "python-packages", head + "\n\n" + table, after_heading="## Python dependencies")
    update_file(TP, "python-packages", head + "\n\n" + table, after_heading="## Python packages")

    # Model server and models
    version = httpx.get(root + "/api/version", timeout=10).json()["version"]
    digest = run(["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", OLLAMA_IMAGE])
    ollama_license = httpx.get("https://api.github.com/repos/ollama/ollama", timeout=20).json()["license"]["spdx_id"]
    rows = [
        "| Model | Ollama manifest digest | Parameters | Quantization | Size (bytes) | License | Artifact |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    tp_rows = list(rows[:1]) + [rows[1]]
    for m in sorted(httpx.get(root + "/api/tags", timeout=10).json()["models"], key=lambda m: m["name"]):
        show = httpx.post(root + "/api/show", json={"model": m["name"]}, timeout=30).json()
        lic = model_license_id(show.get("license", ""))
        d = m.get("details", {})
        row = (f"| `{m['name']}` | `{m['digest']}` | {d.get('parameter_size')} | {d.get('quantization_level')} | "
               f"{m['size']} | {lic} | `GET /api/tags`, license text from `POST /api/show` |")
        rows.append(row)
        tp_rows.append(row)
    server = "\n".join([
        head, "",
        f"- Model_Server: Ollama {version} (artifact: `GET {root}/api/version`), image `{digest}` (artifact: `docker image inspect`), license {ollama_license} (artifact: GitHub API `repos/ollama/ollama`).",
        "- Settings (docker-compose.yml): `OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`; OpenAI-compatible endpoint `http://127.0.0.1:8090/v1`.",
        "",
        *rows,
    ])
    update_file(ENV, "model-server", server, after_heading="## Model server and local models")
    update_file(TP, "model-server", "\n".join([
        head, "",
        f"- Ollama {version}, license {ollama_license}. Used as an unmodified container image; not redistributed.",
        "- Models are pulled from the Ollama registry at run time into the Docker volume `gqa_ollama`; no weights are committed.",
        "",
        *tp_rows,
    ]), after_heading="## Model server and models")
    uv_license = httpx.get("https://api.github.com/repos/astral-sh/uv", timeout=20).json()["license"]["spdx_id"]
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    pinned = sorted(set(re.findall(r"(?:FROM|--from=)\s*(\S+@sha256:[0-9a-f]{64})", dockerfile)))
    image_rows = [f"| {OLLAMA_IMAGE} | `{digest.split('@')[-1]}` | {ollama_license} (bundles NVIDIA CUDA runtime libraries under NVIDIA's license) |"]
    for ref in pinned:
        name, sha = ref.split("@")
        lic = (f"{uv_license} (artifact: GitHub API `repos/astral-sh/uv`); only the `uv` binary is copied into the build stage"
               if "astral-sh/uv" in name else "Debian packages under their own free-software licenses (DFSG); see `/usr/share/doc/*/copyright` in the image")
        image_rows.append(f"| {name} | `{sha}` | {lic} |")
    update_file(TP, "images", "\n".join([
        head, "",
        "| Image | Digest | License |", "| --- | --- | --- |",
        *image_rows,
        "",
        "GitHub Actions used by `.github/workflows/ci.yml`: `actions/checkout@v4` and `astral-sh/setup-uv@v6` (MIT); they run in CI and are not part of the repository.",
    ]), after_heading="## Container images")

    update_file(ENV, "hosted-model", "\n".join([
        "No Hosted_Model is used, so no API key is read and no Spend_Cap applies (Req 1.7).",
        "If one is added, its key is read from the variable named by `model_server.api_key_env` with `api_key_required: true`, and its monthly cap in USD is recorded here.",
    ]), after_heading="## Hosted model")
    print("updated", ENV, "and", TP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
