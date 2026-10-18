"""Record exact versions and licenses with the artifact that proves each (Req 0.5, 0.6, 0.7).

Every row names the command that was run on this machine to obtain the value, so a reader can
re-run it. Python package licenses come from the installed distributions' metadata.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata

Runner = Callable[[list[str]], str]


def run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, check=False, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


@dataclass(frozen=True)
class ToolVersion:
    name: str
    version: str
    artifact: str


def tool_versions(runner: Runner = run) -> list[ToolVersion]:
    """Versions of the tools the project runs, each from a query on this machine."""
    probes = [
        ("Python (project environment)", ["python", "--version"], r"Python (\S+)"),
        ("uv", ["uv", "--version"], r"uv (\S+)"),
        ("Docker Engine", ["docker", "version", "--format", "{{.Server.Version}}"], r"(\S+)"),
        ("Docker Compose plugin", ["docker", "compose", "version", "--short"], r"(\S+)"),
        ("NVIDIA Container Toolkit", ["nvidia-ctk", "--version"], r"version (\S+)"),
        ("NVIDIA driver (Windows host)", ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], r"(\S+)"),
        ("WSL kernel", ["uname", "-r"], r"(\S+)"),
        ("Ubuntu (WSL distro)", ["grep", "PRETTY_NAME", "/etc/os-release"], r'"(.+)"'),
        ("Git for Windows", ["git.exe", "--version"], r"git version (\S+)"),
        ("GitHub CLI", ["gh.exe", "--version"], r"gh version (\S+)"),
    ]
    out = []
    for name, args, pattern in probes:
        try:
            text = runner(args)
            m = re.search(pattern, text)
            version = m.group(1) if m else text.splitlines()[0]
        except (RuntimeError, OSError, IndexError) as exc:
            version = f"unavailable ({exc})"
        out.append(ToolVersion(name, version, "`" + " ".join(args) + "`"))
    return out


def license_of(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return expr.strip()
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    classifiers = [c for c in classifiers if c not in ("OSI Approved",)]
    if classifiers:
        return "; ".join(dict.fromkeys(classifiers))
    raw = (meta.get("License") or "").strip()
    if raw and "\n" not in raw and len(raw) <= 60:
        return raw
    if raw:
        first = raw.splitlines()[0].strip()
        return first[:60] + ("…" if len(first) > 60 else "")
    return "UNKNOWN"


@dataclass(frozen=True)
class PackageInfo:
    name: str
    version: str
    license: str


def installed_packages(dists: Iterable[metadata.Distribution] | None = None) -> list[PackageInfo]:
    seen = {}
    for dist in dists if dists is not None else metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        seen[name.lower()] = PackageInfo(name, dist.version, license_of(dist))
    return sorted(seen.values(), key=lambda p: p.name.lower())


def direct_dependencies(pyproject_text: str) -> set[str]:
    names = set()
    for m in re.finditer(r'^\s*"([A-Za-z0-9_.\-]+)==', pyproject_text, re.M):
        names.add(m.group(1).lower())
    return names


def render_tools(tools: list[ToolVersion]) -> str:
    rows = ["| Tool | Version | Artifact (command run on this machine) |", "| --- | --- | --- |"]
    rows += [f"| {t.name} | {t.version} | {t.artifact} |" for t in tools]
    return "\n".join(rows)


def render_packages(pkgs: list[PackageInfo], direct: set[str], artifact: str) -> str:
    rows = [
        f"Artifact: {artifact}. Direct dependencies (pinned with `==` in `pyproject.toml`) are bold; the rest are locked transitively in `uv.lock`.",
        "",
        "| Package | Version | License |",
        "| --- | --- | --- |",
    ]
    for p in pkgs:
        name = f"**{p.name}**" if p.name.lower() in direct else p.name
        rows.append(f"| {name} | {p.version} | {p.license} |")
    return "\n".join(rows)


def ollama_models(http_get: Callable[[str], dict]) -> list[dict]:
    return http_get("/api/tags").get("models", [])


def model_license_id(license_text: str) -> str:
    if "Apache License" in license_text and "Version 2.0" in license_text:
        return "Apache-2.0"
    return "see license text"


def render_json_block(data: object) -> str:
    return "```json\n" + json.dumps(data, indent=2, sort_keys=True) + "\n```"
