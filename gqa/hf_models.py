"""Pinned Hugging Face revisions for the models Rag_Lite loads by name (Req 0.5, 0.10).

Rag_Lite asks for ``all-MiniLM-L6-v2`` and ``cross-encoder/ms-marco-MiniLM-L-6-v2`` without a
revision, which would follow whatever ``main`` points to upstream. ``fetch_pinned`` downloads the
pinned commit and points the local cache's ``main`` ref at it; scripts then run with
``HF_HUB_OFFLINE=1`` so nothing can move. ``verify_pins`` is checked before every index build.
"""

from __future__ import annotations

import os
from pathlib import Path

PINNED = {
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
}
LICENSES = {repo: "Apache-2.0" for repo in PINNED}


class PinError(RuntimeError):
    """A cached model is missing or not at its pinned revision."""


def hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _repo_dir(repo: str, cache: Path) -> Path:
    return cache / ("models--" + repo.replace("/", "--"))


def cached_revision(repo: str, cache: Path | None = None) -> str | None:
    ref = _repo_dir(repo, cache or hub_cache()) / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def verify_pins(cache: Path | None = None) -> list[str]:
    """Problems found; empty when every model is cached at its pinned revision."""
    cache = cache or hub_cache()
    problems = []
    for repo, sha in PINNED.items():
        got = cached_revision(repo, cache)
        if got != sha:
            problems.append(f"{repo}: cached main is {got or 'missing'}, pinned {sha}")
        elif not (_repo_dir(repo, cache) / "snapshots" / sha).is_dir():
            problems.append(f"{repo}: snapshot {sha} missing")
    return problems


def require_pins(cache: Path | None = None) -> None:
    problems = verify_pins(cache)
    if problems:
        raise PinError("Hugging Face models not at pinned revisions (run scripts/fetch_hf_models.py): " + "; ".join(problems))


def fetch_pinned(cache: Path | None = None, download=None) -> None:
    """Download each pinned revision and make the cache's ``main`` ref point at it."""
    cache = cache or hub_cache()
    if download is None:  # pragma: no cover - exercised by scripts/fetch_hf_models.py
        from huggingface_hub import snapshot_download as download
    for repo, sha in PINNED.items():
        download(repo_id=repo, revision=sha, cache_dir=str(cache))
        ref = _repo_dir(repo, cache) / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(sha)
