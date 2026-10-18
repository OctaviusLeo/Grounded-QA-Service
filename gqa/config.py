"""Configuration loading and validation (Req 0.2, 4.6).

``config/default.yaml`` holds every tunable value, including the seed. Refusal thresholds are
tuned by ``scripts/tune_thresholds.py`` and stored per retrieval mode in ``config/thresholds.yaml``;
each must be a single number in [0, 1].
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MODES = ("bm25", "hybrid", "hybrid_rerank")


class ConfigError(ValueError):
    """A configuration value is missing or out of range."""


@dataclass(frozen=True)
class ModelServerConfig:
    base_url: str
    api_key_env: str
    api_key_required: bool
    timeout_s: float
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class Config:
    seed: int
    corpus_path: str
    evalset_path: str
    tuning_path: str
    thresholds_path: str
    chunk_size: int
    overlap: int
    k: int
    candidates: int
    hybrid_weights: dict[str, float]
    model_server: ModelServerConfig
    models: dict[str, str]
    default_threshold: float
    service_model: str
    service_mode: str
    max_query_len: int
    first_chunk_timeout_s: float
    thresholds: dict[str, float] = field(default_factory=dict)

    def threshold_for(self, mode: str) -> float:
        return self.thresholds.get(mode, self.default_threshold)


def validate_threshold(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number in [0, 1], got {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise ConfigError(f"{name} must lie in [0, 1], got {value}")
    return float(value)


def _require(data: dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"missing configuration value: {dotted}")
        node = node[part]
    return node


def load_thresholds(path: str | Path) -> dict[str, float]:
    """Read tuned per-mode thresholds; a missing file means none are tuned yet."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = data.get("thresholds", {})
    out = {}
    for mode, value in raw.items():
        if mode not in MODES:
            raise ConfigError(f"unknown retrieval mode in {p}: {mode}")
        out[mode] = validate_threshold(value, f"thresholds.{mode}")
    return out


def load_config(path: str | Path = "config/default.yaml", env: dict[str, str] | None = None) -> Config:
    env = dict(os.environ if env is None else env)
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} is not a mapping")
    seed = _require(data, "seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"seed must be an integer, got {seed!r}")
    ms = _require(data, "model_server")
    server = ModelServerConfig(
        base_url=env.get("GQA_MODEL_BASE_URL", _require(data, "model_server.base_url")),
        api_key_env=str(_require(data, "model_server.api_key_env")),
        api_key_required=bool(_require(data, "model_server.api_key_required")),
        timeout_s=float(_require(data, "model_server.timeout_s")),
        max_tokens=int(_require(data, "model_server.max_tokens")),
        temperature=float(ms.get("temperature", 0.0)),
    )
    weights = {k: float(v) for k, v in _require(data, "retrieval.hybrid_weights").items()}
    mode = env.get("GQA_SERVICE_MODE", _require(data, "service.mode"))
    if mode not in MODES:
        raise ConfigError(f"service.mode must be one of {MODES}, got {mode!r}")
    k = int(_require(data, "retrieval.k"))
    candidates = int(_require(data, "retrieval.candidates"))
    if k < 1 or candidates < k:
        raise ConfigError(f"need 1 <= retrieval.k <= retrieval.candidates, got k={k}, candidates={candidates}")
    thresholds_path = str(_require(data, "paths.thresholds"))
    return Config(
        seed=seed,
        corpus_path=str(_require(data, "paths.corpus")),
        evalset_path=str(_require(data, "paths.evalset")),
        tuning_path=str(_require(data, "paths.tuning")),
        thresholds_path=thresholds_path,
        chunk_size=int(_require(data, "chunking.chunk_size")),
        overlap=int(_require(data, "chunking.overlap")),
        k=k,
        candidates=candidates,
        hybrid_weights=weights,
        model_server=server,
        models={k2: str(v) for k2, v in _require(data, "models").items()},
        default_threshold=validate_threshold(_require(data, "refusal.default_threshold"), "refusal.default_threshold"),
        service_model=env.get("GQA_SERVICE_MODEL", _require(data, "service.model")),
        service_mode=mode,
        max_query_len=int(_require(data, "service.max_query_len")),
        first_chunk_timeout_s=float(_require(data, "service.first_chunk_timeout_s")),
        thresholds=load_thresholds(thresholds_path),
    )
