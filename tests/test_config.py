from pathlib import Path

import pytest
import yaml

from gqa import config as cfg

DEFAULT = Path("config/default.yaml")


def _write(tmp_path, mutate, thresholds=None):
    data = yaml.safe_load(DEFAULT.read_text())
    th_path = tmp_path / "thresholds.yaml"
    data["paths"]["thresholds"] = str(th_path)
    if thresholds is not None:
        th_path.write_text(yaml.safe_dump({"thresholds": thresholds}))
    mutate(data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_default_config_loads():
    c = cfg.load_config(DEFAULT, env={})
    assert isinstance(c.seed, int)
    assert 0.0 <= c.default_threshold <= 1.0
    assert c.k <= c.candidates
    assert c.service_mode in cfg.MODES


def test_env_overrides_base_url_and_service(tmp_path):
    path = _write(tmp_path, lambda d: None)
    c = cfg.load_config(path, env={"GQA_MODEL_BASE_URL": "http://ollama:11434/v1", "GQA_SERVICE_MODEL": "m", "GQA_SERVICE_MODE": "bm25"})
    assert c.model_server.base_url == "http://ollama:11434/v1"
    assert c.service_model == "m"
    assert c.service_mode == "bm25"


def test_thresholds_read_from_file(tmp_path):
    path = _write(tmp_path, lambda d: None, thresholds={"hybrid_rerank": 0.25, "bm25": 1.0})
    c = cfg.load_config(path, env={})
    assert c.threshold_for("hybrid_rerank") == 0.25
    assert c.threshold_for("bm25") == 1.0
    assert c.threshold_for("hybrid") == c.default_threshold


@pytest.mark.parametrize("bad", [-0.1, 1.5, "0.3", True, None])
def test_threshold_out_of_range_rejected(tmp_path, bad):
    path = _write(tmp_path, lambda d: None, thresholds={"hybrid": bad})
    with pytest.raises(cfg.ConfigError):
        cfg.load_config(path, env={})


def test_unknown_threshold_mode_rejected(tmp_path):
    path = _write(tmp_path, lambda d: None, thresholds={"dense": 0.5})
    with pytest.raises(cfg.ConfigError, match="unknown retrieval mode"):
        cfg.load_config(path, env={})


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.pop("seed"), "missing configuration value: seed"),
        (lambda d: d.__setitem__("seed", "x"), "seed must be an integer"),
        (lambda d: d["refusal"].__setitem__("default_threshold", 2), "must lie in"),
        (lambda d: d["service"].__setitem__("mode", "dense"), "service.mode"),
        (lambda d: d["retrieval"].__setitem__("k", 50), "retrieval.k"),
        (lambda d: d["model_server"].pop("api_key_env"), "model_server.api_key_env"),
    ],
)
def test_invalid_config_rejected(tmp_path, mutate, message):
    path = _write(tmp_path, mutate)
    with pytest.raises(cfg.ConfigError, match=message):
        cfg.load_config(path, env={})


def test_non_mapping_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("- 1\n")
    with pytest.raises(cfg.ConfigError):
        cfg.load_config(p, env={})
