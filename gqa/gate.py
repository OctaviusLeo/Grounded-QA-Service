"""CI regression gate (Req 10.2, 10.3, 10.5, 10.6).

The CI job evaluates the documented CI subset and compares its Faithfulness and citation
precision with the Baseline_File. The gate fails a metric when ``baseline - subset > threshold``
(an absolute drop in the metric's own units), and fails outright if the Baseline_File is missing
or unreadable or a compared metric was not computed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

GATED = ("faithfulness", "citation_precision")


class BaselineError(RuntimeError):
    """The Baseline_File is missing or cannot be read."""


@dataclass(frozen=True)
class BaselineFile:
    faithfulness: float
    citation_precision: float
    command: str
    seed: int
    commit: str


def load_baseline(path: str | Path) -> BaselineFile:
    p = Path(path)
    if not p.exists():
        raise BaselineError(f"Baseline_File missing: {p.as_posix()}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        stamp = data["stamp"]
        baseline = BaselineFile(float(data["faithfulness"]), float(data["citation_precision"]),
                                str(stamp["command"]), int(stamp["seed"]), str(stamp["commit"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BaselineError(f"Baseline_File unreadable: {p.as_posix()}: {exc}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", baseline.commit):
        raise BaselineError(f"Baseline_File unreadable: {p.as_posix()}: commit is not a 40-character hash")
    return baseline


def load_thresholds(path: str | Path = "config/ci.yaml") -> dict[str, float]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    th = data["regression_threshold"]
    out = {m: float(th[m]) for m in GATED}
    for m, v in out.items():
        if v < 0:
            raise ValueError(f"regression_threshold.{m} must be >= 0")
    return out


def subset_values(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, float | str]:
    """Point values from a harness output, or the not-computed cause."""
    out: dict[str, float | str] = {}
    for m in GATED:
        v = metrics.get(m, {})
        out[m] = float(v["point"]) if "point" in v else f"not computed: {v.get('not_computed', 'missing')}"
    return out


def check_regression(subset: Mapping[str, float | str], baseline: BaselineFile,
                     threshold: float | Mapping[str, float]) -> tuple[bool, str]:
    """``(failed, report)``. Each gated metric fails iff baseline - subset > its threshold."""
    th = {m: float(threshold) for m in GATED} if isinstance(threshold, (int, float)) else dict(threshold)
    failed, lines = False, []
    for m in GATED:
        base, value = getattr(baseline, m), subset.get(m)
        if not isinstance(value, float):
            failed = True
            lines.append(f"FAIL {m}: subset value {value or 'missing'}")
            continue
        drop = base - value
        bad = drop > th[m]
        failed = failed or bad
        lines.append(f"{'FAIL' if bad else 'PASS'} {m}: subset {value:.3f}, baseline {base:.3f}, "
                     f"drop {drop:+.3f}, allowed {th[m]:.3f}")
    lines.append(f"baseline from `{baseline.command}` (seed {baseline.seed}, commit {baseline.commit})")
    return failed, "\n".join(lines)


def run_gate(subset_result: str | Path, baseline_path: str | Path, ci_config: str | Path = "config/ci.yaml") -> tuple[bool, str]:
    try:
        baseline = load_baseline(baseline_path)
    except BaselineError as exc:
        return True, f"FAIL {exc}"
    body = json.loads(Path(subset_result).read_text(encoding="utf-8"))
    if body.get("failed"):
        return True, f"FAIL the evaluation subset run failed: {body['failed']}"
    return check_regression(subset_values(body["metrics"]), baseline, load_thresholds(ci_config))
