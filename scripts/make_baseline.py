"""Create the Baseline_File: python scripts/make_baseline.py

Evaluates the CI subset with the configuration in config/ci.yaml (the same command CI runs) and
writes results/baseline.json with faithfulness, citation precision, and the run's stamp.
"""

from __future__ import annotations

import json
import sys

import yaml

from gqa import eval as harness
from gqa.repro import read_stamped_json, write_json
from gqa.gate import subset_values

SOURCE = "results/eval/baseline_source.json"


def main() -> int:
    ci = yaml.safe_load(open("config/ci.yaml", encoding="utf-8"))
    argv = ["--config", "config/default.yaml", "--mode", ci["mode"], "--model", ci["model"],
            "--judge-model", ci["judge_model"], "--subset", "ci", "--out", SOURCE]
    if harness.main(argv) != 0:
        print("baseline evaluation failed", file=sys.stderr)
        return 1
    body, stamp = read_stamped_json(SOURCE)
    values = subset_values(body["metrics"])
    if not all(isinstance(v, float) for v in values.values()):
        print(f"cannot make a baseline: {values}", file=sys.stderr)
        return 1
    write_json("results/baseline.json", {**values, "source": SOURCE, "subset": ci["subset"], "mode": ci["mode"],
                                         "model": ci["model"], "judge_model": ci["judge_model"],
                                         "generating_command": "python -m gqa.eval " + " ".join(argv)}, stamp)
    print(json.dumps(values))
    return 0


if __name__ == "__main__":
    sys.exit(main())
