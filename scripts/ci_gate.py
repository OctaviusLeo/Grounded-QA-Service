"""CI regression gate: python scripts/ci_gate.py --subset-result <eval.json> [--baseline results/baseline.json]

Exits 1 if faithfulness or citation precision dropped more than config/ci.yaml allows, if the
Baseline_File is missing or unreadable, or if the subset evaluation failed.
"""

import argparse
import sys

from gqa.gate import run_gate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-result", required=True)
    parser.add_argument("--baseline", default="results/baseline.json")
    parser.add_argument("--ci-config", default="config/ci.yaml")
    args = parser.parse_args()
    failed, report = run_gate(args.subset_result, args.baseline, args.ci_config)
    print(report)
    print("regression gate: " + ("FAILED" if failed else "passed"))
    sys.exit(1 if failed else 0)
