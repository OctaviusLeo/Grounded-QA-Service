"""Regenerate results/summary.md from the stamped result files: python scripts/write_results_table.py"""

import sys

from gqa.config import load_config
from gqa.repro import make_stamp
from gqa.results_table import write_results_table

if __name__ == "__main__":
    write_results_table(make_stamp(seed=load_config().seed))
    print("wrote results/summary.md")
    sys.exit(0)
