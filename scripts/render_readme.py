"""Refresh the generated blocks of README.md from results/summary.md: python scripts/render_readme.py"""

import sys
from pathlib import Path

from gqa.readme_check import render_blocks

if __name__ == "__main__":
    readme = Path("README.md")
    text = render_blocks(readme.read_text(encoding="utf-8"), Path("results/summary.md").read_text(encoding="utf-8"))
    readme.write_text(text, encoding="utf-8", newline="\n")
    print("README.md generated blocks refreshed from results/summary.md")
    sys.exit(0)
