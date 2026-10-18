"""Fail if any non-trivial gqa/ module is below 80% statement coverage (Req 0.11).

A module is non-trivial when it has more than one public function or more than 50 source lines.
Usage: python scripts/check_coverage.py <coverage.json>
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

MIN_PERCENT = 80.0


def is_non_trivial(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    public = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")
    ]
    source_lines = [ln for ln in source.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return len(public) > 1 or len(source_lines) > 50


def main(argv: list[str]) -> int:
    report = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    files = {Path(k).as_posix(): v for k, v in report["files"].items()}
    failures = []
    for module in sorted(Path("gqa").glob("**/*.py")):
        if not is_non_trivial(module):
            continue
        entry = files.get(module.as_posix())
        percent = entry["summary"]["percent_covered"] if entry else 0.0
        status = "ok" if percent >= MIN_PERCENT else "LOW"
        print(f"coverage {status:>3} {percent:6.1f}%  {module.as_posix()}")
        if percent < MIN_PERCENT:
            failures.append(module.as_posix())
    if failures:
        print(f"FAILED: {len(failures)} non-trivial module(s) below {MIN_PERCENT:.0f}%: {', '.join(failures)}")
        return 1
    print(f"coverage gate passed (>= {MIN_PERCENT:.0f}% for every non-trivial module)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
