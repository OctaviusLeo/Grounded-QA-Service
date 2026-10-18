"""Milestone gate checks (replace the owner review stops). Usage: python scripts/gates.py m0|m1|...

Each check prints PASS or FAIL with the reason; the exit status is non-zero if any check fails.
The unit suite itself is run by scripts/gate_m<N>.sh before this script.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from gqa.docblock import read_block

FAILS: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + what + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(what)


def commit_exists(commit: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode == 0


def stamped(path: str) -> dict:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    st = body.get("stamp", {})
    check(bool(re.fullmatch(r"[0-9a-f]{40}", st.get("commit", ""))) and commit_exists(st["commit"]),
          f"{path} stamped with an existing 40-char commit", st.get("commit", "missing"))
    check(isinstance(st.get("seed"), int) and bool(st.get("command")), f"{path} records seed and command")
    return body


def gate_m0() -> None:
    env = Path("docs/ENVIRONMENT.md").read_text(encoding="utf-8")
    for model in ("granite3.3:8b", "granite3.3:2b"):
        block = read_block(env, f"smoke:{model}") or ""
        check("Status: passed" in block, f"smoke test passed for {model}")
        m = re.search(r"peak during generation: \*\*(\d+) MiB\*\* of (\d+) MiB", block)
        check(bool(m) and int(m.group(1)) <= int(m.group(2)), f"{model} peak GPU memory <= total", m.group(0) if m else "not recorded")
        check(bool(re.search(r"commit `[0-9a-f]{40}`", block)), f"{model} evidence stamped")
    for name in ("model-server", "tools", "python-packages", "machine", "hosted-model"):
        check(read_block(env, name) is not None, f"ENVIRONMENT.md block {name}")
    tp = Path("THIRD_PARTY.md").read_text(encoding="utf-8")
    for needle in ("Ollama", "granite3.3:2b", "granite3.3:8b", "Apache-2.0"):
        check(needle in tp, f"THIRD_PARTY.md lists {needle}")


GATES = {"m0": gate_m0}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        GATES[name]()
    print(f"gate {' '.join(sys.argv[1:])}: {'FAILED' if FAILS else 'passed'}")
    sys.exit(1 if FAILS else 0)
