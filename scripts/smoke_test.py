"""M0 smoke test: python scripts/smoke_test.py --model granite3.3:8b --model granite3.3:2b

Writes one evidence block per model into docs/ENVIRONMENT.md (Req 1.1-1.8) and exits non-zero
if any model fails, after recording the failed condition and its cause.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from gqa.config import load_config
from gqa.docblock import update_file
from gqa.repro import make_stamp
from gqa.smoke import default_deps, render_block, run_smoke

DOC = "docs/ENVIRONMENT.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="model to test (repeatable)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--no-start", action="store_true", help="do not run docker compose up")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    stamp = make_stamp(seed=cfg.seed)
    deps = default_deps(cfg.model_server.base_url, cfg.model_server.api_key_env, cfg.seed)
    failures = 0
    for model in args.model:
        result = run_smoke(model, deps, start_server=not args.no_start)
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        update_file(DOC, f"smoke:{model}", render_block(result, stamp, when), after_heading="## Smoke test evidence")
        if result.ok:
            print(f"PASS {model}: peak {result.peak_used_mib}/{result.total_mib} MiB, {result.elapsed_s:.1f} s, answer: {result.answer[:120]!r}")
        else:
            failures += 1
            print(f"FAIL {model}: {result.failed_condition}: {result.cause}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
