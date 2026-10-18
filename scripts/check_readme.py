"""README number traceability check (Req 11.4, 11.5): python scripts/check_readme.py

Lists every number in README.md that cannot be traced to results/summary.md or another
script-generated file, and exits 1 if there is any.
"""

import sys

from gqa.readme_check import run

if __name__ == "__main__":
    findings = run()
    for f in findings:
        print(f"untraceable: {f.number} (README.md line {f.line}): {f.context}")
    print(f"README number check: {'FAILED' if findings else 'passed'} ({len(findings)} untraceable)")
    sys.exit(1 if findings else 0)
