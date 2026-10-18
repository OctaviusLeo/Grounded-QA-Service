"""README number traceability (Req 11.4, 11.5).

Every number in the README must come from a file a script generated. Two ways qualify:

1. It sits inside a generated block, ``<!-- BEGIN summary:<section> -->`` ... ``<!-- END ... -->``,
   whose text is a verbatim copy of that section of ``results/summary.md`` (checked here; a block
   that differs from its source is itself reported).
2. It appears, as a whole number, in one of the generated source files (the Results_Table, the
   stamped result JSON files, the tuned thresholds, the corpus and question-set manifests, and
   the script-written blocks of ``docs/ENVIRONMENT.md``).

Numbers that are not measurements are skipped: anything in code (inline or fenced), link
targets, HTML comments, dates, version strings, and labels such as ``Req 7.5`` or ``task 6.13``.
Any other number is reported as untraceable and the check fails.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from gqa.docblock import read_block

# A number not glued to a word, path, or label (Recall@5, P95, v2.0, top-5 are labels, not measurements).
NUMBER = re.compile(r"(?<![\w.\-/@])[-+]?\d[\d,]*(?:\.\d+)?%?(?![\w\-/]|\.\d)")
_BLOCK = re.compile(r"<!-- BEGIN (summary:[^>]+?) -->\n?(.*?)\n?<!-- END \1 -->", re.S)
_SKIP = [
    re.compile(r"```.*?```", re.S),                     # fenced code
    re.compile(r"`[^`\n]*`"),                           # inline code
    re.compile(r"<!--.*?-->", re.S),                    # comments
    re.compile(r"\]\([^)]*\)"),                         # link targets
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),               # dates
    re.compile(r"\bv?\d+\.\d+\.\d+\b"),                 # versions
    re.compile(r"\b(?:Req(?:uirements?)?|Propert(?:y|ies)|tasks?|M)\s?\d+(?:\.\d+)?(?:\s*(?:,|and|-|–)\s*\d+(?:\.\d+)?)*", re.I),
]


@dataclass(frozen=True)
class Finding:
    number: str
    line: int
    context: str


def normalize(number: str) -> str:
    return number.replace(",", "").rstrip("%").lstrip("+")


def numbers_in(text: str) -> set[str]:
    return {normalize(m.group(0)) for m in NUMBER.finditer(text)}


def summary_sections(summary: str) -> dict[str, str]:
    """``{heading text: section text}`` for every ``##``/``###`` heading of the Results_Table."""
    out = {}
    matches = list(re.finditer(r"^(#{2,3}) (.+)$", summary, re.M))
    for i, m in enumerate(matches):
        level = len(m.group(1))
        end = len(summary)
        for nxt in matches[i + 1:]:
            if len(nxt.group(1)) <= level:
                end = nxt.start()
                break
        out[m.group(2).strip()] = summary[m.start():end].strip()
    return out


def _mask(text: str, pattern: re.Pattern) -> str:
    return pattern.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def source_numbers(paths: Iterable[Path]) -> set[str]:
    found: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        if p.name == "ENVIRONMENT.md":  # only the script-written blocks count
            text = "\n".join(m.group(2) for m in re.finditer(r"<!-- BEGIN ([^>]+?) -->(.*?)<!-- END \1 -->", text, re.S))
        found |= numbers_in(text)
    return found


def check(readme: str, summary: str, sources: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    sections = summary_sections(summary)
    # 1. Generated blocks must match their source section exactly; then their numbers are traced.
    masked = readme
    for m in _BLOCK.finditer(readme):
        name, body = m.group(1), m.group(2).strip()
        section = name.split(":", 1)[1]
        line = readme[: m.start()].count("\n") + 1
        if sections.get(section) != body:
            findings.append(Finding(f"[block {name}]", line, "generated block differs from results/summary.md"))
        masked = masked.replace(m.group(0), re.sub(r"[^\n]", " ", m.group(0)), 1)
    # 2. Every other number must appear in a generated source file.
    for pattern in _SKIP:
        masked = _mask(masked, pattern)
    for lineno, line in enumerate(masked.splitlines(), start=1):
        for m in NUMBER.finditer(line):
            if normalize(m.group(0)) not in sources:
                findings.append(Finding(m.group(0), lineno, readme.splitlines()[lineno - 1].strip()[:120]))
    return findings


DEFAULT_SOURCES = [
    "results/summary.md", "results/experiments.json", "results/retrieval_baseline.json", "results/injection.json",
    "results/baseline.json", "results/judge/agreement.json", "results/eval/determinism.json", "config/thresholds.yaml",
    "data/corpus/manifest.json", "data/evalset_manifest.json", "docs/ENVIRONMENT.md",
]


def run(readme_path: str | Path = "README.md", summary_path: str | Path = "results/summary.md",
        sources: Iterable[str | Path] = DEFAULT_SOURCES) -> list[Finding]:
    summary = Path(summary_path).read_text(encoding="utf-8")
    return check(Path(readme_path).read_text(encoding="utf-8"), summary, source_numbers(Path(s) for s in sources))


def render_blocks(readme: str, summary: str) -> str:
    """Refresh every ``summary:<section>`` block of the README from the Results_Table."""
    from gqa.docblock import replace_block

    sections = summary_sections(summary)
    for m in list(_BLOCK.finditer(readme)):
        name = m.group(1)
        section = name.split(":", 1)[1]
        if section not in sections:
            raise KeyError(f"README block {name} names a section that results/summary.md does not have")
        readme = replace_block(readme, name, sections[section])
    assert all(read_block(readme, m.group(1)) is not None for m in _BLOCK.finditer(readme))
    return readme
