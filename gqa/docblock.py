"""Script-written blocks inside Markdown files.

A block is delimited by ``<!-- BEGIN name -->`` and ``<!-- END name -->``. Scripts replace the
block's contents; everything outside the markers is prose that people edit.
"""

from __future__ import annotations

import re
from pathlib import Path


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"


def replace_block(text: str, name: str, content: str, after_heading: str | None = None) -> str:
    """Return ``text`` with block ``name`` set to ``content``.

    A missing block is inserted at the end of the section ``after_heading`` (a full heading line,
    for example ``"## Tools"``) or, failing that, at the end of the text.
    """
    begin, end = _markers(name)
    block = f"{begin}\n{content.rstrip()}\n{end}"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        return pattern.sub(lambda _: block, text, count=1)
    if after_heading and after_heading in text:
        start = text.index(after_heading) + len(after_heading)
        # The section ends at the next heading of the same or a higher level that is not inside
        # another script-written block (blocks may contain their own deeper headings).
        level = len(after_heading) - len(after_heading.lstrip("#"))
        rest = re.sub(r"<!-- BEGIN (.+?) -->.*?<!-- END \1 -->", lambda m: " " * len(m.group(0)), text[start:], flags=re.S)
        nxt = re.search(rf"^#{{1,{level}}} ", rest, re.M)
        cut = start + nxt.start() if nxt else len(text)
        head, tail = text[:cut].rstrip("\n"), text[cut:]
        return f"{head}\n\n{block}\n\n{tail}" if tail else f"{head}\n\n{block}\n"
    return f"{text.rstrip()}\n\n{block}\n"


def read_block(text: str, name: str) -> str | None:
    begin, end = _markers(name)
    m = re.search(re.escape(begin) + r"\n?(.*?)\n?" + re.escape(end), text, re.S)
    return m.group(1) if m else None


def update_file(path: str | Path, name: str, content: str, after_heading: str | None = None) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    p.write_text(replace_block(text, name, content, after_heading), encoding="utf-8", newline="\n")
