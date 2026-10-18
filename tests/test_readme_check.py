from hypothesis import given, settings
from hypothesis import strategies as st

from gqa import readme_check as rc

SUMMARY = """# Results

## Configuration experiments (Results_Table)

| Configuration | F1 |
| --- | --- |
| hybrid_rerank_8b | 0.612 [0.551, 0.670] |

### Retrieval

| Configuration | Recall@5 |
| --- | --- |
| hybrid_rerank_8b | 0.957 [0.920, 0.987] |

## Injection test

Followed fraction 0.167 over 22 cases.
"""


def _readme(*body):
    block = rc.summary_sections(SUMMARY)["Retrieval"]
    return "\n".join(["# Title", "", "<!-- BEGIN summary:Retrieval -->", block, "<!-- END summary:Retrieval -->", "", *body])


def test_sections_and_blocks():
    s = rc.summary_sections(SUMMARY)
    assert set(s) == {"Configuration experiments (Results_Table)", "Retrieval", "Injection test"}
    assert s["Configuration experiments (Results_Table)"].count("###") == 1  # includes its subsection
    stale = _readme().replace("0.957", "0.999")
    findings = rc.check(stale, SUMMARY, set())
    assert [f.number for f in findings] == ["[block summary:Retrieval]"]
    refreshed = rc.render_blocks(stale, SUMMARY)
    assert rc.check(refreshed, SUMMARY, set()) == []


def test_skipped_contexts_and_traced_numbers():
    sources = rc.numbers_in(SUMMARY)
    text = _readme(
        "Run `docker compose up` on port `8000` with Python `3.12.15`.",
        "```bash\npython scripts/run.py --seed 1234\n```",
        "See [the docs](https://example.com/v2/page?id=77) and Req 7.5, Requirements 4.4 and 4.5, task 6.13, M0 to M5.",
        "Checked 2026-10-17 with Ollama 0.40.6 on granite3.3:8b.",
        "The judge flagged 22 cases; the injected passages were followed 0.167 of the time; F1 reached 0.612.",
        "<!-- 99 hidden -->",
    )
    assert rc.check(text, SUMMARY, sources) == []


def test_untraceable_numbers_fail():
    text = _readme("Our F1 is 0.73 and latency 1,234 ms, a 15% gain.")
    findings = rc.check(text, SUMMARY, rc.numbers_in(SUMMARY))
    assert [f.number for f in findings] == ["0.73", "1,234", "15%"]
    assert findings[0].line == text.splitlines().index("Our F1 is 0.73 and latency 1,234 ms, a 15% gain.") + 1
    assert "0.73" in findings[0].context


def test_source_numbers_reads_only_generated_environment_blocks(tmp_path):
    env = tmp_path / "ENVIRONMENT.md"
    env.write_text("prose 111\n<!-- BEGIN tools -->\n| uv | 222 |\n<!-- END tools -->\n")
    other = tmp_path / "x.json"
    other.write_text('{"a": 0.333, "b": "1,000"}')
    found = rc.source_numbers([env, other, tmp_path / "missing.json"])
    assert {"222", "0.333", "1000"} <= found and "111" not in found


def test_run_on_files(tmp_path):
    (tmp_path / "summary.md").write_text(SUMMARY)
    (tmp_path / "README.md").write_text(_readme("We report 0.612 and 0.42."))
    findings = rc.run(tmp_path / "README.md", tmp_path / "summary.md", [tmp_path / "summary.md"])
    assert [f.number for f in findings] == ["0.42"]


def test_render_blocks_unknown_section():
    import pytest

    with pytest.raises(KeyError):
        rc.render_blocks("<!-- BEGIN summary:Nope -->\nx\n<!-- END summary:Nope -->", SUMMARY)


traced = st.sampled_from(["0.612", "0.957", "22", "0.167", "0.551"])
untraced = st.sampled_from(["0.731", "41", "3.5", "12,345", "88%", "0.009"])
words = st.sampled_from(["The", "result", "is", "about", "and", "with", "F1"])


@settings(max_examples=100, derandomize=True)
@given(st.lists(st.tuples(st.booleans(), traced, untraced, words), min_size=1, max_size=8))
def test_property_16_readme_number_traceability(parts):
    """Feature: grounded-qa-service, Property 16: README number traceability."""
    sources = rc.numbers_in(SUMMARY)
    tokens, expected = [], []
    for use_untraced, good, bad, word in parts:
        if use_untraced:
            tokens += [word, bad]
            expected.append(bad)
        else:
            tokens += [word, good]
    text = _readme(" ".join(tokens) + ".")
    findings = rc.check(text, SUMMARY, sources)
    assert [f.number for f in findings] == expected
    assert bool(findings) == bool(expected)  # the check fails exactly when an untraceable number exists
