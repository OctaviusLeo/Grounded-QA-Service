import json
import subprocess

import pytest

from gqa import repro


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "code.py").write_text("x = 1\n")
    _git(tmp_path, "add", "code.py")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_commit_hash_is_40_hex(repo):
    value = repro.commit_hash(repo)
    assert len(value) == 40
    assert all(c in "0123456789abcdef" for c in value)


def test_commit_hash_outside_repo_raises(tmp_path):
    with pytest.raises(repro.ReproError):
        repro.commit_hash(tmp_path)


def test_clean_tree_ignores_generated_paths(repo):
    (repo / "results").mkdir()
    (repo / "results" / "summary.md").write_text("generated\n")
    (repo / "docs" / "media").mkdir(parents=True)
    (repo / "docs" / "media" / "demo.gif").write_bytes(b"GIF")
    (repo / ".kiro").mkdir()
    (repo / ".kiro" / "tasks.md").write_text("- [x]\n")
    assert repro.dirty_paths(repo) == []
    repro.require_clean_tree(repo)


def test_clean_tree_flags_source_edits_and_untracked(repo):
    (repo / "code.py").write_text("x = 2\n")
    (repo / "new_module.py").write_text("y = 1\n")
    dirty = repro.dirty_paths(repo)
    assert set(dirty) == {"code.py", "new_module.py"}
    with pytest.raises(repro.ReproError, match="uncommitted"):
        repro.require_clean_tree(repo)
    with pytest.raises(repro.ReproError):
        repro.make_stamp(seed=1, command="cmd", repo=repo)


def test_make_stamp_and_round_trip(repo, tmp_path):
    stamp = repro.make_stamp(seed=7, command="python scripts/x.py --seed 7", repo=repo)
    assert stamp.commit == repro.commit_hash(repo)
    assert repro.stamp_from_dict(repro.stamp_to_dict(stamp)) == stamp
    out = tmp_path / "out" / "result.json"
    repro.write_json(out, {"value": 0.5}, stamp)
    body, again = repro.read_stamped_json(out)
    assert body == {"value": 0.5}
    assert again == stamp
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    assert json.loads(raw)["stamp"]["seed"] == 7


def test_make_stamp_dirty_allowed_when_not_required(repo):
    (repo / "code.py").write_text("x = 3\n")
    stamp = repro.make_stamp(seed=0, command="c", repo=repo, require_clean=False)
    assert stamp.seed == 0


@pytest.mark.parametrize("commit", ["abc", "G" * 40, "A" * 40])
def test_stamp_rejects_bad_commit(commit):
    with pytest.raises(repro.ReproError):
        repro.Stamp(command="c", seed=1, commit=commit)


@pytest.mark.parametrize("seed", [1.5, "3", True])
def test_stamp_rejects_non_integer_seed(seed):
    with pytest.raises(repro.ReproError):
        repro.Stamp(command="c", seed=seed, commit="a" * 40)


def test_stamp_markdown_contains_all_fields():
    stamp = repro.Stamp(command="python scripts/x.py", seed=3, commit="b" * 40)
    line = repro.stamp_markdown(stamp)
    assert "python scripts/x.py" in line and "`3`" in line and "b" * 40 in line


def test_current_command_formats_script_path():
    assert repro.current_command(["scripts/run.py", "--seed", "1"]) == "python scripts/run.py --seed 1"
    assert repro.current_command(["tool", "a b"]) == "tool 'a b'"
