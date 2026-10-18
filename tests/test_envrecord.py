from email.message import Message

import pytest

from gqa import envrecord as er


class FakeDist:
    def __init__(self, name, version, **fields):
        self.version = version
        self.metadata = Message()
        self.metadata["Name"] = name
        for key, value in fields.items():
            if isinstance(value, list):
                for v in value:
                    self.metadata[key.replace("_", "-")] = v
            else:
                self.metadata[key.replace("_", "-")] = value


def test_tool_versions_parse_and_record_artifacts():
    outputs = {
        "python": "Python 3.12.15", "uv": "uv 0.13.1 (x)", "docker": "29.9.1", "nvidia-ctk": "NVIDIA Container Toolkit CLI version 1.20.1\ncommit: x",
        "nvidia-smi": "617.42", "uname": "6.18", "grep": 'PRETTY_NAME="Ubuntu 24.04.5 LTS"', "git.exe": "git version 2.55.0.windows.5", "gh.exe": "gh version 2.102.0 (2026)",
    }

    def runner(args):
        if args[:2] == ["docker", "compose"]:
            return "5.6.0"
        return outputs[args[0]]

    tools = {t.name: t for t in er.tool_versions(runner)}
    assert tools["Python (project environment)"].version == "3.12.15"
    assert tools["NVIDIA Container Toolkit"].version == "1.20.1"
    assert tools["Ubuntu (WSL distro)"].version == "Ubuntu 24.04.5 LTS"
    assert tools["Docker Compose plugin"].version == "5.6.0"
    assert tools["uv"].artifact == "`uv --version`"
    table = er.render_tools(list(tools.values()))
    assert table.count("\n") == len(tools) + 1


def test_tool_versions_marks_unavailable():
    def runner(args):
        raise RuntimeError("missing")

    assert all(t.version.startswith("unavailable") for t in er.tool_versions(runner))


def test_license_sources_in_priority_order():
    assert er.license_of(FakeDist("a", "1", License_Expression="MIT")) == "MIT"
    assert er.license_of(FakeDist("b", "1", Classifier=["License :: OSI Approved :: BSD License", "Programming Language :: Python"])) == "BSD License"
    assert er.license_of(FakeDist("c", "1", License="Apache 2.0")) == "Apache 2.0"
    long = er.license_of(FakeDist("d", "1", License="Copyright (c) 2020 Someone with a very long first line of license text here\nmore"))
    assert long.endswith("…") and len(long) <= 61
    assert er.license_of(FakeDist("e", "1")) == "UNKNOWN"


def test_installed_packages_sorted_and_rendered():
    pkgs = er.installed_packages([FakeDist("Zeta", "2", License="MIT"), FakeDist("alpha", "1", License="BSD"), FakeDist("", "0")])
    assert [p.name for p in pkgs] == ["alpha", "Zeta"]
    direct = er.direct_dependencies('dependencies = [\n    "alpha==1",\n    "other>=2",\n]')
    assert direct == {"alpha"}
    text = er.render_packages(pkgs, direct, "`uv pip list`")
    assert "| **alpha** | 1 | BSD |" in text and "| Zeta | 2 | MIT |" in text


def test_model_helpers():
    assert er.model_license_id("Apache License\nVersion 2.0, January 2004") == "Apache-2.0"
    assert er.model_license_id("custom") == "see license text"
    assert er.ollama_models(lambda path: {"models": [{"name": "m"}]}) == [{"name": "m"}]
    assert er.render_json_block({"b": 1, "a": 2}).startswith("```json\n{\n  \"a\": 2")


def test_run_raises_on_failure():
    with pytest.raises(RuntimeError):
        er.run(["false"])
    assert er.run(["echo", "hi"]) == "hi"
