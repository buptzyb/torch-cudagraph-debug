from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".agents" / "skills" / "tcgd-investigate"
SKILL = SKILL_ROOT / "SKILL.md"
CLAUDE_SKILL = ROOT / ".claude" / "skills" / "tcgd-investigate"
CODEX_AGENT = ROOT / ".codex" / "agents" / "tcgd-debugger.toml"
CLAUDE_AGENT = ROOT / ".claude" / "agents" / "tcgd-debugger.md"
PLUGIN_ROOT = ROOT / "plugins" / "tcgd"


def _frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---(?:\n|\Z)", text, flags=re.DOTALL)
    assert match is not None, f"{path}: missing YAML frontmatter"

    values: dict[str, object] = {}
    current_list: str | None = None
    for line in match.group(1).splitlines():
        if line.startswith("  - "):
            assert current_list is not None, f"{path}: list item without key"
            value = values.setdefault(current_list, [])
            assert isinstance(value, list)
            value.append(line[4:])
            continue

        key, separator, value = line.partition(":")
        assert separator, f"{path}: unsupported frontmatter line: {line}"
        key = key.strip()
        value = value.strip()
        if value:
            values[key] = value
            current_list = None
        else:
            values[key] = []
            current_list = key
    return values


def test_shared_repository_instructions_are_discoverable() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert claude.strip() == "@AGENTS.md"
    assert "Tool-First Investigation Contract" in agents
    assert "tcgd-investigate" in agents
    assert "tcgd-debugger" in agents


def test_canonical_skill_has_required_metadata_and_support_files() -> None:
    metadata = _frontmatter(SKILL)

    assert metadata["name"] == "tcgd-investigate"
    description = metadata["description"]
    assert isinstance(description, str)
    assert "PyTorch CUDA Graph" in description
    assert "Do not use for implementing" in description
    assert (SKILL_ROOT / "references" / "workflow-map.md").is_file()
    assert (SKILL_ROOT / "assets" / "report-template.md").is_file()


def test_claude_skill_is_a_single_source_link() -> None:
    if CLAUDE_SKILL.is_symlink():
        assert CLAUDE_SKILL.resolve() == SKILL_ROOT.resolve()
        assert (CLAUDE_SKILL / "SKILL.md").samefile(SKILL)
        return

    assert not (ROOT / ".git").exists(), "source checkout must use a skill link"
    for relative_path in (
        Path("SKILL.md"),
        Path("assets/report-template.md"),
        Path("references/workflow-map.md"),
    ):
        assert (CLAUDE_SKILL / relative_path).read_bytes() == (
            SKILL_ROOT / relative_path
        ).read_bytes()


def test_source_distribution_manifest_includes_agent_assets() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "include AGENTS.md" in manifest
    assert "include CLAUDE.md" in manifest
    assert "recursive-include .agents *.md" in manifest
    assert "recursive-include .claude *.md" in manifest
    assert "recursive-include .codex *.toml" in manifest


def test_codex_agent_uses_the_shared_skill_contract() -> None:
    # tomllib is stdlib only on Python >= 3.11; the package floor is 3.10.
    # Only this test needs it — the other asset gates must keep running.
    tomllib = pytest.importorskip("tomllib")
    with CODEX_AGENT.open("rb") as file:
        agent = tomllib.load(file)

    assert agent["name"] == "tcgd-debugger"
    assert "torch-cudagraph-debug" in agent["description"]
    instructions = agent["developer_instructions"]
    assert ".agents/skills/tcgd-investigate/SKILL.md" in instructions
    assert "Do not\nmodify the torch-cudagraph-debug library" in instructions
    assert "model" not in agent
    assert "sandbox_mode" not in agent


def test_claude_agent_preloads_the_canonical_skill() -> None:
    agent = _frontmatter(CLAUDE_AGENT)

    assert agent["name"] == "tcgd-debugger"
    assert agent["model"] == "inherit"
    assert agent["skills"] == ["tcgd-investigate"]
    assert "torch-cudagraph-debug" in str(agent["description"])


def test_skill_routes_to_existing_public_examples() -> None:
    workflow_map = (SKILL_ROOT / "references" / "workflow-map.md").read_text(
        encoding="utf-8"
    )
    example_paths = set(re.findall(r"`(examples/[^`]+\.(?:py|sh))`", workflow_map))

    assert example_paths
    for relative_path in sorted(example_paths):
        assert (ROOT / relative_path).is_file(), relative_path


def test_agent_workflow_documentation_is_linked() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "agent_workflows.md").read_text(encoding="utf-8")

    assert "](docs/agent_workflows.md)" in readme
    assert "$tcgd-investigate" in guide
    assert "/tcgd-investigate" in guide
    assert "tcgd-debugger" in guide


def test_plugin_wraps_canonical_assets_through_marketplace_links() -> None:
    marketplace = json.loads(
        (ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    assert marketplace["name"] == "torch-cudagraph-debug"
    entries = {item["name"]: item for item in marketplace["plugins"]}
    assert entries["tcgd"]["source"] == "./plugins/tcgd"

    manifest = json.loads(
        (PLUGIN_ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    assert manifest["name"] == "tcgd"
    assert manifest["license"] == "Apache-2.0"

    skill_link = PLUGIN_ROOT / "skills" / "tcgd-investigate"
    agent_link = PLUGIN_ROOT / "agents" / "tcgd-debugger.md"
    assert skill_link.is_symlink(), "plugin skill must link to the canonical skill"
    assert skill_link.resolve() == SKILL_ROOT.resolve()
    assert agent_link.is_symlink(), "plugin agent must link to the Claude agent"
    assert agent_link.resolve() == CLAUDE_AGENT.resolve()
