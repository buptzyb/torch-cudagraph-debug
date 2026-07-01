from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
OLD_EXAMPLE_PATHS = (
    "examples/grad_probe_patterns.py",
    "examples/memory_debug_basic.py",
    "examples/memory_debug_lifetimes.py",
    "examples/memory_debug_timeline.py",
    "examples/multiple_invocations_record_compare.py",
    "examples/tensor_debug_basic.py",
    "examples/tensor_debug_record_compare.py",
    "examples/tensorboard_export_records.py",
    "examples/transformer_block_probe.py",
)


def test_example_index_covers_every_script_once() -> None:
    index = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    scripts = sorted(
        path.relative_to(EXAMPLES).as_posix()
        for path in EXAMPLES.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    )

    assert scripts
    for script in scripts:
        assert index.count(f"]({script})") == 1, script


def test_examples_are_grouped_by_domain() -> None:
    assert not list(EXAMPLES.glob("*.py"))
    assert not list(EXAMPLES.glob("*.sh"))

    index = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    categories = ("tensor_debug", "memory_debug", "integrations")
    for category in categories:
        category_readme = EXAMPLES / category / "README.md"
        assert category_readme.is_file(), category
        assert f"]({category}/README.md)" in index, category


def test_cli_workflow_uses_configured_python_for_torchrun() -> None:
    script = (EXAMPLES / "memory_debug" / "cli_workflows.sh").read_text(
        encoding="utf-8"
    )
    assert '"${PYTHON_BIN}" -m torch.distributed.run' in script
    assert "GROUP_ROOT=" in script
    assert "GROUPS=" not in script


def test_source_distribution_includes_all_example_assets() -> None:
    rules = [
        line.split()
        for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        if line.startswith("recursive-include examples ")
    ]
    assert len(rules) == 1
    assert set(rules[0][2:]) >= {"*.md", "*.py", "*.sh"}


def test_docs_do_not_reference_removed_example_paths() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "docs" / "api.md",
        ROOT / "docs" / "release_checklist.md",
        EXAMPLES / "README.md",
    )
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in documents
    )
    for old_path in OLD_EXAMPLE_PATHS:
        assert old_path not in combined
