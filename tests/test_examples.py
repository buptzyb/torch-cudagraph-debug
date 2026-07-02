from __future__ import annotations

import ast
import importlib
from pathlib import Path
import re

import torch_cudagraph_debug
import torch_cudagraph_debug.memory_debug as memory_debug
import torch_cudagraph_debug.tensor_debug as tensor_debug
from torch_cudagraph_debug.memory_debug import advanced

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


def test_example_package_imports_resolve() -> None:
    for path in sorted(EXAMPLES.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if not node.module.startswith("torch_cudagraph_debug"):
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                if alias.name == "*":
                    continue
                assert hasattr(module, alias.name), (
                    f"{path}: {node.module}.{alias.name} is not exported"
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


def test_root_readme_delegates_domain_details() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guides = {
        "tensor_debug.md": "# Tensor Debug Guide",
        "memory_debug.md": "# Memory Debug Guide",
    }

    for filename, title in guides.items():
        guide = ROOT / "docs" / filename
        assert guide.is_file(), filename
        assert guide.read_text(encoding="utf-8").startswith(title)
        assert f"](docs/{filename})" in readme

    assert "### Gradient Probes" not in readme
    assert "### Allocator History Is Application-Owned" not in readme


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

    csrc_rules = [
        line.split()
        for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        if line.startswith("recursive-include src/torch_cudagraph_debug/csrc ")
    ]
    assert len(csrc_rules) == 1
    assert set(csrc_rules[0][2:]) >= {"*.cpp", "*.cu", "*.h"}


def test_docs_do_not_reference_removed_example_paths() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "docs" / "api.md",
        ROOT / "docs" / "tensor_debug.md",
        ROOT / "docs" / "memory_debug.md",
        ROOT / "docs" / "release_checklist.md",
        EXAMPLES / "README.md",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in documents)
    for old_path in OLD_EXAMPLE_PATHS:
        assert old_path not in combined


def _public_markdown_files() -> tuple[Path, ...]:
    paths = {
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "SECURITY.md",
        *ROOT.joinpath("docs").glob("*.md"),
        *EXAMPLES.rglob("README.md"),
    }
    return tuple(sorted(paths))


def _heading_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match is None:
            continue
        value = match.group(1).strip().lower()
        value = re.sub(r"[^\w\- ]", "", value)
        value = re.sub(r"\s+", "-", value)
        anchors.add(value)
    return anchors


def test_public_markdown_links_resolve() -> None:
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
    for document in _public_markdown_files():
        text = document.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            file_part, separator, anchor = target.partition("#")
            target_path = document if not file_part else document.parent / file_part
            assert target_path.is_file(), f"{document}: missing link target {target}"
            if separator and anchor:
                target_text = target_path.read_text(encoding="utf-8")
                assert anchor in _heading_anchors(target_text), (
                    f"{document}: missing anchor {target}"
                )


def test_api_reference_covers_every_supported_export() -> None:
    reference = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    modules = (torch_cudagraph_debug, tensor_debug, memory_debug, advanced)
    for module in modules:
        for name in module.__all__:
            assert name in reference, (
                f"docs/api.md does not cover {module.__name__}.{name}"
            )


def test_public_markdown_has_balanced_fences() -> None:
    for document in _public_markdown_files():
        lines = document.read_text(encoding="utf-8").splitlines()
        assert sum(line.startswith("```") for line in lines) % 2 == 0, document


def test_docs_do_not_split_hyphenated_words_across_lines() -> None:
    pattern = re.compile(r"[A-Za-z0-9]-\n[ \t]*[a-z]")
    for document in sorted((ROOT / "docs").glob("*.md")):
        text = document.read_text(encoding="utf-8")
        assert pattern.search(text) is None, document


def test_release_checklist_covers_every_runnable_example() -> None:
    checklist = (ROOT / "docs" / "release_checklist.md").read_text(encoding="utf-8")
    scripts = sorted(
        path.relative_to(ROOT).as_posix()
        for path in EXAMPLES.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    )
    for script in scripts:
        assert script in checklist, script
    assert "TCGD_REPO_ROOT" in checklist
    assert "TCGD_TEST_INSTALLED=1" in checklist
    assert "architecture spec" not in checklist.lower()
    assert "\npip install" not in checklist


def test_public_docs_use_supported_api_terminology() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in _public_markdown_files()
    )
    assert "stable API" not in combined
    assert "stable facade" not in combined
