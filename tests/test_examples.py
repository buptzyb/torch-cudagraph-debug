from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import torch_cudagraph_debug
import torch_cudagraph_debug.memory_debug as memory_debug
import torch_cudagraph_debug.tensor_debug as tensor_debug
from torch_cudagraph_debug.memory_debug import advanced
from torch_cudagraph_debug.memory_debug.cli import build_parser as memory_cli_parser
from torch_cudagraph_debug.tensor_debug.cli import build_parser as tensor_cli_parser

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
    "examples/tensor_debug/quickstart.py",
    "examples/tensor_debug/snapshot_comparison.py",
    "examples/tensor_debug/record_and_check.py",
    "examples/tensor_debug/multiple_invocations.py",
    "examples/tensor_debug/gradient_probes.py",
    "examples/tensor_debug/probe_modes.py",
    "examples/tensor_debug/module_integration.py",
    "examples/tensor_debug/eager_vs_cuda_graph.py",
    "examples/tensor_debug/replay_series.py",
    "examples/tensor_debug/cli_workflows.sh",
    "examples/memory_debug/quickstart.py",
    "examples/memory_debug/snapshot_comparison.py",
    "examples/memory_debug/timeline_and_reports.py",
    "examples/memory_debug/attribution_modes.py",
    "examples/memory_debug/allocation_lifetimes.py",
    "examples/memory_debug/compare_runs_and_phases.py",
    "examples/memory_debug/distributed_run_groups.py",
    "examples/memory_debug/cli_workflows.sh",
)


MAJOR_WORKFLOW_COVERAGE = {
    "tensor_debug/probe/quickstart.py": (
        "TensorProbe(",
        "RecordAction(",
        "snapshot(",
    ),
    "tensor_debug/probe/snapshot_comparison.py": ("compare_snapshots(",),
    "tensor_debug/probe/replay_comparison.py": ("probe.compare(",),
    "tensor_debug/probe/actions.py": (
        "PrintAction(",
        "RecordAction(",
        "CheckAction(",
        "check_status(",
    ),
    "tensor_debug/probe/gradients.py": ("watch_grad(",),
    "tensor_debug/probe/capture_modes.py": (
        'when="always"',
        'non_contiguous="copy"',
    ),
    "tensor_debug/probe/module_integration.py": ("torch.nn.Module",),
    "tensor_debug/recorder/eager_vs_cuda_graph.py": (
        "TensorRecorder(",
        "TensorRun.load(",
        "compare_points(",
        "compare_runs(",
    ),
    "tensor_debug/recorder/forward_backward.py": (
        "recorder.observe(",
        "recorder.watch_grad(",
        "recorder.preview(",
    ),
    "tensor_debug/recorder/replay_series.py": ("compare_point_series(",),
    "tensor_debug/recorder/distributed_run_groups.py": (
        "TensorRunGroup.load(",
        "compare_run_groups(",
        "torch.distributed",
    ),
    "memory_debug/probe/quickstart.py": (
        "probe.snapshot(",
        "probe.compare(",
        "snapshot_index",
    ),
    "memory_debug/probe/private_pool_inactive.py": (
        "inactive_bytes",
        "graph.replay()",
        "pool_key.pool_id",
        "pool_key.label",
    ),
    "memory_debug/probe/expandable_segments.py": (
        "expandable_inactive_bytes",
        "empty_cache()",
        "removed_segment_bytes",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    ),
    "memory_debug/probe/device_memory.py": (
        "device_memory",
        "delta_cuda_allocator_residual_bytes",
        "delta_total_bytes",
        "delta_allocator_reserved_bytes",
        "cudaMalloc(",
    ),
    "memory_debug/probe/snapshot_comparison.py": ("compare_snapshots(",),
    "memory_debug/recorder/timeline_and_reports.py": (
        'record_point("during_capture")',
        "recorder.preview(",
        "MemoryRun.load(",
        ".timeline(",
    ),
    "memory_debug/recorder/history_requirements.py": (
        "MemoryHistoryDisabledError",
        ".lifetimes(",
    ),
    "memory_debug/recorder/stack_and_event_attribution.py": (
        "stacks=True",
        "events=True",
        "lifetimes=True",
    ),
    "memory_debug/recorder/allocation_lifetimes.py": (".lifetimes(",),
    "memory_debug/recorder/compare_runs_and_phases.py": (
        "compare_phases(",
        "MemoryPoolKey",
        "pool_key.device_index",
        "pool_key.pool_id",
        "row.components.identity_holds",
    ),
    "memory_debug/recorder/distributed_run_groups.py": (
        "MemoryRunGroup.load(",
        "compare_run_group_phases(",
    ),
}


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


def test_major_workflows_have_primary_examples() -> None:
    for relative_path, required_snippets in MAJOR_WORKFLOW_COVERAGE.items():
        path = EXAMPLES / relative_path
        assert path.is_file(), relative_path
        source = path.read_text(encoding="utf-8")
        for snippet in required_snippets:
            assert snippet in source, f"{relative_path}: missing {snippet}"


def test_examples_are_grouped_by_domain() -> None:
    assert not list(EXAMPLES.glob("*.py"))
    assert not list(EXAMPLES.glob("*.sh"))
    for domain_name in ("tensor_debug", "memory_debug"):
        domain = EXAMPLES / domain_name
        assert not list(domain.glob("*.py")), domain_name
        assert not list(domain.glob("*.sh")), domain_name
        for workflow in ("probe", "recorder", "cli"):
            workflow_dir = domain / workflow
            assert workflow_dir.is_dir(), workflow_dir
            assert any(workflow_dir.iterdir()), workflow_dir

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


def test_root_readme_memory_output_uses_device_aware_identities() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    memory_section = readme.split("## Memory Quick Start", 1)[1].split(
        "## Agent Workflows", 1
    )[0]
    output = memory_section.split("Output from the tested run:", 1)[1].split(
        "Read the report top-down:", 1
    )[0]

    assert "  device[0]\n" in output
    assert "pool[0,0] (default)" in output
    assert "pool[1,0] (private)" in output
    assert "stream[" in output
    assert "CUDA scope: device-wide, includes other processes" in output
    assert "residual:" in output
    assert "diagnostics:" in output
    assert "(delta " not in output
    assert "each device's `pool[0,0]`" in memory_section


def test_tensor_summary_statistics_are_defined() -> None:
    guide = (ROOT / "docs" / "tensor_debug.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")

    for document in (guide, api):
        normalized = re.sub(r"\s+", " ", document)
        assert "population standard deviation" in normalized
        assert "finite elements" in normalized
        assert "`zero_count`" in normalized


def test_memory_report_glossary_covers_rendered_terms() -> None:
    guide = (ROOT / "docs" / "memory_debug.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    checklist = (ROOT / "docs" / "release_checklist.md").read_text(encoding="utf-8")

    for label in (
        "`snapshot_peak`",
        "`snapshot_blocks`",
        "`owner_event_peak`",
        "`unreusable_event_peak`",
    ):
        assert label in guide
        assert label in api
    assert "`diagnostics` line is intentionally sparse" in guide
    assert "`allocation stack coverage` is attributed" in guide
    assert "`instance stack coverage` is the sum" in guide
    for confidence in ("`reported`", "`matched`", "`ambiguous`", "`unknown`"):
        assert confidence in guide
        assert confidence in api
    for match_kind in (
        "`same_probe`",
        "`same_run`",
        "`default`",
        "`mapped`",
        "`reference_only`",
        "`candidate_only`",
    ):
        assert match_kind in guide
        assert match_kind in api
    assert "`change_gap = candidate_change - baseline_change`" in guide
    assert "`change_gap = candidate_change - baseline_change`" in api
    assert "every user-visible text/HTML label" in checklist


def test_expandable_inactive_metric_is_documented() -> None:
    guide = (ROOT / "docs" / "memory_debug.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    checklist = (ROOT / "docs" / "release_checklist.md").read_text(encoding="utf-8")

    stats_section = api.split("`MemoryStats` stores", 1)[1].split("### MemoryRun", 1)[0]
    phase_section = api.split("`PhaseMetric` contains", 1)[1].split(
        "A private-pool mapping", 1
    )[0]
    for document in (guide, stats_section, phase_section):
        assert "`expandable_inactive_bytes`" in document
    assert "expandable inactive" in checklist
    assert "](../examples/memory_debug/probe/expandable_segments.py)" in guide


def test_workflow_docs_separate_control_flow_from_containment() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "docs" / "architecture.md",
        ROOT / "docs" / "tensor_debug.md",
        ROOT / "docs" / "memory_debug.md",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in documents)
    normalized = re.sub(r"\s+", " ", combined.replace("`", ""))

    forbidden = (
        "Probe -> ProbeSnapshot -> Observation",
        "Run -> Point -> Observation",
        "Recorder -> Run -> Point -> Observation",
        "Both public workflows use the same domain hierarchy",
    )
    for phrase in forbidden:
        assert phrase not in normalized

    readme = documents[0].read_text(encoding="utf-8")
    readme_normalized = re.sub(r"\s+", " ", readme.replace("`", ""))
    assert "snapshot() returns a standalone ProbeSnapshot" in readme_normalized
    assert "finish() returns a Run" in readme_normalized
    assert "same ownerless Observation model within each domain" in readme_normalized


def test_mermaid_edges_name_their_relationships() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "docs" / "architecture.md",
    )
    for document in documents:
        in_mermaid = False
        for line in document.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped == "```mermaid":
                in_mermaid = True
                continue
            if in_mermaid and stripped == "```":
                in_mermaid = False
                continue
            if not in_mermaid or ("-->" not in line and ".->" not in line):
                continue
            if "-->" in line:
                assert "-->|" in line, f"{document}: unlabeled edge: {stripped}"
            else:
                assert "-.->" not in line, f"{document}: unlabeled edge: {stripped}"


def test_cli_workflow_uses_configured_python_for_torchrun() -> None:
    script = (EXAMPLES / "memory_debug" / "cli" / "workflows.sh").read_text(
        encoding="utf-8"
    )
    assert '"${PYTHON_BIN}" -m torch.distributed.run' in script
    assert "GROUP_ROOT=" in script
    assert "GROUPS=" not in script


def _command_names(parser_builder) -> set[str]:
    parser = parser_builder()
    subcommands = next(action for action in parser._actions if action.dest == "command")
    return set(subcommands.choices)


def test_cli_workflows_cover_every_command() -> None:
    cases = (
        (
            tensor_cli_parser,
            EXAMPLES / "tensor_debug" / "cli" / "workflows.sh",
            "TCGD_TENSOR_BIN",
        ),
        (
            memory_cli_parser,
            EXAMPLES / "memory_debug" / "cli" / "workflows.sh",
            "TCGD_MEMORY_BIN",
        ),
    )
    for parser_builder, path, variable in cases:
        script = path.read_text(encoding="utf-8")
        pattern = re.compile(rf'"\$\{{{variable}\}}"\s+([a-z][a-z-]+)')
        assert set(pattern.findall(script)) == _command_names(parser_builder)


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


def test_package_root_public_facade() -> None:
    assert set(torch_cudagraph_debug.__all__) == {
        "__version__",
        "CudaGraphDebugError",
        "FrozenJSONValue",
        "JSONScalar",
        "JSONValue",
        "NativeExtensionUnavailableError",
    }
    assert all(
        hasattr(torch_cudagraph_debug, name) for name in torch_cudagraph_debug.__all__
    )


def test_api_reference_covers_every_supported_export() -> None:
    reference = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    modules = (torch_cudagraph_debug, tensor_debug, memory_debug, advanced)
    for module in modules:
        for name in module.__all__:
            assert name in reference, (
                f"docs/api.md does not cover {module.__name__}.{name}"
            )


def test_memory_result_models_have_field_level_reference() -> None:
    reference = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    section = reference.split("Supporting public row models", 1)[1].split(
        "State comparisons", 1
    )[0]
    models = (
        "MemoryAllocatorScopeComparison",
        "MemoryPoolComparison",
        "MemoryObservationComparison",
        "MemoryAllocatorScopeTimelineEntry",
        "MemoryPoolTimelineEntry",
        "MemoryObservationTimelineEntry",
        "MemoryPhaseComponents",
        "MemoryAllocatorScopePhaseDecomposition",
        "MemoryPoolPhaseDecomposition",
        "AllocationStackCoverage",
        "AllocationStackSummary",
        "AllocationStackDelta",
        "AllocatorEventSummary",
        "CohortPointState",
        "CohortSizeBucket",
        "CohortSizeOutcome",
        "CohortBirth",
        "CohortFreeRequest",
        "CohortFreeCompletion",
        "AllocationCohort",
        "MemoryRankPointState",
        "MemoryRankPointAggregate",
        "MemoryRankPhaseDecomposition",
        "MemoryRankPoolPhaseDecomposition",
        "MemoryMetricExtrema",
        "MemoryRunGroupPhaseAggregate",
    )
    for model in models:
        assert f"`{model}`" in section, model


def test_public_markdown_has_balanced_fences() -> None:
    for document in _public_markdown_files():
        lines = document.read_text(encoding="utf-8").splitlines()
        assert sum(line.startswith("```") for line in lines) % 2 == 0, document


def test_docs_do_not_split_hyphenated_words_across_lines() -> None:
    pattern = re.compile(r"[A-Za-z0-9]-\n[ \t]*[a-z]")
    for document in sorted((ROOT / "docs").glob("*.md")):
        text = document.read_text(encoding="utf-8")
        assert pattern.search(text) is None, document


def test_markdown_shell_continuations_are_single_backslashes() -> None:
    doubled = re.compile(r"\\\\$", re.MULTILINE)
    for document in _public_markdown_files():
        assert doubled.search(document.read_text(encoding="utf-8")) is None, document


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
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in checklist
    assert "python -m pytest -s -q" in checklist
    assert "--no-build-isolation" in checklist
    assert "architecture spec" not in checklist.lower()
    assert "\npip install" not in checklist


def test_public_docs_use_supported_api_terminology() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in _public_markdown_files()
    )
    assert "stable API" not in combined
    assert "stable facade" not in combined
    assert "clear_snapshot" not in combined
    assert re.search(r"status queries,\s+clear,", combined) is None


def test_examples_use_current_memory_option_keywords() -> None:
    allowed = {
        "MemoryAttributionOptions": {
            "stacks",
            "events",
            "lifetimes",
            "display",
        },
        "MemoryLifetimeOptions": {"display"},
    }
    for path in sorted(EXAMPLES.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if function_name not in allowed:
                continue
            keywords = {item.arg for item in node.keywords if item.arg is not None}
            assert keywords <= allowed[function_name], (
                f"{path}: unsupported {function_name} keywords "
                f"{sorted(keywords - allowed[function_name])}"
            )


def test_memory_docs_distinguish_lifetime_and_attribution_options() -> None:
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "memory_debug.md").read_text(encoding="utf-8")

    assert "MemoryLifetimeOptions" in api
    assert "options: MemoryLifetimeOptions | None" in api
    assert "options=MemoryLifetimeOptions(" in guide
    assert "attribution=MemoryAttributionOptions(" in guide
