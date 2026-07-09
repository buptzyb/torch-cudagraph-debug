from __future__ import annotations

from dataclasses import fields

import torch_cudagraph_debug.memory_debug as memory_debug
import torch_cudagraph_debug.tensor_debug as tensor_debug
from torch_cudagraph_debug.memory_debug import (
    MemoryAllocationLifetimeAnalysis,
    MemoryObservation,
    MemoryPoint,
    MemoryPointComparison,
    MemoryProbeSnapshot,
    MemorySnapshotComparison,
)
from torch_cudagraph_debug.memory_debug._collector import _MemoryCollector
from torch_cudagraph_debug.memory_debug.advanced import (
    AllocationStackDelta,
    AllocationStackSummary,
    AllocatorEventSummary,
)
from torch_cudagraph_debug.memory_debug.reports import (
    MemoryPhaseComparison,
    MemoryRunGroupPhaseComparison,
    MemoryRunGroupSummary,
    MemoryTimeline,
)
from torch_cudagraph_debug.tensor_debug import (
    TensorObservation,
    TensorPoint,
    TensorPointComparison,
    TensorProbeSnapshot,
    TensorSnapshotComparison,
)
from torch_cudagraph_debug.tensor_debug._collector import (
    _EagerTensorCollector,
    _TensorCollector,
)


def _field_names(model: type[object]) -> tuple[str, ...]:
    return tuple(item.name for item in fields(model))


def test_public_workflow_roles_are_symmetric() -> None:
    for role in (
        "Probe",
        "ProbeSnapshot",
        "Recorder",
        "Run",
        "Point",
        "Observation",
        "SnapshotComparison",
    ):
        assert f"Tensor{role}" in tensor_debug.__all__
        assert f"Memory{role}" in memory_debug.__all__
    assert "compare_snapshots" in tensor_debug.__all__
    assert "compare_snapshots" in memory_debug.__all__


def test_observations_are_ownerless_shared_leaves() -> None:
    ownership_fields = {"probe_id", "run_id", "point_index", "replay_index"}
    assert ownership_fields.isdisjoint(_field_names(TensorObservation))
    assert ownership_fields.isdisjoint(_field_names(MemoryObservation))
    assert _field_names(TensorObservation)[:3] == (
        "order",
        "name",
        "invocation_index",
    )
    assert "probe_name" not in _field_names(TensorObservation)
    assert "probe_name" in _field_names(TensorProbeSnapshot)
    assert "observations" in _field_names(TensorProbeSnapshot)
    assert "observations" in _field_names(MemoryProbeSnapshot)
    assert "observations" in _field_names(TensorPoint)
    assert "observations" in _field_names(MemoryPoint)


def test_collectors_remain_private_and_domain_specific() -> None:
    assert _TensorCollector.__name__.startswith("_")
    assert _EagerTensorCollector.__name__.startswith("_")
    assert _MemoryCollector.__name__.startswith("_")
    assert _TensorCollector.__bases__ == (object,)
    assert _MemoryCollector.__bases__ == (object,)


def test_snapshot_comparisons_are_sibling_state_results() -> None:
    assert TensorSnapshotComparison.__bases__ == TensorPointComparison.__bases__
    assert MemorySnapshotComparison.__bases__ == MemoryPointComparison.__bases__
    assert "point_comparison" not in _field_names(TensorSnapshotComparison)
    assert _field_names(TensorSnapshotComparison)[:4] == (
        "reference",
        "candidate",
        "options",
        "observation_comparisons",
    )
    assert _field_names(MemorySnapshotComparison)[:5] == (
        "reference",
        "candidate",
        "allocator_scope_comparisons",
        "pool_comparisons",
        "observation_comparisons",
    )


def test_lifetime_source_is_workflow_neutral() -> None:
    assert _field_names(MemoryAllocationLifetimeAnalysis)[:3] == (
        "source_kind",
        "source_id",
        "source_name",
    )
    assert "run" not in _field_names(MemoryAllocationLifetimeAnalysis)


def test_memory_result_fields_encode_their_semantics() -> None:
    assert tuple(field.name for field in fields(MemoryPointComparison))[:5] == (
        "reference",
        "candidate",
        "allocator_scope_comparisons",
        "pool_comparisons",
        "observation_comparisons",
    )
    assert "attribution_status" in _field_names(MemoryPointComparison)
    assert tuple(field.name for field in fields(MemoryTimeline))[:5] == (
        "run",
        "allocator_scope_entries",
        "pool_entries",
        "observation_entries",
        "point_comparisons",
    )
    assert tuple(field.name for field in fields(MemoryPhaseComparison)) == (
        "baseline_name",
        "candidate_name",
        "baseline_change",
        "candidate_change",
        "start_gap",
        "end_gap",
        "allocator_scope_decomposition",
        "pool_decomposition",
        "display_stack_depth",
        "display_limit",
    )
    assert tuple(field.name for field in fields(MemoryRunGroupSummary)) == (
        "run_group",
        "rank_points",
        "point_aggregates",
        "warnings",
    )
    assert tuple(field.name for field in fields(MemoryRunGroupPhaseComparison)) == (
        "baseline_group",
        "candidate_group",
        "rank_comparisons",
        "rank_decomposition",
        "rank_pool_decomposition",
        "phase_aggregates",
        "warnings",
        "display_stack_depth",
        "display_limit",
    )


def test_memory_attribution_models_own_structured_frames() -> None:
    for model in (AllocationStackSummary, AllocationStackDelta, AllocatorEventSummary):
        names = _field_names(model)
        assert "stack_frames" in names
        assert "stack_key" not in names


def test_public_classes_have_semantic_docstrings() -> None:
    for domain in (tensor_debug, memory_debug):
        for name in domain.__all__:
            value = getattr(domain, name)
            if not isinstance(value, type):
                continue
            doc = value.__doc__ or ""
            assert doc.strip(), f"{domain.__name__}.{name} has no docstring"
            assert not doc.startswith(f"{value.__name__}("), (
                f"{domain.__name__}.{name} exposes only an auto-generated "
                "constructor signature"
            )
