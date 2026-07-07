from __future__ import annotations

import pytest

import torch_cudagraph_debug.memory_debug as memory_debug
from torch_cudagraph_debug import CudaGraphDebugError
from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryBundleError,
    MemoryDebugError,
    MemoryDisplayOptions,
    MemoryHistoryError,
    MemoryOwnershipError,
)


def test_memory_facade_exports_public_result_types() -> None:
    assert set(memory_debug.__all__) == {
        "AllocationCohort",
        "AllocationStackCoverage",
        "AllocationStackDelta",
        "AllocationStackSummary",
        "AllocatorEventSummary",
        "AllocatorScope",
        "CohortBirth",
        "CohortFreeCompletion",
        "CohortFreeRequest",
        "CohortPointState",
        "CohortSizeBucket",
        "CohortSizeOutcome",
        "MatchKind",
        "MemoryAllocationLifetimeAnalysis",
        "MemoryAllocatorScopeComparison",
        "MemoryAllocatorScopePhaseDecomposition",
        "MemoryAllocatorScopeTimelineEntry",
        "MemoryAttributionOptions",
        "MemoryAttributionStatus",
        "MemoryBundleError",
        "MemoryDebugError",
        "MemoryDisplayOptions",
        "MemoryEvidenceStatus",
        "MemoryHistoryError",
        "MemoryLifetimeOptions",
        "MemoryLifetimeSelection",
        "MemoryLifecycleDelta",
        "MemoryMetricExtrema",
        "MemoryObservation",
        "MemoryObservationComparison",
        "MemoryObservationKey",
        "MemoryObservationTimelineEntry",
        "MemoryOwnershipError",
        "MemoryPhaseComparison",
        "MemoryPhaseComponents",
        "MemoryPoint",
        "MemoryPointComparison",
        "MemoryPoolComparison",
        "MemoryPoolKey",
        "MemoryPoolPhaseDecomposition",
        "MemoryPoolTimelineEntry",
        "MemoryProbe",
        "MemoryProbeSnapshot",
        "MemoryRange",
        "MemoryRankPhaseDecomposition",
        "MemoryRankPointAggregate",
        "MemoryRankPointState",
        "MemoryRankPoolPhaseDecomposition",
        "MemoryRecorder",
        "MemoryRun",
        "MemoryRunGroup",
        "MemoryRunGroupPhaseAggregate",
        "MemoryRunGroupPhaseComparison",
        "MemoryRunGroupSummary",
        "MemorySnapshotComparison",
        "MemoryStatMetric",
        "MemoryStats",
        "MemoryStatsDelta",
        "MemoryTimeline",
        "MissingPolicy",
        "PhaseMetric",
        "PoolId",
        "StreamId",
        "compare_phases",
        "compare_points",
        "compare_run_group_phases",
        "compare_snapshots",
    }
    assert len(memory_debug.__all__) == len(set(memory_debug.__all__))
    assert all(hasattr(memory_debug, name) for name in memory_debug.__all__)


def test_memory_errors_share_package_base() -> None:
    for error in (
        MemoryDebugError,
        MemoryHistoryError,
        MemoryBundleError,
        MemoryOwnershipError,
    ):
        assert issubclass(error, CudaGraphDebugError)


def test_attribution_and_display_options_validate_independently() -> None:
    with pytest.raises(ValueError, match="on_missing"):
        MemoryAttributionOptions(on_missing="ignore")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stack_depth"):
        MemoryDisplayOptions(stack_depth=0)
    with pytest.raises(ValueError, match="limit"):
        MemoryDisplayOptions(limit=0)
    with pytest.raises(TypeError, match="MemoryDisplayOptions"):
        MemoryAttributionOptions(display={})  # type: ignore[arg-type]


def test_advanced_helpers_live_in_explicit_module() -> None:
    from torch_cudagraph_debug.memory_debug import advanced

    assert set(advanced.__all__) == {
        "ALLOCATION_LIFETIME_ACTIONS",
        "AllocatorEventSummary",
        "AllocatorSnapshotData",
        "AllocatorTraceEntry",
        "AllocationStackCoverage",
        "AllocationStackDelta",
        "AllocationStackSummary",
        "EventWindow",
        "KNOWN_TRACE_ACTIONS",
        "MemoryObservationKey",
        "MemoryPoolKey",
        "MemoryStats",
        "allocation_stack_coverage",
        "compare_allocation_stacks",
        "extract_event_window",
        "format_bytes",
        "format_comparison",
        "format_delta_bytes",
        "mutable_snapshot",
        "normalize_pool_id",
        "normalize_snapshot",
        "normalize_trace_entries",
        "pool_id_label",
        "stack_key_from_frames",
        "stream_label",
        "summarize_allocation_stacks",
        "summarize_allocator_events",
        "summarize_pools",
        "summarize_segments",
        "summarize_snapshot",
    }
    assert len(advanced.__all__) == len(set(advanced.__all__))
    assert all(hasattr(advanced, name) for name in advanced.__all__)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"reserved_bytes": 1, "active_bytes": 2},
        {"reserved_bytes": 2, "active_bytes": 1, "allocated_bytes": 2},
        {"reserved_bytes": 2, "active_bytes": 1, "requested_bytes": 2},
        {"block_count": 1, "inactive_block_count": 2},
        {"segment_count": 1, "expandable_segment_count": 2},
        {"reserved_bytes": 1, "expandable_reserved_bytes": 2},
        {"reserved_bytes": 2, "active_bytes": 1, "largest_inactive_block_bytes": 2},
    ),
)
def test_memory_stats_reject_impossible_allocator_state(
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        memory_debug.MemoryStats(**kwargs)
