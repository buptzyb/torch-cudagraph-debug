from __future__ import annotations

import pytest

import torch_cudagraph_debug.memory_debug as memory_debug
from torch_cudagraph_debug import CudaGraphDebugError
from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryBundleError,
    MemoryDebugError,
    MemoryHistoryError,
    MemoryOwnershipError,
)


def test_memory_facade_is_intentionally_small() -> None:
    assert memory_debug.__all__ == [
        "MemoryProbe",
        "MemoryProbeSnapshot",
        "MemoryRecorder",
        "MemoryRun",
        "MemoryRunGroup",
        "MemoryPoint",
        "MemoryObservation",
        "MemoryObservationKey",
        "MemoryStats",
        "MemoryRange",
        "MemoryTimeline",
        "MemorySnapshotComparison",
        "MemoryPointComparison",
        "MemoryPhaseComparison",
        "MemoryRunGroupSummary",
        "MemoryRunGroupPhaseComparison",
        "MemoryAllocationLifetimeAnalysis",
        "MemoryAttributionOptions",
        "MemoryLifetimeOptions",
        "compare_snapshots",
        "compare_points",
        "compare_phases",
        "compare_run_group_phases",
        "MemoryDebugError",
        "MemoryHistoryError",
        "MemoryBundleError",
        "MemoryOwnershipError",
    ]


def test_memory_errors_share_package_base() -> None:
    for error in (
        MemoryDebugError,
        MemoryHistoryError,
        MemoryBundleError,
        MemoryOwnershipError,
    ):
        assert issubclass(error, CudaGraphDebugError)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"on_missing": "ignore"}, "on_missing"),
        ({"stack_depth": 0}, "stack_depth"),
        ({"limit": 0}, "limit"),
    ],
)
def test_attribution_options_validate(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MemoryAttributionOptions(**kwargs)  # type: ignore[arg-type]


def test_advanced_helpers_live_in_explicit_module() -> None:
    from torch_cudagraph_debug.memory_debug import advanced

    assert advanced.__all__ == [
        "AllocatorEventSummary",
        "AllocationStackCoverage",
        "AllocationStackDelta",
        "AllocationStackSummary",
        "EventWindow",
        "MemoryObservationKey",
        "MemoryStats",
        "AllocatorSnapshotData",
        "AllocatorTraceEntry",
        "ALLOCATION_LIFETIME_ACTIONS",
        "KNOWN_TRACE_ACTIONS",
        "allocation_stack_coverage",
        "compare_allocation_stacks",
        "extract_event_window",
        "format_comparison",
        "format_bytes",
        "format_delta_bytes",
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
    ]
    assert callable(advanced.summarize_snapshot)
