"""Experimental low-level allocator snapshot helpers.

These names are public for custom analysis, but may change in minor releases.
The high-level user-facing API lives in torch_cudagraph_debug.memory_debug.
"""

from typing import cast

from torch_cudagraph_debug.types import (
    FrozenJSONValue,
    JSONValue,
    _freeze_json,
    _thaw_json,
)

from ._pool_identity import (
    MemoryObservationKey,
    MemoryPoolKey,
    normalize_pool_id,
    pool_id_label,
    stream_label,
)
from .aggregation import summarize_pools
from .allocator_snapshot import (
    ALLOCATION_LIFETIME_ACTIONS,
    KNOWN_TRACE_ACTIONS,
    AllocatorSnapshotData,
    AllocatorTraceEntry,
    format_bytes,
    format_comparison,
    format_delta_bytes,
    normalize_snapshot,
    normalize_trace_entries,
    stack_key_from_frames,
    summarize_segments,
    summarize_snapshot,
)
from .events import (
    AllocatorEventSummary,
    EventWindow,
    extract_event_window,
    summarize_allocator_events,
)
from .recording import MemoryPoint
from .snapshots import MemoryProbeSnapshot
from .stacks import (
    AllocationStackCoverage,
    AllocationStackDelta,
    AllocationStackSummary,
    allocation_stack_coverage,
    compare_allocation_stacks,
    summarize_allocation_stacks,
)
from .stats import MemoryStats


def mutable_snapshot(
    source: MemoryPoint | MemoryProbeSnapshot | FrozenJSONValue,
) -> JSONValue:
    """Return a deep mutable copy of a point, probe snapshot, or raw view."""

    raw = (
        source.allocator_state()
        if isinstance(source, MemoryPoint)
        else source.raw_snapshot()
        if isinstance(source, MemoryProbeSnapshot)
        else source
    )
    return _thaw_json(_freeze_json(cast(FrozenJSONValue, raw)))


__all__ = [
    "AllocatorEventSummary",
    "AllocationStackCoverage",
    "AllocationStackDelta",
    "AllocationStackSummary",
    "EventWindow",
    "MemoryObservationKey",
    "MemoryPoolKey",
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
]
