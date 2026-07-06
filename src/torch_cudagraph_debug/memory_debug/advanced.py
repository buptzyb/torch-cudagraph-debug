"""Experimental low-level allocator snapshot helpers.

These names are public for custom analysis, but may change in minor releases.
The high-level user-facing API lives in torch_cudagraph_debug.memory_debug.
"""

from .events import (
    AllocatorEventSummary,
    EventWindow,
    extract_event_window,
    summarize_allocator_events,
)
from .stats import MemoryStats
from .aggregation import summarize_pools
from .stacks import (
    AllocationStackCoverage,
    AllocationStackDelta,
    AllocationStackSummary,
    allocation_stack_coverage,
    compare_allocation_stacks,
    summarize_allocation_stacks,
)
from .allocator_snapshot import (
    ALLOCATION_LIFETIME_ACTIONS,
    AllocatorSnapshotData,
    AllocatorTraceEntry,
    KNOWN_TRACE_ACTIONS,
    MemoryObservationKey,
    format_comparison,
    format_bytes,
    format_delta_bytes,
    normalize_pool_id,
    normalize_snapshot,
    normalize_trace_entries,
    pool_id_label,
    stack_key_from_frames,
    stream_label,
    summarize_segments,
    summarize_snapshot,
)

__all__ = [
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
