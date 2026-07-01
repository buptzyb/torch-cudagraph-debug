"""Experimental low-level allocator snapshot helpers.

These names are public for custom analysis, but may change in minor releases.
The stable user-facing API lives in torch_cudagraph_debug.memory_debug.
"""

from .events import (
    AllocatorEventSummary,
    EventWindow,
    extract_event_window,
    summarize_allocator_events,
)
from .models import MemoryStats
from .totals import summarize_pools
from .stacks import (
    AllocationStackCoverage,
    AllocationStackDelta,
    AllocationStackSummary,
    allocation_stack_coverage,
    compare_allocation_stacks,
    summarize_allocation_stacks,
)
from .summary import (
    GroupKey,
    SnapshotInput,
    TraceEntry,
    format_before_after,
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
    "GroupKey",
    "MemoryStats",
    "SnapshotInput",
    "TraceEntry",
    "allocation_stack_coverage",
    "compare_allocation_stacks",
    "extract_event_window",
    "format_before_after",
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
