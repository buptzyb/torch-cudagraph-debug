"""CUDA allocator memory debugging with explicit points and immutable runs."""

from .core import (
    AttributionOptions,
    MemoryPoint,
    MemoryRange,
    MemoryRecorder,
    MemoryRun,
    compare_phases,
    compare_points,
)
from .errors import (
    MemoryBundleError,
    MemoryDebugError,
    MemoryHistoryError,
    MemoryOwnershipError,
)
from .groups import MemoryRunGroup, compare_group_phases
from .reports import (
    AllocationLifetimeReport,
    GroupPhaseComparison,
    MemoryComparison,
    MemoryGroupSummary,
    MemoryTimeline,
    PhaseComparison,
)

__all__ = [
    "MemoryRecorder",
    "MemoryRun",
    "MemoryRunGroup",
    "MemoryPoint",
    "MemoryRange",
    "MemoryTimeline",
    "MemoryComparison",
    "PhaseComparison",
    "MemoryGroupSummary",
    "GroupPhaseComparison",
    "AllocationLifetimeReport",
    "AttributionOptions",
    "compare_points",
    "compare_phases",
    "compare_group_phases",
    "MemoryDebugError",
    "MemoryHistoryError",
    "MemoryBundleError",
    "MemoryOwnershipError",
]
