"""CUDA allocator memory debugging with explicit points and immutable runs."""

from .allocator_snapshot import MemoryObservationKey
from .attribution import MemoryAttributionOptions, MemoryLifetimeOptions
from .comparison import compare_phases, compare_points, compare_snapshots
from .errors import (
    MemoryBundleError,
    MemoryDebugError,
    MemoryHistoryError,
    MemoryOwnershipError,
)
from .probe import MemoryProbe
from .snapshots import MemoryProbeSnapshot
from .recording import (
    MemoryObservation,
    MemoryPoint,
    MemoryRange,
    MemoryRecorder,
    MemoryRun,
)
from .reports import (
    MemoryAllocationLifetimeAnalysis,
    MemoryPhaseComparison,
    MemoryPointComparison,
    MemorySnapshotComparison,
    MemoryRunGroupPhaseComparison,
    MemoryRunGroupSummary,
    MemoryTimeline,
)
from .run_groups import MemoryRunGroup, compare_run_group_phases
from .stats import MemoryStats

__all__ = [
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
