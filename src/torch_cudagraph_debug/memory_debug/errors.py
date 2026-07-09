"""Memory debug domain exceptions."""

from torch_cudagraph_debug._errors import CudaGraphDebugError


class MemoryDebugError(CudaGraphDebugError):
    """Base error for CUDA allocator memory debugging."""


class MemoryHistoryError(MemoryDebugError):
    """Requested allocator history data is unavailable or incomplete."""


class MemoryHistoryDisabledError(MemoryHistoryError):
    """Required allocator history evidence is unavailable.

    A device in the analyzed range has no usable allocator event history, or —
    for stack attribution — nonempty active state has zero allocation-frame
    coverage at a compared endpoint. Enable
    ``torch.cuda.memory._record_memory_history()`` (with stack context for
    stack attribution) before the allocations of interest are made on every
    analyzed device.
    """


class MemoryHistoryBoundaryError(MemoryHistoryError):
    """A point boundary could not be recorded for allocator event analysis."""


class MemoryHistoryTruncatedError(MemoryHistoryError):
    """A required boundary marker is missing from the device trace.

    The bounded history ring buffer may have overwritten it, or history
    recording was enabled after the boundary; the tool cannot distinguish
    these. Enable ``_record_memory_history()`` before the first analyzed
    point and keep it enabled, raise the history buffer size
    (``max_entries``), or record points more frequently so each interval
    fits inside the ring buffer.
    """


class MemoryReconciliationError(MemoryDebugError):
    """Allocator event boundaries or events disagree with snapshot state.

    The recorded boundary order is invalid, or the event stream and allocator
    snapshots disagree without an identifiable recording gap. Possible causes
    include overlapping marker-bearing collection, allocator activity during
    the non-atomic marker/snapshot window, corrupted input, or a bug in
    torch-cudagraph-debug.
    """


class MemoryBundleError(MemoryDebugError):
    """A persisted memory run is missing, invalid, or unsupported."""


class MemoryOwnershipError(MemoryDebugError):
    """A snapshot, point, or range was used with an object that does not own
    it, or same-run points were passed to a cross-run API."""
