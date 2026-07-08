"""Memory debug domain exceptions."""

from torch_cudagraph_debug._errors import CudaGraphDebugError


class MemoryDebugError(CudaGraphDebugError):
    """Base error for CUDA allocator memory debugging."""


class MemoryHistoryError(MemoryDebugError):
    """Requested allocator history data is unavailable or incomplete."""


class MemoryHistoryDisabledError(MemoryHistoryError):
    """A device in the analyzed range has no allocator event history.

    Enable ``torch.cuda.memory._record_memory_history()`` before the
    allocations of interest are made on every analyzed device.
    """


class MemoryHistoryTruncatedError(MemoryHistoryError):
    """A recorded boundary marker was overwritten in the history ring buffer.

    Raise the history buffer size (``max_entries``) or record points more
    frequently so each interval fits inside the ring buffer.
    """


class MemoryReconciliationError(MemoryDebugError):
    """Complete event history could not be reconciled with a snapshot.

    The event stream and the allocator snapshots disagree without an
    identifiable recording gap. This indicates corrupted input data or a bug
    in torch-cudagraph-debug itself; please report it with the message
    details.
    """


class MemoryBundleError(MemoryDebugError):
    """A persisted memory run is missing, invalid, or unsupported."""


class MemoryOwnershipError(MemoryDebugError):
    """A point or range was used with a run that does not own it."""
