"""Memory debug domain exceptions."""

from torch_cudagraph_debug._errors import CudaGraphDebugError


class MemoryDebugError(CudaGraphDebugError):
    """Base error for CUDA allocator memory debugging."""


class MemoryHistoryError(MemoryDebugError):
    """Requested allocator history data is unavailable or incomplete."""


class MemoryBundleError(MemoryDebugError):
    """A persisted memory run is missing, invalid, or unsupported."""


class MemoryOwnershipError(MemoryDebugError):
    """A point or range was used with a run that does not own it."""
