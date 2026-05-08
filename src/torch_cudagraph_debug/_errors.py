"""Common exceptions for torch-cudagraph-debug."""


class CudaGraphDebugError(RuntimeError):
    """Base error raised by torch-cudagraph-debug."""


class NativeExtensionUnavailableError(CudaGraphDebugError):
    """Raised when the compiled CUDA extension cannot be imported."""
