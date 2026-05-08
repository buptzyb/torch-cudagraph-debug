"""Tensor debug domain exceptions."""

from torch_cudagraph_debug._errors import CudaGraphDebugError


class TensorDebugError(CudaGraphDebugError):
    """Base error for tensor debug probes."""


class TensorCompareMismatchError(TensorDebugError, AssertionError):
    """Raised when a tensor comparison action reported a mismatch."""
