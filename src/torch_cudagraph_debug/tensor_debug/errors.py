"""Tensor debug domain exceptions."""

from torch_cudagraph_debug._errors import CudaGraphDebugError


class TensorDebugError(CudaGraphDebugError):
    """Base error for tensor debug probes."""


class TensorMismatchError(TensorDebugError, AssertionError):
    """Raised when a tensor comparison action reported a mismatch."""


class TensorBundleError(TensorDebugError):
    """Raised for malformed, unsupported, or unreadable tensor bundles."""


class TensorOwnershipError(TensorDebugError):
    """Raised when an object is used with a tensor run that does not own it."""


class TensorPayloadUnavailableError(TensorDebugError):
    """Raised when an operation requires a summary-only tensor payload."""
