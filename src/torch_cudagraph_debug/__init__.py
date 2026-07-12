"""CUDA Graph debugging utilities for PyTorch."""

__version__ = "0.2.0"

from ._build_config import _emit_offline_mode_notice, tensor_debug_mode
from ._errors import CudaGraphDebugError, NativeExtensionUnavailableError
from .types import FrozenJSONValue, JSONScalar, JSONValue

_emit_offline_mode_notice()

__all__ = [
    "__version__",
    "CudaGraphDebugError",
    "FrozenJSONValue",
    "JSONScalar",
    "JSONValue",
    "NativeExtensionUnavailableError",
    "tensor_debug_mode",
]
