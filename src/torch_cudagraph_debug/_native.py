"""Internal native extension loader."""

from __future__ import annotations

from types import ModuleType

from ._errors import NativeExtensionUnavailableError

try:
    from . import _C as _EXTENSION
except Exception as exc:  # pragma: no cover - depends on local CUDA build.
    _EXTENSION: ModuleType | None = None
    _EXTENSION_ERROR: BaseException | None = exc
else:
    _EXTENSION_ERROR = None


def extension_available() -> bool:
    """Return whether the compiled CUDA extension is importable."""

    return _EXTENSION is not None


def require_native() -> ModuleType:
    """Return the native extension or raise a user-facing error."""

    if _EXTENSION is not None:
        return _EXTENSION
    raise NativeExtensionUnavailableError(
        "torch-cudagraph-debug native extension is unavailable. In full Tensor Debug "
        "mode, reinstall against a CUDA-enabled PyTorch installation with "
        "`TCGD_TENSOR_DEBUG_MODE=full python -m pip install --no-cache-dir "
        "--force-reinstall --no-build-isolation torch-cudagraph-debug`. "
        "Offline mode omits the extension by design; select it with "
        "TCGD_TENSOR_DEBUG_MODE=offline during "
        "installation. Memory Debug and offline tensor bundle analysis remain "
        "available."
    ) from _EXTENSION_ERROR
