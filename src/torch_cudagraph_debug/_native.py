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
        "torch-cudagraph-debug native extension is unavailable, so this install cannot "
        "collect tensors from a live process. Build the package against a CUDA-enabled "
        "PyTorch installation, for example with `pip install --no-build-isolation .`. "
        "Installs made with TCGD_NO_TENSOR_COLLECTION=1 omit the extension by design; "
        "memory collection and analysis and offline tensor bundle analysis remain "
        "fully available."
    ) from _EXTENSION_ERROR
