"""Installed build configuration."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, distribution
from typing import Literal, cast

_DISTRIBUTION_NAME = "torch-cudagraph-debug"
_TENSOR_DEBUG_MODE_FILE = "tcgd_tensor_debug_mode"
_TENSOR_DEBUG_MODES = frozenset({"full", "offline"})

TensorDebugMode = Literal["full", "offline"]


def _load_tensor_debug_mode() -> TensorDebugMode:
    try:
        installed = distribution(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return "full"
    value = installed.read_text(_TENSOR_DEBUG_MODE_FILE)
    if value is None:
        return "full"
    mode = value.strip()
    if mode not in _TENSOR_DEBUG_MODES:
        raise RuntimeError(
            "torch-cudagraph-debug installation has an invalid tensor-debug mode "
            f"marker: {mode!r}; reinstall the package"
        )
    return cast(TensorDebugMode, mode)


_TENSOR_DEBUG_MODE = _load_tensor_debug_mode()
_OFFLINE_MODE_NOTICE_EMITTED = False


def _emit_offline_mode_notice() -> None:
    """Report reduced capabilities once when an offline install is imported."""

    global _OFFLINE_MODE_NOTICE_EMITTED
    if _TENSOR_DEBUG_MODE != "offline" or _OFFLINE_MODE_NOTICE_EMITTED:
        return
    _OFFLINE_MODE_NOTICE_EMITTED = True
    print(
        "torch-cudagraph-debug: offline mode (live TensorProbe/TensorRecorder disabled). "
        "To enable full mode, reinstall with TCGD_TENSOR_DEBUG_MODE=full; "
        "see the installation guide.",
        file=sys.stderr,
    )


def tensor_debug_mode() -> TensorDebugMode:
    """Return the installed Tensor Debug capability mode."""

    return _TENSOR_DEBUG_MODE
