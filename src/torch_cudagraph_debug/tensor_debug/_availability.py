"""Tensor Debug capability-mode checks."""

from torch_cudagraph_debug._build_config import tensor_debug_mode

from .errors import LiveTensorDebugUnavailableError


def require_live_tensor_debug() -> None:
    """Raise when this install supports offline tensor analysis only."""

    if tensor_debug_mode() == "offline":
        raise LiveTensorDebugUnavailableError(
            "live tensor debugging is unavailable because torch-cudagraph-debug "
            "was installed with TCGD_TENSOR_DEBUG_MODE=offline. Reinstall full "
            "mode with `TCGD_TENSOR_DEBUG_MODE=full python -m pip install "
            "--no-cache-dir --force-reinstall --no-build-isolation "
            "torch-cudagraph-debug`. Offline tensor bundle "
            "analysis and Memory Debug remain available."
        )
