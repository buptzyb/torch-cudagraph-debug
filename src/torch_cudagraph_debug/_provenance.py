"""Runtime provenance helpers shared by debug domains."""

from __future__ import annotations

import platform
import socket
from typing import Any


def runtime_provenance() -> dict[str, Any]:
    import torch

    from torch_cudagraph_debug import __version__

    return {
        "producer": {
            "name": "torch-cudagraph-debug",
            "version": __version__,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }


def initialized_device_provenance(
    device: object | None = None,
) -> dict[str, Any] | None:
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    index = (
        torch.cuda.current_device() if device is None else torch.device(device).index
    )
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    value: dict[str, Any] = {
        "index": index,
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
    }
    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is not None:
        value["uuid"] = str(device_uuid)
    return value
