"""Compatibility imports for shared runtime provenance helpers."""

from __future__ import annotations

from torch_cudagraph_debug._provenance import (
    initialized_device_provenance,
    runtime_provenance,
)

__all__ = ["initialized_device_provenance", "runtime_provenance"]
