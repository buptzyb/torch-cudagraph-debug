"""Private allocator snapshot acquisition for Probe and Recorder workflows."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from .allocator_snapshot import (
    AllocatorSnapshotData,
    MemoryObservationKey,
    normalize_snapshot,
    summarize_segments,
)
from .stats import MemoryStats

SynchronizeTarget = bool | torch.cuda.Stream | torch.device
SnapshotProvider = Callable[[str], AllocatorSnapshotData]


@dataclass(frozen=True)
class _CollectedMemorySnapshot:
    timestamp: float
    boundary_marker: str
    raw_snapshot: AllocatorSnapshotData
    summaries: tuple[tuple[MemoryObservationKey, MemoryStats], ...]
    warnings: tuple[str, ...]


def validate_synchronize_target(synchronize: SynchronizeTarget) -> None:
    if not isinstance(synchronize, (bool, torch.cuda.Stream, torch.device)):
        raise TypeError(
            "synchronize must be a bool, torch.cuda.Stream, or torch.device"
        )
    if isinstance(synchronize, torch.device) and synchronize.type != "cuda":
        raise ValueError("synchronize device must identify a CUDA device")


class _MemoryCollector:
    """Capture and normalize the complete private PyTorch allocator snapshot."""

    def __init__(
        self,
        *,
        synchronize: SynchronizeTarget = True,
        snapshot_provider: SnapshotProvider | None = None,
    ) -> None:
        validate_synchronize_target(synchronize)
        self.synchronize = synchronize
        self._snapshot_provider = snapshot_provider

    @property
    def uses_snapshot_provider(self) -> bool:
        return self._snapshot_provider is not None

    def capture(
        self,
        boundary_marker: str,
        *,
        synchronize: SynchronizeTarget | None = None,
    ) -> _CollectedMemorySnapshot:
        if not boundary_marker:
            raise ValueError("boundary_marker must be non-empty")
        selected = self.synchronize if synchronize is None else synchronize
        validate_synchronize_target(selected)
        warnings: list[str] = []

        if self._snapshot_provider is not None:
            raw = self._snapshot_provider(boundary_marker)
        else:
            self._synchronize(selected, warnings)
            raw = self._capture_torch_snapshot(boundary_marker, warnings)

        snapshot = _snapshot_envelope(raw)
        segments = normalize_snapshot(snapshot)
        return _CollectedMemorySnapshot(
            timestamp=time.time(),
            boundary_marker=boundary_marker,
            raw_snapshot=snapshot,
            summaries=tuple(summarize_segments(segments).items()),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _synchronize(
        synchronize: SynchronizeTarget,
        warnings: list[str],
    ) -> None:
        if isinstance(synchronize, bool) and not synchronize:
            return
        if not torch.cuda.is_available():
            return
        capture_state = _current_stream_capture_state()
        if capture_state is None:
            warnings.append(
                "could not determine CUDA graph capture state; requested allocator "
                "synchronization was skipped"
            )
            return
        if capture_state:
            warnings.append(
                "requested allocator synchronization was skipped during "
                "CUDA graph capture"
            )
            return
        if isinstance(synchronize, torch.cuda.Stream):
            synchronize.synchronize()
        elif isinstance(synchronize, torch.device):
            torch.cuda.synchronize(synchronize)
        else:
            torch.cuda.synchronize()

    @staticmethod
    def _capture_torch_snapshot(
        boundary_marker: str,
        warnings: list[str],
    ) -> AllocatorSnapshotData:
        memory = torch.cuda.memory
        get_metadata = getattr(memory, "_get_memory_metadata", None)
        set_metadata = getattr(memory, "_set_memory_metadata", None)
        previous_metadata: str | None = None
        marker_set = False
        if callable(get_metadata) and callable(set_metadata):
            try:
                previous_metadata = get_metadata()
                set_metadata(boundary_marker)
                marker_set = True
            except Exception as exc:
                warnings.append(
                    "could not set allocator metadata marker: "
                    f"{type(exc).__name__}: {exc}"
                )
        else:
            warnings.append(
                "PyTorch memory metadata APIs are unavailable; "
                "event boundaries cannot be marked"
            )
        try:
            return memory._snapshot()
        finally:
            if marker_set and callable(set_metadata):
                try:
                    set_metadata(previous_metadata or "")
                except Exception as exc:
                    warnings.append(
                        "could not restore allocator metadata: "
                        f"{type(exc).__name__}: {exc}"
                    )


def _snapshot_envelope(snapshot: AllocatorSnapshotData) -> AllocatorSnapshotData:
    if isinstance(snapshot, Mapping):
        envelope = dict(snapshot)
    else:
        envelope = {"segments": list(snapshot)}
    envelope.setdefault("device_traces", [])
    envelope.setdefault("external_annotations", [])
    envelope.setdefault("allocator_settings", {})
    return envelope


def _current_stream_capture_state() -> bool | None:
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    if not callable(is_capturing):
        return None
    try:
        return bool(is_capturing())
    except Exception:
        return None
