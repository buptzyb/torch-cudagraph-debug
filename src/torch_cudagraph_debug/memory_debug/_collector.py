"""Private allocator snapshot acquisition for Probe and Recorder workflows."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from ._pool_identity import MemoryObservationKey, normalize_device_index
from .allocator_snapshot import (
    AllocatorSnapshotData,
    normalize_snapshot,
    summarize_segments,
)
from .stats import MemoryStats

SynchronizeTarget = bool | torch.cuda.Stream | torch.device
DeviceLike: TypeAlias = int | str | torch.device
DeviceSelector: TypeAlias = DeviceLike | Sequence[DeviceLike] | Literal["all"] | None
SnapshotProvider = Callable[[str], AllocatorSnapshotData]


@dataclass(frozen=True)
class _CollectedMemorySnapshot:
    timestamp: float
    boundary_marker: str
    boundary_recorded: bool
    devices: tuple[int, ...]
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
    """Capture allocator state for the selected CUDA devices; snapshot-derived
    selections (default and "all") adopt devices that first appear after
    binding."""

    def __init__(
        self,
        *,
        devices: DeviceSelector = None,
        synchronize: SynchronizeTarget = True,
        snapshot_provider: SnapshotProvider | None = None,
    ) -> None:
        validate_synchronize_target(synchronize)
        self.synchronize = synchronize
        self._device_selector = devices
        self._requested_devices = _normalize_device_selector(devices)
        self._resolved_devices: tuple[int, ...] | None = None
        self._snapshot_provider = snapshot_provider

    @property
    def uses_snapshot_provider(self) -> bool:
        return self._snapshot_provider is not None

    @property
    def devices(self) -> tuple[int, ...] | None:
        return self._resolved_devices

    def capture(
        self,
        boundary_marker: str,
        *,
        synchronize: SynchronizeTarget | None = None,
    ) -> _CollectedMemorySnapshot:
        if not boundary_marker:
            raise ValueError("boundary_marker must be non-empty")
        selected_sync = self.synchronize if synchronize is None else synchronize
        validate_synchronize_target(selected_sync)
        warnings: list[str] = []

        if self._snapshot_provider is not None:
            raw = self._snapshot_provider(boundary_marker)
            envelope = _snapshot_envelope(raw)
            devices = self._resolve_devices(envelope, warnings)
            boundary_recorded = True
        else:
            devices = self._resolve_devices(None)
            self._synchronize(selected_sync, devices, warnings)
            raw, boundary_recorded = self._capture_torch_snapshot(
                boundary_marker, warnings
            )
            envelope = _snapshot_envelope(raw)

        snapshot = _filter_snapshot_devices(envelope, devices, warnings)
        segments = normalize_snapshot(snapshot, warnings=warnings)
        return _CollectedMemorySnapshot(
            timestamp=time.time(),
            boundary_marker=boundary_marker,
            boundary_recorded=boundary_recorded,
            devices=devices,
            raw_snapshot=snapshot,
            summaries=tuple(summarize_segments(segments).items()),
            warnings=tuple(warnings),
        )

    def _resolve_devices(
        self,
        snapshot: AllocatorSnapshotData | None,
        warnings: list[str] | None = None,
    ) -> tuple[int, ...]:
        if self._resolved_devices is not None:
            # Snapshot-derived selections (default and "all") stay open to
            # devices that first appear after binding: a late device must
            # join the selection instead of silently vanishing from every
            # later capture.
            if snapshot is not None and self._requested_devices in (None, "all"):
                discovered = _snapshot_device_indices(snapshot)
                added = tuple(
                    device
                    for device in discovered
                    if device not in self._resolved_devices
                )
                if added:
                    self._resolved_devices = tuple(
                        sorted({*self._resolved_devices, *added})
                    )
                    if warnings is not None:
                        warnings.append(
                            f"memory collection adopted device(s) {list(added)} "
                            "that first appeared after device binding; earlier "
                            "captures do not cover them"
                        )
            return self._resolved_devices
        if self._requested_devices == "all":
            if snapshot is not None:
                devices = _snapshot_device_indices(snapshot)
            elif torch.cuda.is_available():
                devices = tuple(range(torch.cuda.device_count()))
            else:
                devices = ()
        elif isinstance(self._requested_devices, tuple):
            devices = self._requested_devices
        elif snapshot is not None and self._snapshot_provider is not None:
            devices = _snapshot_device_indices(snapshot)
        elif torch.cuda.is_available():
            devices = (torch.cuda.current_device(),)
        else:
            devices = ()
        # Snapshot-derived selections (default and "all") must not bind an
        # empty device set: a first capture taken before any allocation would
        # otherwise pin every later capture to zero devices.
        if (
            not devices
            and snapshot is not None
            and self._snapshot_provider is not None
            and self._requested_devices in (None, "all")
        ):
            return ()
        self._resolved_devices = devices
        return devices

    @staticmethod
    def _synchronize(
        synchronize: SynchronizeTarget,
        devices: tuple[int, ...],
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
            stream_device = normalize_device_index(synchronize.device.index)
            if stream_device not in devices:
                raise ValueError("synchronize stream is not on a selected device")
            synchronize.synchronize()
            return
        if isinstance(synchronize, torch.device):
            device_index = _device_index(synchronize)
            if device_index not in devices:
                raise ValueError("synchronize device is not selected by this collector")
            torch.cuda.synchronize(device_index)
            return
        for device_index in devices:
            torch.cuda.synchronize(device_index)

    @staticmethod
    def _capture_torch_snapshot(
        boundary_marker: str,
        warnings: list[str],
    ) -> tuple[AllocatorSnapshotData, bool]:
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
            return memory._snapshot(), marker_set
        finally:
            if marker_set and callable(set_metadata):
                try:
                    set_metadata(previous_metadata or "")
                except Exception as exc:
                    warnings.append(
                        "could not restore allocator metadata: "
                        f"{type(exc).__name__}: {exc}"
                    )


def _normalize_device_selector(
    devices: DeviceSelector,
) -> tuple[int, ...] | Literal["all"] | None:
    if devices == "all":
        return "all"
    if devices is None:
        return None
    if isinstance(devices, Sequence) and not isinstance(devices, (str, bytes)):
        normalized = tuple(_device_index(item) for item in devices)
    else:
        normalized = (_device_index(devices),)
    if not normalized:
        raise ValueError("devices must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("devices must not contain duplicates")
    # Device order is an implementation detail everywhere downstream, and the
    # bundle loader requires history rows in ascending device order — a
    # caller-ordered selection must not produce an unloadable bundle.
    return tuple(sorted(normalized))


def _device_index(device: DeviceLike) -> int:
    if type(device) is int:
        return normalize_device_index(device)
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("memory debug devices must identify CUDA devices")
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    return normalize_device_index(index)


def _snapshot_envelope(snapshot: AllocatorSnapshotData) -> dict[str, object]:
    if isinstance(snapshot, Mapping):
        envelope = dict(snapshot)
    else:
        envelope = {"segments": list(snapshot)}
    envelope.setdefault("device_traces", [])
    envelope.setdefault("external_annotations", [])
    envelope.setdefault("allocator_settings", {})
    return envelope


def _snapshot_device_indices(snapshot: AllocatorSnapshotData) -> tuple[int, ...]:
    if isinstance(snapshot, Mapping):
        raw_segments = snapshot.get("segments", ())
    else:
        raw_segments = snapshot
    devices: set[int] = set()
    if isinstance(raw_segments, Sequence):
        for segment in raw_segments:
            if isinstance(segment, Mapping):
                devices.add(normalize_device_index(segment.get("device", 0)))
    if isinstance(snapshot, Mapping):
        raw_traces = snapshot.get("device_traces", ())
        if isinstance(raw_traces, Sequence):
            devices.update(range(len(raw_traces)))
    return tuple(sorted(devices))


def _filter_snapshot_devices(
    snapshot: Mapping[str, object],
    devices: tuple[int, ...],
    warnings: list[str] | None = None,
) -> dict[str, object]:
    selected = set(devices)
    filtered = dict(snapshot)
    raw_segments = snapshot.get("segments", ())
    if not isinstance(raw_segments, Sequence):
        raise TypeError("allocator snapshot segments must be a sequence")
    kept_segments = []
    dropped_deviceless = 0
    for segment in raw_segments:
        if not isinstance(segment, Mapping):
            continue
        if normalize_device_index(segment.get("device", 0)) in selected:
            kept_segments.append(segment)
        elif "device" not in segment:
            # Deviceless segments default to device 0; dropping them here
            # would otherwise bypass the normalization-time missing-device
            # warning entirely.
            dropped_deviceless += 1
    filtered["segments"] = kept_segments
    if dropped_deviceless and warnings is not None:
        warnings.append(
            f"{dropped_deviceless} allocator segment(s) missing 'device' were"
            " attributed to device 0 and excluded by the device selection;"
            " their bytes are not represented in this snapshot"
        )
    raw_traces = snapshot.get("device_traces", ())
    if not isinstance(raw_traces, Sequence):
        raise TypeError("allocator snapshot device_traces must be a sequence")
    filtered["device_traces"] = [
        trace if index in selected else [] for index, trace in enumerate(raw_traces)
    ]
    return filtered


def _current_stream_capture_state() -> bool | None:
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    if not callable(is_capturing):
        return None
    try:
        return bool(is_capturing())
    except Exception:
        return None
