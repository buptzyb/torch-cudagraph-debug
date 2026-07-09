"""Low-ceremony allocator snapshot workflow."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from ._collector import (
    DeviceSelector,
    SnapshotProvider,
    SynchronizeTarget,
    _MemoryCollector,
)
from .errors import MemoryOwnershipError
from .recording import MemoryObservation
from .snapshots import MemoryProbeSnapshot

if TYPE_CHECKING:
    from .attribution import MemoryAttributionOptions
    from .reports import MemorySnapshotComparison


class MemoryProbe:
    """Capture standalone allocator snapshots without creating a run."""

    def __init__(
        self,
        name: str = "memory",
        *,
        devices: DeviceSelector = None,
        synchronize: SynchronizeTarget = True,
    ) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        self.name = name
        self.synchronize = synchronize
        self._probe_id = uuid.uuid4().hex
        self._next_snapshot_index = 0
        self._collector = _MemoryCollector(
            devices=devices,
            synchronize=synchronize,
        )

    @property
    def devices(self) -> tuple[int, ...] | None:
        """Selected device indices, resolved lazily by the first snapshot."""

        return self._collector.devices

    @classmethod
    def _from_snapshot_provider(
        cls,
        provider: SnapshotProvider,
        *,
        name: str = "memory",
        devices: DeviceSelector = None,
        synchronize: SynchronizeTarget = True,
    ) -> "MemoryProbe":
        probe = cls(name=name, devices=devices, synchronize=synchronize)
        probe._collector = _MemoryCollector(
            devices=devices,
            synchronize=synchronize,
            snapshot_provider=provider,
        )
        return probe

    def snapshot(
        self,
        *,
        synchronize: SynchronizeTarget | None = None,
    ) -> MemoryProbeSnapshot:
        """Capture one immutable allocator snapshot for the selected devices.

        ``synchronize`` overrides the probe default for this call only;
        requested synchronization is skipped with a warning during CUDA
        stream capture. Real PyTorch collection resolves device binding on the
        first snapshot, even when that snapshot contains no allocations.
        """

        snapshot_index = self._next_snapshot_index
        marker = (
            "torch-cudagraph-debug:memory-probe:"
            f"{self._probe_id}:{snapshot_index}:{self.name}"
        )
        capture = self._collector.capture(marker, synchronize=synchronize)
        observations = tuple(
            MemoryObservation(
                order=order,
                device_index=key.device_index,
                pool_id=key.pool_id,
                stream=key.stream,
                stats=stats,
            )
            for order, (key, stats) in enumerate(capture.summaries)
        )
        snapshot = MemoryProbeSnapshot(
            probe_id=self._probe_id,
            probe_name=self.name,
            snapshot_index=snapshot_index,
            timestamp=capture.timestamp,
            boundary_marker=capture.boundary_marker,
            observations=observations,
            warnings=capture.warnings,
            _boundary_recorded=capture.boundary_recorded,
            _raw_snapshot=capture.raw_snapshot,
        )
        self._next_snapshot_index += 1
        return snapshot

    def compare(
        self,
        reference: MemoryProbeSnapshot,
        candidate: MemoryProbeSnapshot,
        *,
        attribution: MemoryAttributionOptions | None = None,
    ) -> MemorySnapshotComparison:
        """Compare two chronologically ordered snapshots owned by this probe."""

        for role, snapshot in (
            ("reference", reference),
            ("candidate", candidate),
        ):
            if snapshot.probe_id != self._probe_id:
                raise MemoryOwnershipError(
                    f"{role} snapshot does not belong to MemoryProbe({self.name!r})"
                )
        if candidate.snapshot_index <= reference.snapshot_index:
            raise ValueError("candidate snapshot must follow reference snapshot")

        from .comparison import compare_snapshots

        return compare_snapshots(
            reference,
            candidate,
            attribution=attribution,
        )
