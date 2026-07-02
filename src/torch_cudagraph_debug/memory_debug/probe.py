"""Low-ceremony allocator snapshot workflow."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ._collector import (
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
        synchronize: SynchronizeTarget = True,
    ) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        self.name = name
        self.synchronize = synchronize
        self._probe_id = uuid.uuid4().hex
        self._next_index = 0
        self._collector = _MemoryCollector(synchronize=synchronize)

    @classmethod
    def _from_snapshot_provider(
        cls,
        provider: SnapshotProvider,
        *,
        name: str = "memory",
        synchronize: SynchronizeTarget = True,
    ) -> "MemoryProbe":
        probe = cls(name=name, synchronize=synchronize)
        probe._collector = _MemoryCollector(
            synchronize=synchronize,
            snapshot_provider=provider,
        )
        return probe

    def snapshot(
        self,
        *,
        synchronize: SynchronizeTarget | None = None,
    ) -> MemoryProbeSnapshot:
        index = self._next_index
        marker = (
            f"torch-cudagraph-debug:memory-probe:{self._probe_id}:{index}:{self.name}"
        )
        capture = self._collector.capture(marker, synchronize=synchronize)
        observations = tuple(
            MemoryObservation(
                order=order,
                pool_id=key.pool_id,
                stream=key.stream,
                stats=stats,
            )
            for order, (key, stats) in enumerate(capture.summaries)
        )
        snapshot = MemoryProbeSnapshot(
            probe_id=self._probe_id,
            probe_name=self.name,
            index=index,
            timestamp=capture.timestamp,
            boundary_marker=capture.boundary_marker,
            observations=observations,
            warnings=capture.warnings,
            _raw_snapshot=capture.raw_snapshot,
        )
        self._next_index += 1
        return snapshot

    def compare(
        self,
        reference: MemoryProbeSnapshot,
        candidate: MemoryProbeSnapshot,
        *,
        pool_mapping: Mapping[object, object] | None = None,
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
        if candidate.index <= reference.index:
            raise ValueError("candidate snapshot must follow reference snapshot")

        from .comparison import compare_snapshots

        return compare_snapshots(
            reference,
            candidate,
            pool_mapping=pool_mapping,
            attribution=attribution,
        )
