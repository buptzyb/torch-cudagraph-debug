"""Standalone snapshots returned by memory debug probes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Any

from ._pool_identity import PoolId
from .aggregation import summarize_allocator_scopes, summarize_pools
from .allocator_snapshot import (
    AllocatorSnapshotData,
    MemoryObservationKey,
    normalize_pool_id,
    normalize_stream,
)
from .recording import MemoryObservation
from .stats import MemoryStats


@dataclass(frozen=True)
class MemoryProbeSnapshot:
    """One complete allocator snapshot captured by a MemoryProbe."""

    probe_id: str
    probe_name: str
    index: int
    timestamp: float
    boundary_marker: str
    observations: tuple[MemoryObservation, ...]
    warnings: tuple[str, ...] = ()
    _raw_snapshot: AllocatorSnapshotData = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("probe_id must be non-empty")
        if not self.probe_name:
            raise ValueError("probe_name must be non-empty")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("snapshot index must be a non-negative integer")
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not math.isfinite(float(self.timestamp))
        ):
            raise ValueError("snapshot timestamp must be finite")
        if not self.boundary_marker:
            raise ValueError("boundary_marker must be non-empty")
        expected = list(range(len(self.observations)))
        orders = [item.order for item in self.observations]
        if orders != expected:
            raise ValueError(
                "memory snapshot observations must be contiguous and ordered"
            )
        keys = [item.key for item in self.observations]
        if len(set(keys)) != len(keys):
            raise ValueError("memory snapshot observation keys must be unique")

    @cached_property
    def by_key(self) -> Mapping[MemoryObservationKey, MemoryObservation]:
        return MappingProxyType({item.key: item for item in self.observations})

    @cached_property
    def observation_stats(self) -> Mapping[MemoryObservationKey, MemoryStats]:
        return MappingProxyType(
            {key: observation.stats for key, observation in self.by_key.items()}
        )

    @cached_property
    def pool_stats(self) -> Mapping[PoolId, MemoryStats]:
        return MappingProxyType(summarize_pools(self.observation_stats))

    @cached_property
    def allocator_scope_stats(self) -> Mapping[str, MemoryStats]:
        return MappingProxyType(summarize_allocator_scopes(self.pool_stats))

    def observation(self, pool_id: Any, stream: Any) -> MemoryObservation:
        key = MemoryObservationKey(normalize_pool_id(pool_id), normalize_stream(stream))
        try:
            return self.by_key[key]
        except KeyError as exc:
            raise KeyError(
                f"memory observation pool={key.pool_id!r} stream={key.stream!r} "
                f"does not exist at snapshot {self.index}"
            ) from exc

    def raw_snapshot(self) -> AllocatorSnapshotData:
        return self._raw_snapshot

    def descriptor(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "probe_name": self.probe_name,
            "index": self.index,
            "timestamp": self.timestamp,
            "boundary_marker": self.boundary_marker,
            "observation_count": len(self.observations),
            "warnings": list(self.warnings),
        }
