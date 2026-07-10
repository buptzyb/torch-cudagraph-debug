"""Standalone snapshots returned by memory debug probes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Any, cast

from torch_cudagraph_debug.types import (
    FrozenJSONValue,
    JSONValue,
    _freeze_json,
)

from ._pool_identity import MemoryObservationKey, MemoryPoolKey
from .aggregation import summarize_allocator_scopes, summarize_pools
from .recording import MemoryObservation
from .stats import (
    AllocatorScope,
    DeviceMemorySample,
    MemoryStats,
    validate_device_memory,
)


@dataclass(frozen=True)
class MemoryProbeSnapshot:
    """One complete allocator snapshot captured by a MemoryProbe."""

    probe_id: str
    probe_name: str
    snapshot_index: int
    timestamp: float
    boundary_marker: str
    observations: tuple[MemoryObservation, ...]
    warnings: tuple[str, ...] = ()
    device_memory: Mapping[int, DeviceMemorySample] = field(default_factory=dict)
    _boundary_recorded: bool = field(default=True, repr=False, compare=False)
    _raw_snapshot: JSONValue | FrozenJSONValue = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("probe_id must be non-empty")
        if not self.probe_name:
            raise ValueError("probe_name must be non-empty")
        if type(self.snapshot_index) is not int or self.snapshot_index < 0:
            raise ValueError("snapshot_index must be a non-negative integer")
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not math.isfinite(float(self.timestamp))
        ):
            raise ValueError("snapshot timestamp must be finite")
        if not self.boundary_marker:
            raise ValueError("boundary_marker must be non-empty")
        if type(self._boundary_recorded) is not bool:
            raise TypeError("memory snapshot boundary status must be boolean")
        expected = list(range(len(self.observations)))
        orders = [item.order for item in self.observations]
        if orders != expected:
            raise ValueError(
                "memory snapshot observations must be contiguous and ordered"
            )
        keys = [item.key for item in self.observations]
        if len(set(keys)) != len(keys):
            raise ValueError("memory snapshot observation keys must be unique")
        object.__setattr__(
            self, "device_memory", validate_device_memory(self.device_memory)
        )
        object.__setattr__(
            self,
            "_raw_snapshot",
            _freeze_json(cast(JSONValue, self._raw_snapshot)),
        )

    @cached_property
    def by_key(self) -> Mapping[MemoryObservationKey, MemoryObservation]:
        return MappingProxyType({item.key: item for item in self.observations})

    @cached_property
    def observation_stats(self) -> Mapping[MemoryObservationKey, MemoryStats]:
        return MappingProxyType(
            {key: observation.stats for key, observation in self.by_key.items()}
        )

    @cached_property
    def pool_stats(self) -> Mapping[MemoryPoolKey, MemoryStats]:
        return MappingProxyType(summarize_pools(self.observation_stats))

    @cached_property
    def allocator_scope_stats(self) -> Mapping[AllocatorScope, MemoryStats]:
        return MappingProxyType(summarize_allocator_scopes(self.pool_stats))

    def observation(
        self, device_index: int, pool_id: Any, stream: Any
    ) -> MemoryObservation:
        key = MemoryObservationKey(device_index, pool_id, stream)
        try:
            return self.by_key[key]
        except KeyError as exc:
            raise KeyError(
                f"memory observation {key.label} does not exist "
                f"at snapshot {self.snapshot_index}"
            ) from exc

    @property
    def allocator_settings(self) -> Mapping[str, FrozenJSONValue]:
        raw = self.allocator_state()
        if not isinstance(raw, Mapping):
            return MappingProxyType({})
        settings = raw.get("allocator_settings", MappingProxyType({}))
        if not isinstance(settings, Mapping):
            return MappingProxyType({})
        return cast(Mapping[str, FrozenJSONValue], settings)

    @cached_property
    def _state(self) -> Mapping[str, FrozenJSONValue]:
        raw = self.raw_snapshot()
        if not isinstance(raw, Mapping):
            return MappingProxyType({})
        state = {key: value for key, value in raw.items() if key != "device_traces"}
        return cast(
            Mapping[str, FrozenJSONValue],
            _freeze_json(cast(JSONValue, state)),
        )

    def allocator_state(self) -> Mapping[str, FrozenJSONValue]:
        """Return allocator state without cumulative device event traces."""

        return self._state

    def raw_snapshot(self) -> FrozenJSONValue:
        """Return a recursively immutable view of the allocator snapshot."""

        return cast(FrozenJSONValue, self._raw_snapshot)

    def descriptor(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "probe_name": self.probe_name,
            "snapshot_index": self.snapshot_index,
            "timestamp": self.timestamp,
            "boundary_marker": self.boundary_marker,
            "observation_count": len(self.observations),
            "device_memory": {
                str(device): sample.to_dict()
                for device, sample in self.device_memory.items()
            },
            "warnings": list(self.warnings),
        }
