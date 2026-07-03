"""Snapshot types returned by tensor debug probes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

import torch


from .recording import (
    TensorObservation,
    TensorObservationKey,
    _validate_observation_name,
    _validate_observation_sequence,
)


@dataclass(frozen=True)
class TensorCheckStatus:
    """Latest check status for a tensor probe."""

    ok: bool
    message: str
    replay_index: int
    order: int
    name: str | None
    invocation_index: int

    @property
    def key(self) -> TensorObservationKey | None:
        if self.name is None or self.invocation_index < 0:
            return None
        return TensorObservationKey(self.name, self.invocation_index)


@dataclass(frozen=True)
class TensorProbeSnapshot:
    """All tensor observations recorded from one CUDA Graph replay."""

    probe_id: str
    probe_name: str
    replay_index: int
    timestamp: float
    observations: tuple[TensorObservation, ...]

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("probe_id must be non-empty")
        if not self.probe_name:
            raise ValueError("probe_name must be non-empty")
        if self.replay_index < 0:
            raise ValueError("replay_index must be non-negative")
        _validate_observation_sequence(self.observations, owner="tensor snapshot")

    @cached_property
    def by_key(self) -> Mapping[TensorObservationKey, TensorObservation]:
        return MappingProxyType({item.key: item for item in self.observations})

    def observation(
        self,
        name: str | None = None,
        invocation_index: int = 0,
    ) -> TensorObservation:
        resolved_name = (
            self.probe_name if name is None else _validate_observation_name(name)
        )
        key = TensorObservationKey(resolved_name, invocation_index)
        try:
            return self.by_key[key]
        except KeyError as exc:
            raise KeyError(
                f"tensor observation {resolved_name!r}[{invocation_index}] "
                f"does not exist at replay {self.replay_index}"
            ) from exc

    def tensor(
        self,
        name: str | None = None,
        invocation_index: int = 0,
    ) -> torch.Tensor:
        return self.observation(name, invocation_index).tensor()

    def descriptor(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "probe_name": self.probe_name,
            "replay_index": self.replay_index,
            "timestamp": self.timestamp,
            "observation_count": len(self.observations),
        }
