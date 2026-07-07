"""Snapshot types returned by tensor debug probes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

import torch

from ._identity import TensorObservationKey, validate_observation_name
from .recording import (
    TensorObservation,
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

    def __post_init__(self) -> None:
        if type(self.ok) is not bool:
            raise TypeError("ok must be a boolean")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if type(self.replay_index) is not int or self.replay_index < 0:
            raise ValueError("replay_index must be a non-negative integer")
        if type(self.order) is not int or type(self.invocation_index) is not int:
            raise TypeError("order and invocation_index must be integers")
        if self.ok:
            if self.order != -1 or self.name is not None or self.invocation_index != -1:
                raise ValueError("successful check status must not identify a failure")
        else:
            if self.order < 0 or self.invocation_index < 0 or not self.name:
                raise ValueError("failed check status must identify one observation")

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
    snapshot_index: int
    replay_index: int
    timestamp: float
    observations: tuple[TensorObservation, ...]

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("probe_id must be non-empty")
        if not self.probe_name:
            raise ValueError("probe_name must be non-empty")
        if type(self.snapshot_index) is not int or self.snapshot_index < 0:
            raise ValueError("snapshot_index must be a non-negative integer")
        if type(self.replay_index) is not int or self.replay_index < 0:
            raise ValueError("replay_index must be a non-negative integer")
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not math.isfinite(float(self.timestamp))
        ):
            raise ValueError("timestamp must be finite")
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
            self.probe_name if name is None else validate_observation_name(name)
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
            "snapshot_index": self.snapshot_index,
            "replay_index": self.replay_index,
            "timestamp": self.timestamp,
            "observation_count": len(self.observations),
        }
