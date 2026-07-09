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
    """All observations returned by one query point.

    The snapshot contains one CUDA Graph replay, the latest eager sample per
    name, or zero-filled captured slots before the first replay. Eager and
    pre-replay snapshots use replay index 0.
    """

    probe_id: str
    probe_name: str
    snapshot_index: int
    replay_index: int
    timestamp: float
    observations: tuple[TensorObservation, ...]
    eager_overwrites: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("probe_id must be non-empty")
        if not self.probe_name:
            raise ValueError("probe_name must be non-empty")
        overwrite_names: list[str] = []
        for entry in self.eager_overwrites:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not entry[0]
                or type(entry[1]) is not int
                or entry[1] < 1
            ):
                raise ValueError(
                    "eager_overwrites entries must be (name, positive count) pairs"
                )
            name = entry[0]
            if name in overwrite_names:
                raise ValueError("eager_overwrites observation names must be unique")
            overwrite_names.append(name)
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

        if overwrite_names and self.replay_index != 0:
            raise ValueError(
                "eager_overwrites require an eager snapshot with replay_index 0"
            )
        for name in overwrite_names:
            matching_invocations = [
                item.invocation_index for item in self.observations if item.name == name
            ]
            if matching_invocations != [0]:
                raise ValueError(
                    "eager_overwrites names must reference exactly one "
                    "invocation-0 observation"
                )

    @cached_property
    def eager_overwrite_counts(self) -> Mapping[str, int]:
        """Re-sample counts for the eager names that were overwritten.

        Only names with a count of one or more appear; a nonzero count means
        intermediate values were superseded before this snapshot, which holds
        only the latest sample of that name.
        """

        return MappingProxyType(dict(self.eager_overwrites))

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
