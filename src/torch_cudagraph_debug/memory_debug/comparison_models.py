"""Immutable models for reference/candidate memory comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Literal

from ._pool_identity import MemoryObservationKey, MemoryPoolKey
from .stats import (
    AllocatorScope,
    DeviceMemorySample,
    MemoryStats,
    MemoryStatsDelta,
)

MatchKind = Literal[
    "same_run",
    "same_probe",
    "default",
    "mapped",
    "reference_only",
    "candidate_only",
]


PhaseMetric = Literal[
    "reserved_bytes",
    "allocated_bytes",
    "active_bytes",
    "awaiting_free_bytes",
    "inactive_bytes",
    "requested_bytes",
    "internal_fragmentation_bytes",
    "expandable_reserved_bytes",
    "expandable_inactive_bytes",
]

DeviceMemoryMetric = Literal[
    "used_bytes",
    "free_bytes",
    "total_bytes",
    "allocator_reserved_bytes",
    "unattributed_device_bytes",
]

DEVICE_MEMORY_METRICS: tuple[DeviceMemoryMetric, ...] = (
    "used_bytes",
    "free_bytes",
    "total_bytes",
    "allocator_reserved_bytes",
    "unattributed_device_bytes",
)


PHASE_METRICS: tuple[PhaseMetric, ...] = (
    "reserved_bytes",
    "allocated_bytes",
    "active_bytes",
    "awaiting_free_bytes",
    "inactive_bytes",
    "requested_bytes",
    "internal_fragmentation_bytes",
    "expandable_reserved_bytes",
    "expandable_inactive_bytes",
)


@dataclass(frozen=True)
class MemoryPhaseComponents:
    """The four-point memory equation for one metric."""

    metric: PhaseMetric | DeviceMemoryMetric
    start_gap_bytes: int
    baseline_change_bytes: int
    candidate_change_bytes: int
    end_gap_bytes: int

    @property
    def change_gap_bytes(self) -> int:
        return self.candidate_change_bytes - self.baseline_change_bytes

    @property
    def identity_holds(self) -> bool:
        return (
            self.end_gap_bytes
            == self.start_gap_bytes
            + self.candidate_change_bytes
            - self.baseline_change_bytes
        )

    @property
    def changed(self) -> bool:
        return any(
            (
                self.start_gap_bytes,
                self.baseline_change_bytes,
                self.candidate_change_bytes,
                self.end_gap_bytes,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "start_gap_bytes": self.start_gap_bytes,
            "baseline_change_bytes": self.baseline_change_bytes,
            "candidate_change_bytes": self.candidate_change_bytes,
            "change_gap_bytes": self.change_gap_bytes,
            "end_gap_bytes": self.end_gap_bytes,
            "identity_holds": self.identity_holds,
        }


@dataclass(frozen=True)
class MemoryAllocatorScopePhaseDecomposition:
    """Four-point decomposition for one allocator scope and metric."""

    scope: AllocatorScope
    components: MemoryPhaseComponents

    def __post_init__(self) -> None:
        if self.components.metric not in PHASE_METRICS:
            raise ValueError(
                "allocator-scope decomposition requires an allocator metric"
            )

    @property
    def changed(self) -> bool:
        return self.components.changed

    def to_dict(self) -> dict[str, object]:
        return {"scope": self.scope, **self.components.to_dict()}

    def to_row(self) -> dict[str, object]:
        return self.to_dict()


@dataclass(frozen=True)
class MemoryPoolPhaseDecomposition:
    """Four-point decomposition for one mapped pool pair and metric."""

    baseline_key: MemoryPoolKey
    candidate_key: MemoryPoolKey
    components: MemoryPhaseComponents

    def __post_init__(self) -> None:
        if self.components.metric not in PHASE_METRICS:
            raise ValueError("pool decomposition requires an allocator metric")

    @property
    def changed(self) -> bool:
        return self.components.changed

    @property
    def label(self) -> str:
        return f"{self.baseline_key.label} -> {self.candidate_key.label}"

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_pool_key": self.baseline_key.label,
            "candidate_pool_key": self.candidate_key.label,
            "pool": self.label,
            **self.components.to_dict(),
        }

    def to_row(self) -> dict[str, object]:
        return self.to_dict()


@dataclass(frozen=True)
class MemoryDevicePhaseDecomposition:
    """Four-point decomposition for one CUDA device and metric."""

    device_index: int
    components: MemoryPhaseComponents

    def __post_init__(self) -> None:
        if type(self.device_index) is not int or self.device_index < 0:
            raise ValueError("device_index must be a non-negative integer")
        if self.components.metric not in DEVICE_MEMORY_METRICS:
            raise ValueError("device decomposition requires a device memory metric")

    @property
    def changed(self) -> bool:
        return self.components.changed

    def to_dict(self) -> dict[str, object]:
        return {"device_index": self.device_index, **self.components.to_dict()}

    def to_row(self) -> dict[str, object]:
        return self.to_dict()


@dataclass(frozen=True)
class MemoryAllocatorScopeComparison:
    """Reference/candidate allocator totals for one logical pool scope."""

    scope: AllocatorScope
    reference: MemoryStats
    candidate: MemoryStats
    delta: MemoryStatsDelta

    @property
    def changed(self) -> bool:
        return self.delta.changed

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "reference": self.reference.to_dict(),
            "candidate": self.candidate.to_dict(),
            "delta": self.delta.to_dict(),
        }

    def to_row(self) -> dict[str, object]:
        return _comparison_row(
            {"scope": self.scope},
            self.reference,
            self.candidate,
            self.delta,
            None,
        )


@dataclass(frozen=True)
class MemoryDeviceComparison:
    """Reference/candidate device-wide CUDA Runtime memory for one device.

    Samples come from ``torch.cuda.mem_get_info`` and cover the whole device.
    Unattributed device bytes subtract allocator reserved bytes from the
    recording process snapshot from device-global usage. The remainder can
    include that process context and external allocations plus every other
    process sharing the GPU. A side without a sample reports ``None`` and
    produces no deltas.
    """

    device_index: int
    reference: DeviceMemorySample | None
    candidate: DeviceMemorySample | None
    reference_allocator_reserved_bytes: int
    candidate_allocator_reserved_bytes: int

    def __post_init__(self) -> None:
        if type(self.device_index) is not int or self.device_index < 0:
            raise ValueError("device_index must be a non-negative integer")
        if self.reference is None and self.candidate is None:
            raise ValueError("device comparison requires at least one sample")
        for name in ("reference", "candidate"):
            sample = getattr(self, name)
            if sample is not None and not isinstance(sample, DeviceMemorySample):
                raise TypeError(f"{name} must be a DeviceMemorySample or None")
        for name in (
            "reference_allocator_reserved_bytes",
            "candidate_allocator_reserved_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def reference_unattributed_device_bytes(self) -> int | None:
        if self.reference is None:
            return None
        return self.reference.used_bytes - self.reference_allocator_reserved_bytes

    @property
    def candidate_unattributed_device_bytes(self) -> int | None:
        if self.candidate is None:
            return None
        return self.candidate.used_bytes - self.candidate_allocator_reserved_bytes

    @property
    def delta_used_bytes(self) -> int | None:
        if self.reference is None or self.candidate is None:
            return None
        return self.candidate.used_bytes - self.reference.used_bytes

    @property
    def delta_free_bytes(self) -> int | None:
        if self.reference is None or self.candidate is None:
            return None
        return self.candidate.free_bytes - self.reference.free_bytes

    @property
    def delta_total_bytes(self) -> int | None:
        if self.reference is None or self.candidate is None:
            return None
        return self.candidate.total_bytes - self.reference.total_bytes

    @property
    def delta_allocator_reserved_bytes(self) -> int:
        return (
            self.candidate_allocator_reserved_bytes
            - self.reference_allocator_reserved_bytes
        )

    @property
    def delta_unattributed_device_bytes(self) -> int | None:
        reference = self.reference_unattributed_device_bytes
        candidate = self.candidate_unattributed_device_bytes
        if reference is None or candidate is None:
            return None
        return candidate - reference

    @property
    def changed(self) -> bool:
        if self.reference is None or self.candidate is None:
            return True
        return bool(
            self.delta_used_bytes
            or self.delta_free_bytes
            or self.delta_total_bytes
            or self.delta_allocator_reserved_bytes
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "device_index": self.device_index,
            "reference": self.reference.to_dict() if self.reference else None,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "reference_allocator_reserved_bytes": (
                self.reference_allocator_reserved_bytes
            ),
            "candidate_allocator_reserved_bytes": (
                self.candidate_allocator_reserved_bytes
            ),
            "reference_unattributed_device_bytes": self.reference_unattributed_device_bytes,
            "candidate_unattributed_device_bytes": self.candidate_unattributed_device_bytes,
            "delta": {
                "used_bytes": self.delta_used_bytes,
                "free_bytes": self.delta_free_bytes,
                "total_bytes": self.delta_total_bytes,
                "allocator_reserved_bytes": self.delta_allocator_reserved_bytes,
                "unattributed_device_bytes": self.delta_unattributed_device_bytes,
            },
        }

    def to_row(self) -> dict[str, object]:
        def side(
            prefix: str, sample: DeviceMemorySample | None, reserved: int
        ) -> dict[str, object]:
            return {
                f"{prefix}_free_bytes": sample.free_bytes if sample else None,
                f"{prefix}_total_bytes": sample.total_bytes if sample else None,
                f"{prefix}_used_bytes": sample.used_bytes if sample else None,
                f"{prefix}_allocator_reserved_bytes": reserved,
                f"{prefix}_unattributed_device_bytes": (
                    sample.used_bytes - reserved if sample else None
                ),
            }

        return {
            "device_index": self.device_index,
            **side(
                "reference", self.reference, self.reference_allocator_reserved_bytes
            ),
            **side(
                "candidate", self.candidate, self.candidate_allocator_reserved_bytes
            ),
            "delta_used_bytes": self.delta_used_bytes,
            "delta_free_bytes": self.delta_free_bytes,
            "delta_total_bytes": self.delta_total_bytes,
            "delta_allocator_reserved_bytes": self.delta_allocator_reserved_bytes,
            "delta_unattributed_device_bytes": self.delta_unattributed_device_bytes,
        }


@dataclass(frozen=True)
class MemoryLifecycleDelta:
    """Lifecycle changes between ordered snapshots with shared identity.

    Complete addresses provide exact identity; missing addresses fall back to
    size-based multiset matching and make the enclosing result approximate.
    Expandable segments compare by mapped address ranges, so their new and
    removed bytes are the bytes mapped in or unmapped out.
    """

    new_segment_bytes: int = 0
    removed_segment_bytes: int = 0
    newly_active_bytes: int = 0
    became_inactive_bytes: int = 0

    @classmethod
    def combine(
        cls, values: Iterable["MemoryLifecycleDelta"]
    ) -> "MemoryLifecycleDelta":
        items = tuple(values)
        return cls(
            new_segment_bytes=sum(item.new_segment_bytes for item in items),
            removed_segment_bytes=sum(item.removed_segment_bytes for item in items),
            newly_active_bytes=sum(item.newly_active_bytes for item in items),
            became_inactive_bytes=sum(item.became_inactive_bytes for item in items),
        )

    @property
    def changed(self) -> bool:
        return any(asdict(self).values())

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryPoolComparison:
    """One device/pool-level reference/candidate comparison."""

    reference_key: MemoryPoolKey | None
    candidate_key: MemoryPoolKey | None
    match: MatchKind
    reference: MemoryStats
    candidate: MemoryStats
    delta: MemoryStatsDelta
    lifecycle: MemoryLifecycleDelta | None

    @property
    def pool_key(self) -> MemoryPoolKey:
        value = self.candidate_key or self.reference_key
        assert value is not None
        return value

    @property
    def changed(self) -> bool:
        return self.delta.changed or bool(self.lifecycle and self.lifecycle.changed)

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_key": _pool_key_value(self.reference_key),
            "candidate_key": _pool_key_value(self.candidate_key),
            "match": self.match,
            "reference": self.reference.to_dict(),
            "candidate": self.candidate.to_dict(),
            "delta": self.delta.to_dict(),
            "lifecycle": self.lifecycle.to_dict() if self.lifecycle else None,
        }

    def to_row(self) -> dict[str, object]:
        return _comparison_row(
            {
                "reference_key": _pool_key_label(self.reference_key),
                "candidate_key": _pool_key_label(self.candidate_key),
                "match": self.match,
            },
            self.reference,
            self.candidate,
            self.delta,
            self.lifecycle,
        )


@dataclass(frozen=True)
class MemoryObservationComparison:
    """One device/pool/stream reference/candidate comparison."""

    reference_key: MemoryObservationKey | None
    candidate_key: MemoryObservationKey | None
    match: MatchKind
    reference: MemoryStats
    candidate: MemoryStats
    delta: MemoryStatsDelta
    lifecycle: MemoryLifecycleDelta | None

    @property
    def changed(self) -> bool:
        return self.delta.changed or bool(self.lifecycle and self.lifecycle.changed)

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_key": _observation_key_value(self.reference_key),
            "candidate_key": _observation_key_value(self.candidate_key),
            "match": self.match,
            "reference": self.reference.to_dict(),
            "candidate": self.candidate.to_dict(),
            "delta": self.delta.to_dict(),
            "lifecycle": self.lifecycle.to_dict() if self.lifecycle else None,
        }

    def to_row(self) -> dict[str, object]:
        return _comparison_row(
            {
                "reference_key": _observation_key_label(self.reference_key),
                "candidate_key": _observation_key_label(self.candidate_key),
                "match": self.match,
            },
            self.reference,
            self.candidate,
            self.delta,
            self.lifecycle,
        )


def _comparison_row(
    identity: dict[str, object],
    reference: MemoryStats,
    candidate: MemoryStats,
    delta: MemoryStatsDelta,
    lifecycle: MemoryLifecycleDelta | None,
) -> dict[str, object]:
    row = dict(identity)
    row.update(
        {f"reference_{key}": value for key, value in reference.to_dict().items()}
    )
    row.update(
        {f"candidate_{key}": value for key, value in candidate.to_dict().items()}
    )
    row.update({f"delta_{key}": value for key, value in delta.to_dict().items()})
    if lifecycle is not None:
        row.update(
            {f"lifecycle_{key}": value for key, value in lifecycle.to_dict().items()}
        )
    return row


def _pool_key_value(key: MemoryPoolKey | None) -> dict[str, object] | None:
    if key is None:
        return None
    return {"device_index": key.device_index, "pool_id": list(key.pool_id)}


def _observation_key_value(
    key: MemoryObservationKey | None,
) -> dict[str, object] | None:
    if key is None:
        return None
    return {
        "device_index": key.device_index,
        "pool_id": list(key.pool_id),
        "stream": key.stream,
    }


def _pool_key_label(key: MemoryPoolKey | None) -> str:
    return key.label if key is not None else ""


def _observation_key_label(key: MemoryObservationKey | None) -> str:
    return key.label if key is not None else ""
