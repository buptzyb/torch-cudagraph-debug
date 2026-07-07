"""Immutable models for reference/candidate memory comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Literal

from ._pool_identity import MemoryObservationKey, MemoryPoolKey
from .stats import AllocatorScope, MemoryStats, MemoryStatsDelta

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
]

PHASE_METRICS: tuple[PhaseMetric, ...] = (
    "reserved_bytes",
    "allocated_bytes",
    "active_bytes",
    "awaiting_free_bytes",
    "inactive_bytes",
    "requested_bytes",
    "internal_fragmentation_bytes",
    "expandable_reserved_bytes",
)


@dataclass(frozen=True)
class MemoryPhaseComponents:
    """The four-point memory equation for one metric."""

    metric: PhaseMetric
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
class MemoryLifecycleDelta:
    """Address-based observations from ordered snapshots with shared identity."""

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
