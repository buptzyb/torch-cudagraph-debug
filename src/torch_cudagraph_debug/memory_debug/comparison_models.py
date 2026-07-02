"""Immutable models for reference/candidate memory comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from ._pool_identity import PoolId, pool_id_label, stream_label
from .stats import AllocatorScope, MemoryStats, MemoryStatsDelta

MatchKind = Literal[
    "same_run",
    "same_probe",
    "default",
    "mapped",
    "reference_only",
    "candidate_only",
]


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
    released_bytes: int = 0

    @classmethod
    def combine(
        cls, values: Iterable["MemoryLifecycleDelta"]
    ) -> "MemoryLifecycleDelta":
        items = tuple(values)
        return cls(
            new_segment_bytes=sum(item.new_segment_bytes for item in items),
            removed_segment_bytes=sum(item.removed_segment_bytes for item in items),
            newly_active_bytes=sum(item.newly_active_bytes for item in items),
            released_bytes=sum(item.released_bytes for item in items),
        )

    @property
    def changed(self) -> bool:
        return any(asdict(self).values())

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryPoolComparison:
    """One pool-level reference/candidate comparison."""

    reference_pool_id: PoolId | None
    candidate_pool_id: PoolId | None
    match: MatchKind
    reference: MemoryStats
    candidate: MemoryStats
    delta: MemoryStatsDelta
    lifecycle: MemoryLifecycleDelta | None

    @property
    def pool_id(self) -> PoolId:
        value = self.candidate_pool_id or self.reference_pool_id
        assert value is not None
        return value

    @property
    def changed(self) -> bool:
        return self.delta.changed or bool(self.lifecycle and self.lifecycle.changed)

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_pool_id": _pool_value(self.reference_pool_id),
            "candidate_pool_id": _pool_value(self.candidate_pool_id),
            "match": self.match,
            "reference": self.reference.to_dict(),
            "candidate": self.candidate.to_dict(),
            "delta": self.delta.to_dict(),
            "lifecycle": self.lifecycle.to_dict() if self.lifecycle else None,
        }

    def to_row(self) -> dict[str, object]:
        return _comparison_row(
            {
                "reference_pool_id": _pool_label(self.reference_pool_id),
                "candidate_pool_id": _pool_label(self.candidate_pool_id),
                "match": self.match,
            },
            self.reference,
            self.candidate,
            self.delta,
            self.lifecycle,
        )


@dataclass(frozen=True)
class MemoryObservationComparison:
    """One pool/stream reference/candidate comparison."""

    reference_pool_id: PoolId | None
    reference_stream: Any | None
    candidate_pool_id: PoolId | None
    candidate_stream: Any | None
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
            "reference_pool_id": _pool_value(self.reference_pool_id),
            "reference_stream": self.reference_stream,
            "candidate_pool_id": _pool_value(self.candidate_pool_id),
            "candidate_stream": self.candidate_stream,
            "match": self.match,
            "reference": self.reference.to_dict(),
            "candidate": self.candidate.to_dict(),
            "delta": self.delta.to_dict(),
            "lifecycle": self.lifecycle.to_dict() if self.lifecycle else None,
        }

    def to_row(self) -> dict[str, object]:
        return _comparison_row(
            {
                "reference_pool_id": _pool_label(self.reference_pool_id),
                "reference_stream": _stream_label(self.reference_stream),
                "candidate_pool_id": _pool_label(self.candidate_pool_id),
                "candidate_stream": _stream_label(self.candidate_stream),
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


def _pool_value(pool_id: PoolId | None) -> list[Any] | None:
    return list(pool_id) if pool_id is not None else None


def _pool_label(pool_id: PoolId | None) -> str:
    return pool_id_label(pool_id) if pool_id is not None else ""


def _stream_label(stream: Any | None) -> str:
    return stream_label(stream) if stream is not None else ""
