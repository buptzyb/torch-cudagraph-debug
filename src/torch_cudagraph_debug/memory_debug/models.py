"""Immutable result models shared by memory comparisons and reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from ._identity import PoolId, pool_id_label, stream_label

AllocatorScope = Literal["all", "default", "private"]
MatchKind = Literal[
    "same_run",
    "default",
    "mapped",
    "before_only",
    "after_only",
]


@dataclass(frozen=True)
class MemoryStats:
    """Absolute allocator state at one point."""

    reserved_bytes: int = 0
    allocated_bytes: int = 0
    active_bytes: int = 0
    requested_bytes: int = 0
    segment_count: int = 0
    block_count: int = 0
    largest_inactive_block_bytes: int = 0

    @property
    def inactive_bytes(self) -> int:
        return max(self.reserved_bytes - self.active_bytes, 0)

    @property
    def fragmentation_bytes(self) -> int:
        return max(self.active_bytes - self.requested_bytes, 0)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryStats":
        return cls(
            reserved_bytes=int(value["reserved_bytes"]),
            allocated_bytes=int(value["allocated_bytes"]),
            active_bytes=int(value["active_bytes"]),
            requested_bytes=int(value["requested_bytes"]),
            segment_count=int(value["segment_count"]),
            block_count=int(value["block_count"]),
            largest_inactive_block_bytes=int(
                value["largest_inactive_block_bytes"]
            ),
        )

    @classmethod
    def combine(cls, values: Iterable["MemoryStats"]) -> "MemoryStats":
        items = tuple(values)
        return cls(
            reserved_bytes=sum(item.reserved_bytes for item in items),
            allocated_bytes=sum(item.allocated_bytes for item in items),
            active_bytes=sum(item.active_bytes for item in items),
            requested_bytes=sum(item.requested_bytes for item in items),
            segment_count=sum(item.segment_count for item in items),
            block_count=sum(item.block_count for item in items),
            largest_inactive_block_bytes=max(
                (item.largest_inactive_block_bytes for item in items), default=0
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "reserved_bytes": self.reserved_bytes,
            "allocated_bytes": self.allocated_bytes,
            "active_bytes": self.active_bytes,
            "inactive_bytes": self.inactive_bytes,
            "requested_bytes": self.requested_bytes,
            "segment_count": self.segment_count,
            "block_count": self.block_count,
            "largest_inactive_block_bytes": self.largest_inactive_block_bytes,
            "fragmentation_bytes": self.fragmentation_bytes,
        }


@dataclass(frozen=True)
class MemoryStatsDelta:
    """Signed allocator-state change from before to after."""

    reserved_bytes: int
    allocated_bytes: int
    active_bytes: int
    inactive_bytes: int
    requested_bytes: int
    segment_count: int
    block_count: int
    largest_inactive_block_bytes: int
    fragmentation_bytes: int

    @classmethod
    def between(cls, before: MemoryStats, after: MemoryStats) -> "MemoryStatsDelta":
        return cls(
            **{
                field: int(getattr(after, field)) - int(getattr(before, field))
                for field in before.to_dict()
            }
        )

    @property
    def changed(self) -> bool:
        return any(asdict(self).values())

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class AllocatorScopeComparison:
    """Before/after allocator totals for one logical pool scope."""

    scope: AllocatorScope
    before: MemoryStats
    after: MemoryStats
    delta: MemoryStatsDelta

    @property
    def changed(self) -> bool:
        return self.delta.changed

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "delta": self.delta.to_dict(),
        }

    def to_row(self) -> dict[str, object]:
        return _comparison_row(
            {"scope": self.scope},
            self.before,
            self.after,
            self.delta,
            None,
        )


@dataclass(frozen=True)
class LifecycleDelta:
    """Address-based observations available only within one run."""

    new_segment_bytes: int = 0
    removed_segment_bytes: int = 0
    newly_active_bytes: int = 0
    released_bytes: int = 0

    @classmethod
    def combine(cls, values: Iterable["LifecycleDelta"]) -> "LifecycleDelta":
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
class PoolComparison:
    """One pool-level before/after comparison."""

    before_pool_id: PoolId | None
    after_pool_id: PoolId | None
    match: MatchKind
    before: MemoryStats
    after: MemoryStats
    delta: MemoryStatsDelta
    lifecycle: LifecycleDelta | None

    @property
    def pool_id(self) -> PoolId:
        value = self.after_pool_id or self.before_pool_id
        assert value is not None
        return value

    @property
    def changed(self) -> bool:
        return self.delta.changed or bool(self.lifecycle and self.lifecycle.changed)

    def to_dict(self) -> dict[str, object]:
        return {
            "before_pool_id": _pool_value(self.before_pool_id),
            "after_pool_id": _pool_value(self.after_pool_id),
            "match": self.match,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "delta": self.delta.to_dict(),
            "lifecycle": self.lifecycle.to_dict() if self.lifecycle else None,
        }

    def to_row(self) -> dict[str, object]:
        return _comparison_row(
            {
                "before_pool_id": _pool_label(self.before_pool_id),
                "after_pool_id": _pool_label(self.after_pool_id),
                "match": self.match,
            },
            self.before,
            self.after,
            self.delta,
            self.lifecycle,
        )


@dataclass(frozen=True)
class PoolStreamComparison:
    """One pool/stream before/after comparison."""

    before_pool_id: PoolId | None
    before_stream: Any | None
    after_pool_id: PoolId | None
    after_stream: Any | None
    match: MatchKind
    before: MemoryStats
    after: MemoryStats
    delta: MemoryStatsDelta
    lifecycle: LifecycleDelta | None

    @property
    def changed(self) -> bool:
        return self.delta.changed or bool(self.lifecycle and self.lifecycle.changed)

    def to_dict(self) -> dict[str, object]:
        return {
            "before_pool_id": _pool_value(self.before_pool_id),
            "before_stream": self.before_stream,
            "after_pool_id": _pool_value(self.after_pool_id),
            "after_stream": self.after_stream,
            "match": self.match,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "delta": self.delta.to_dict(),
            "lifecycle": self.lifecycle.to_dict() if self.lifecycle else None,
        }

    def to_row(self) -> dict[str, object]:
        return _comparison_row(
            {
                "before_pool_id": _pool_label(self.before_pool_id),
                "before_stream": _stream_label(self.before_stream),
                "after_pool_id": _pool_label(self.after_pool_id),
                "after_stream": _stream_label(self.after_stream),
                "match": self.match,
            },
            self.before,
            self.after,
            self.delta,
            self.lifecycle,
        )


@dataclass(frozen=True)
class PointPoolState:
    """One absolute pool state in a timeline."""

    point_index: int
    point_label: str
    pool_id: PoolId
    stats: MemoryStats
    delta: MemoryStatsDelta

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "pool_id": list(self.pool_id),
            "state": self.stats.to_dict(),
            "delta": self.delta.to_dict(),
        }

    def to_row(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "pool_id": pool_id_label(self.pool_id),
            **{f"state_{key}": value for key, value in self.stats.to_dict().items()},
            **{f"delta_{key}": value for key, value in self.delta.to_dict().items()},
        }


@dataclass(frozen=True)
class PointAllocatorScopeState:
    """One allocator-scope state in a timeline."""

    point_index: int
    point_label: str
    scope: AllocatorScope
    stats: MemoryStats
    delta: MemoryStatsDelta

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "scope": self.scope,
            "state": self.stats.to_dict(),
            "delta": self.delta.to_dict(),
        }

    def to_row(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "scope": self.scope,
            **{f"state_{key}": value for key, value in self.stats.to_dict().items()},
            **{f"delta_{key}": value for key, value in self.delta.to_dict().items()},
        }


@dataclass(frozen=True)
class PointPoolStreamState:
    """One absolute pool/stream state in a timeline."""

    point_index: int
    point_label: str
    pool_id: PoolId
    stream: Any
    stats: MemoryStats
    delta: MemoryStatsDelta

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "pool_id": list(self.pool_id),
            "stream": self.stream,
            "state": self.stats.to_dict(),
            "delta": self.delta.to_dict(),
        }

    def to_row(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "pool_id": pool_id_label(self.pool_id),
            "stream": stream_label(self.stream),
            **{f"state_{key}": value for key, value in self.stats.to_dict().items()},
            **{f"delta_{key}": value for key, value in self.delta.to_dict().items()},
        }


def _comparison_row(
    identity: dict[str, object],
    before: MemoryStats,
    after: MemoryStats,
    delta: MemoryStatsDelta,
    lifecycle: LifecycleDelta | None,
) -> dict[str, object]:
    row = dict(identity)
    row.update({f"before_{key}": value for key, value in before.to_dict().items()})
    row.update({f"after_{key}": value for key, value in after.to_dict().items()})
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
