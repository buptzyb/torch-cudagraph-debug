"""Immutable result models shared by memory comparisons and reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

AllocatorScope = Literal["all", "default", "private"]
MemoryStatMetric = Literal[
    "reserved_bytes",
    "allocated_bytes",
    "active_bytes",
    "awaiting_free_bytes",
    "inactive_bytes",
    "requested_bytes",
    "internal_fragmentation_bytes",
    "segment_count",
    "block_count",
    "inactive_block_count",
    "largest_inactive_block_bytes",
    "expandable_segment_count",
    "expandable_reserved_bytes",
]

MEMORY_STAT_METRICS: tuple[MemoryStatMetric, ...] = (
    "reserved_bytes",
    "allocated_bytes",
    "active_bytes",
    "awaiting_free_bytes",
    "inactive_bytes",
    "requested_bytes",
    "internal_fragmentation_bytes",
    "segment_count",
    "block_count",
    "inactive_block_count",
    "largest_inactive_block_bytes",
    "expandable_segment_count",
    "expandable_reserved_bytes",
)


@dataclass(frozen=True)
class MemoryStats:
    """Absolute allocator state at one point."""

    reserved_bytes: int = 0
    allocated_bytes: int = 0
    active_bytes: int = 0
    requested_bytes: int = 0
    segment_count: int = 0
    block_count: int = 0
    inactive_block_count: int = 0
    largest_inactive_block_bytes: int = 0
    expandable_segment_count: int = 0
    expandable_reserved_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "reserved_bytes",
            "allocated_bytes",
            "active_bytes",
            "requested_bytes",
            "segment_count",
            "block_count",
            "inactive_block_count",
            "largest_inactive_block_bytes",
            "expandable_segment_count",
            "expandable_reserved_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.allocated_bytes > self.active_bytes:
            raise ValueError("allocated_bytes must not exceed active_bytes")
        if self.requested_bytes > self.active_bytes:
            raise ValueError("requested_bytes must not exceed active_bytes")
        if self.active_bytes > self.reserved_bytes:
            raise ValueError("active_bytes must not exceed reserved_bytes")
        if self.inactive_block_count > self.block_count:
            raise ValueError("inactive_block_count must not exceed block_count")
        if self.expandable_segment_count > self.segment_count:
            raise ValueError("expandable_segment_count must not exceed segment_count")
        if self.expandable_reserved_bytes > self.reserved_bytes:
            raise ValueError("expandable_reserved_bytes must not exceed reserved_bytes")
        if self.largest_inactive_block_bytes > self.reserved_bytes - self.active_bytes:
            raise ValueError(
                "largest_inactive_block_bytes must not exceed inactive bytes"
            )

    @property
    def awaiting_free_bytes(self) -> int:
        return self.active_bytes - self.allocated_bytes

    @property
    def inactive_bytes(self) -> int:
        return self.reserved_bytes - self.active_bytes

    @property
    def internal_fragmentation_bytes(self) -> int:
        return self.active_bytes - self.requested_bytes

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryStats":
        fields = (
            "reserved_bytes",
            "allocated_bytes",
            "active_bytes",
            "requested_bytes",
            "segment_count",
            "block_count",
            "inactive_block_count",
            "largest_inactive_block_bytes",
            "expandable_segment_count",
            "expandable_reserved_bytes",
        )
        values = {}
        for name in fields:
            raw = value[name]
            if type(raw) is not int or raw < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            values[name] = raw
        return cls(**values)

    @classmethod
    def combine(cls, values: Iterable["MemoryStats"]) -> "MemoryStats":
        """Sum counts and bytes across stats, except
        ``largest_inactive_block_bytes``, which takes the maximum."""

        items = tuple(values)
        return cls(
            reserved_bytes=sum(item.reserved_bytes for item in items),
            allocated_bytes=sum(item.allocated_bytes for item in items),
            active_bytes=sum(item.active_bytes for item in items),
            requested_bytes=sum(item.requested_bytes for item in items),
            segment_count=sum(item.segment_count for item in items),
            block_count=sum(item.block_count for item in items),
            inactive_block_count=sum(item.inactive_block_count for item in items),
            largest_inactive_block_bytes=max(
                (item.largest_inactive_block_bytes for item in items), default=0
            ),
            expandable_segment_count=sum(
                item.expandable_segment_count for item in items
            ),
            expandable_reserved_bytes=sum(
                item.expandable_reserved_bytes for item in items
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "reserved_bytes": self.reserved_bytes,
            "allocated_bytes": self.allocated_bytes,
            "active_bytes": self.active_bytes,
            "awaiting_free_bytes": self.awaiting_free_bytes,
            "inactive_bytes": self.inactive_bytes,
            "requested_bytes": self.requested_bytes,
            "segment_count": self.segment_count,
            "block_count": self.block_count,
            "inactive_block_count": self.inactive_block_count,
            "largest_inactive_block_bytes": self.largest_inactive_block_bytes,
            "expandable_segment_count": self.expandable_segment_count,
            "expandable_reserved_bytes": self.expandable_reserved_bytes,
            "internal_fragmentation_bytes": self.internal_fragmentation_bytes,
        }


@dataclass(frozen=True)
class MemoryStatsDelta:
    """Signed allocator-state change from reference to candidate."""

    reserved_bytes: int
    allocated_bytes: int
    active_bytes: int
    awaiting_free_bytes: int
    inactive_bytes: int
    requested_bytes: int
    segment_count: int
    block_count: int
    inactive_block_count: int
    largest_inactive_block_bytes: int
    expandable_segment_count: int
    expandable_reserved_bytes: int
    internal_fragmentation_bytes: int

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in asdict(self).values()):
            raise TypeError("memory stat deltas must be integers")

    @classmethod
    def between(
        cls, reference: MemoryStats, candidate: MemoryStats
    ) -> "MemoryStatsDelta":
        return cls(
            **{
                field: int(getattr(candidate, field)) - int(getattr(reference, field))
                for field in reference.to_dict()
            }
        )

    @property
    def changed(self) -> bool:
        return any(asdict(self).values())

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
