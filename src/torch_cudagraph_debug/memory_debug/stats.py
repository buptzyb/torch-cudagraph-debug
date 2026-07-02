"""Immutable result models shared by memory comparisons and reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

AllocatorScope = Literal["all", "default", "private"]


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
            largest_inactive_block_bytes=int(value["largest_inactive_block_bytes"]),
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
    """Signed allocator-state change from reference to candidate."""

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
