"""Allocator device, pool, and stream identity models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

PoolId: TypeAlias = tuple[int, int]
StreamId: TypeAlias = int | None
DEFAULT_POOL_ID: PoolId = (0, 0)


def normalize_device_index(device_index: Any) -> int:
    if type(device_index) is not int or device_index < 0:
        raise ValueError("device index must be a non-negative integer")
    return device_index


def normalize_pool_id(pool_id: Any) -> PoolId:
    if pool_id is None:
        return DEFAULT_POOL_ID
    if not isinstance(pool_id, (list, tuple)) or len(pool_id) != 2:
        raise ValueError("pool ID must contain exactly two integers")
    first, second = pool_id
    if type(first) is not int or type(second) is not int:
        raise TypeError("pool ID components must be integers")
    if first < 0 or second < 0:
        raise ValueError("pool ID components must be non-negative")
    return (first, second)


def normalize_stream(stream: Any) -> StreamId:
    if stream is None:
        return None
    if type(stream) is not int:
        raise TypeError("stream must be an integer or None")
    if stream < 0:
        raise ValueError("stream must be non-negative")
    return stream


@dataclass(frozen=True, slots=True)
class MemoryPoolKey:
    """Stable identity for one allocator pool on one CUDA device."""

    device_index: int
    pool_id: PoolId

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "device_index", normalize_device_index(self.device_index)
        )
        object.__setattr__(self, "pool_id", normalize_pool_id(self.pool_id))

    @property
    def label(self) -> str:
        return f"device[{self.device_index}]/{pool_id_label(self.pool_id)}"


@dataclass(frozen=True, slots=True)
class MemoryObservationKey:
    """Stable identity for one device, allocator pool, and stream scope."""

    device_index: int
    pool_id: PoolId
    stream: StreamId

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "device_index", normalize_device_index(self.device_index)
        )
        object.__setattr__(self, "pool_id", normalize_pool_id(self.pool_id))
        object.__setattr__(self, "stream", normalize_stream(self.stream))

    @property
    def pool_key(self) -> MemoryPoolKey:
        return MemoryPoolKey(self.device_index, self.pool_id)

    @property
    def label(self) -> str:
        return f"{self.pool_key.label}/{stream_label(self.stream)}"


def pool_id_label(pool_id: PoolId | Any) -> str:
    normalized = normalize_pool_id(pool_id)
    return "pool[" + ",".join(str(part) for part in normalized) + "]"


def stream_label(stream: StreamId | Any) -> str:
    normalized = normalize_stream(stream)
    return "stream[unknown]" if normalized is None else f"stream[{normalized}]"
