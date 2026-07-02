"""Allocator pool and stream identity helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

PoolId = tuple[Any, ...]
DEFAULT_POOL_ID: PoolId = (0, 0)
UNKNOWN_STREAM = "unknown"


def normalize_pool_id(pool_id: Any) -> PoolId:
    if pool_id is None:
        return DEFAULT_POOL_ID
    if isinstance(pool_id, tuple):
        return pool_id
    if isinstance(pool_id, list):
        return tuple(pool_id)
    return (pool_id,)


def normalize_stream(stream: Any) -> Any:
    return UNKNOWN_STREAM if stream is None else stream


def pool_id_label(pool_id: Sequence[Any] | Any) -> str:
    normalized = normalize_pool_id(pool_id)
    return "pool[" + ",".join(str(part) for part in normalized) + "]"


def stream_label(stream: Any) -> str:
    return f"stream[{stream}]"
