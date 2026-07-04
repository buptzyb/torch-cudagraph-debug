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
        normalized = pool_id
    elif isinstance(pool_id, list):
        normalized = tuple(pool_id)
    else:
        normalized = (pool_id,)
    if not normalized:
        raise ValueError("pool ID must be non-empty")
    for part in normalized:
        if isinstance(part, (list, tuple, dict, set)):
            raise TypeError("pool ID components must be hashable scalar values")
        try:
            hash(part)
        except TypeError as exc:
            raise TypeError(
                "pool ID components must be hashable scalar values"
            ) from exc
    return normalized


def normalize_stream(stream: Any) -> Any:
    if stream is None:
        return UNKNOWN_STREAM
    try:
        hash(stream)
    except TypeError as exc:
        raise TypeError("stream must be a hashable scalar value") from exc
    return stream


def pool_id_label(pool_id: Sequence[Any] | Any) -> str:
    normalized = normalize_pool_id(pool_id)
    return "pool[" + ",".join(str(part) for part in normalized) + "]"


def stream_label(stream: Any) -> str:
    return f"stream[{stream}]"
