"""Allocator-wide totals derived from pool summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .models import (
    AllocatorScope,
    AllocatorScopeComparison,
    MemoryStats,
    MemoryStatsDelta,
)
from .summary import DEFAULT_POOL_ID, GroupKey


ALLOCATOR_SCOPES: tuple[AllocatorScope, ...] = ("all", "default", "private")


def summarize_pools(
    groups: Mapping[GroupKey, MemoryStats],
) -> dict[tuple[object, ...], MemoryStats]:
    """Aggregate pool/stream states into pool states."""

    grouped: defaultdict[tuple[object, ...], list[MemoryStats]] = defaultdict(list)
    for key, stats in groups.items():
        grouped[key.pool_id].append(stats)
    return {pool_id: MemoryStats.combine(values) for pool_id, values in grouped.items()}


def summarize_allocator_scopes(
    pools: Mapping[tuple[object, ...], MemoryStats],
) -> dict[AllocatorScope, MemoryStats]:
    """Return all/default/private totals without inventing synthetic pool IDs."""

    default = [value for pool_id, value in pools.items() if pool_id == DEFAULT_POOL_ID]
    private = [value for pool_id, value in pools.items() if pool_id != DEFAULT_POOL_ID]
    return {
        "all": MemoryStats.combine(pools.values()),
        "default": MemoryStats.combine(default),
        "private": MemoryStats.combine(private),
    }


def compare_allocator_scopes(
    before: Mapping[AllocatorScope, MemoryStats],
    after: Mapping[AllocatorScope, MemoryStats],
) -> tuple[AllocatorScopeComparison, ...]:
    return tuple(
        AllocatorScopeComparison(
            scope=scope,
            before=before[scope],
            after=after[scope],
            delta=MemoryStatsDelta.between(before[scope], after[scope]),
        )
        for scope in ALLOCATOR_SCOPES
    )
