"""Allocator-wide totals derived from device-aware pool summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from ._pool_identity import DEFAULT_POOL_ID, MemoryObservationKey, MemoryPoolKey
from .comparison_models import MemoryAllocatorScopeComparison
from .stats import AllocatorScope, MemoryStats, MemoryStatsDelta

ALLOCATOR_SCOPES: tuple[AllocatorScope, ...] = ("all", "default", "private")


def summarize_pools(
    observations: Mapping[MemoryObservationKey, MemoryStats],
) -> dict[MemoryPoolKey, MemoryStats]:
    """Aggregate device/pool/stream states into device/pool states."""

    by_pool: defaultdict[MemoryPoolKey, list[MemoryStats]] = defaultdict(list)
    for key, stats in observations.items():
        by_pool[key.pool_key].append(stats)
    return {
        pool_key: MemoryStats.combine(values) for pool_key, values in by_pool.items()
    }


def summarize_allocator_scopes(
    pools: Mapping[MemoryPoolKey, MemoryStats],
) -> dict[AllocatorScope, MemoryStats]:
    """Return allocator totals across the selected devices."""

    default = [value for key, value in pools.items() if key.pool_id == DEFAULT_POOL_ID]
    private = [value for key, value in pools.items() if key.pool_id != DEFAULT_POOL_ID]
    return {
        "all": MemoryStats.combine(pools.values()),
        "default": MemoryStats.combine(default),
        "private": MemoryStats.combine(private),
    }


def compare_allocator_scopes(
    reference: Mapping[AllocatorScope, MemoryStats],
    candidate: Mapping[AllocatorScope, MemoryStats],
) -> tuple[MemoryAllocatorScopeComparison, ...]:
    return tuple(
        MemoryAllocatorScopeComparison(
            scope=scope,
            reference=reference[scope],
            candidate=candidate[scope],
            delta=MemoryStatsDelta.between(reference[scope], candidate[scope]),
        )
        for scope in ALLOCATOR_SCOPES
    )
