"""Allocator-wide totals derived from pool summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .comparison_models import MemoryAllocatorScopeComparison
from .stats import AllocatorScope, MemoryStats, MemoryStatsDelta
from .allocator_snapshot import DEFAULT_POOL_ID, MemoryObservationKey

ALLOCATOR_SCOPES: tuple[AllocatorScope, ...] = ("all", "default", "private")


def summarize_pools(
    observations: Mapping[MemoryObservationKey, MemoryStats],
) -> dict[tuple[object, ...], MemoryStats]:
    """Aggregate pool/stream states into pool states."""

    by_pool: defaultdict[tuple[object, ...], list[MemoryStats]] = defaultdict(list)
    for key, stats in observations.items():
        by_pool[key.pool_id].append(stats)
    return {pool_id: MemoryStats.combine(values) for pool_id, values in by_pool.items()}


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
