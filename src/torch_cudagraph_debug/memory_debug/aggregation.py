"""Allocator-wide totals derived from device-aware pool summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from ._pool_identity import DEFAULT_POOL_ID, MemoryObservationKey, MemoryPoolKey
from .comparison_models import MemoryAllocatorScopeComparison, MemoryDeviceComparison
from .stats import (
    AllocatorScope,
    DeviceMemorySample,
    MemoryStats,
    MemoryStatsDelta,
)

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


def summarize_devices(
    pools: Mapping[MemoryPoolKey, MemoryStats],
) -> dict[int, MemoryStats]:
    """Return allocator totals grouped by device index."""

    by_device: defaultdict[int, list[MemoryStats]] = defaultdict(list)
    for key, stats in pools.items():
        by_device[key.device_index].append(stats)
    return {
        device: MemoryStats.combine(values)
        for device, values in sorted(by_device.items())
    }


def compare_device_memory(
    reference_device_memory: Mapping[int, DeviceMemorySample],
    candidate_device_memory: Mapping[int, DeviceMemorySample],
    reference_pools: Mapping[MemoryPoolKey, MemoryStats],
    candidate_pools: Mapping[MemoryPoolKey, MemoryStats],
) -> tuple[MemoryDeviceComparison, ...]:
    """Compare device-wide CUDA Runtime samples against allocator reserved totals.

    Only sampled devices produce rows; a device that has allocator
    observations but no CUDA Runtime sample on either endpoint is omitted.
    """

    devices = sorted({*reference_device_memory, *candidate_device_memory})
    if not devices:
        return ()
    reference_reserved = summarize_devices(reference_pools)
    candidate_reserved = summarize_devices(candidate_pools)
    empty = MemoryStats()
    return tuple(
        MemoryDeviceComparison(
            device_index=device,
            reference=reference_device_memory.get(device),
            candidate=candidate_device_memory.get(device),
            reference_allocator_reserved_bytes=reference_reserved.get(
                device, empty
            ).reserved_bytes,
            candidate_allocator_reserved_bytes=candidate_reserved.get(
                device, empty
            ).reserved_bytes,
        )
        for device in devices
    )


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
