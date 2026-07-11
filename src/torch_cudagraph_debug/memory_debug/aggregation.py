"""Allocator-wide totals derived from device-aware pool summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from ._pool_identity import DEFAULT_POOL_ID, MemoryObservationKey, MemoryPoolKey
from .comparison_models import (
    DeviceMatchKind,
    MemoryAllocatorScopeComparison,
    MemoryDeviceComparison,
)
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
    *,
    device_pairs: Sequence[tuple[int | None, int | None, DeviceMatchKind]]
    | None = None,
    default_match: DeviceMatchKind = "same_index",
) -> tuple[MemoryDeviceComparison, ...]:
    """Compare paired device nodes with CUDA samples and allocator rollups."""

    reference_allocator = summarize_devices(reference_pools)
    candidate_allocator = summarize_devices(candidate_pools)
    reference_devices = {*reference_device_memory, *reference_allocator}
    candidate_devices = {*candidate_device_memory, *candidate_allocator}
    if device_pairs is None:
        common = sorted(reference_devices & candidate_devices)
        pairs: tuple[tuple[int | None, int | None, DeviceMatchKind], ...] = (
            *((device, device, default_match) for device in common),
            *(
                (device, None, "reference_only")
                for device in sorted(reference_devices - set(common))
            ),
            *(
                (None, device, "candidate_only")
                for device in sorted(candidate_devices - set(common))
            ),
        )
    else:
        pairs = tuple(device_pairs)
    empty = MemoryStats()
    return tuple(
        MemoryDeviceComparison(
            reference_device_index=reference_device,
            candidate_device_index=candidate_device,
            match=match,
            reference=(
                reference_device_memory.get(reference_device)
                if reference_device is not None
                else None
            ),
            candidate=(
                candidate_device_memory.get(candidate_device)
                if candidate_device is not None
                else None
            ),
            reference_allocator=(
                reference_allocator.get(reference_device, empty)
                if reference_device is not None
                else empty
            ),
            candidate_allocator=(
                candidate_allocator.get(candidate_device, empty)
                if candidate_device is not None
                else empty
            ),
        )
        for reference_device, candidate_device, match in pairs
    )


def compare_allocator_scopes(
    reference: Mapping[AllocatorScope, MemoryStats],
    candidate: Mapping[AllocatorScope, MemoryStats],
) -> tuple[MemoryAllocatorScopeComparison, ...]:
    """Compare the all/default/private scope totals with signed deltas."""

    return tuple(
        MemoryAllocatorScopeComparison(
            scope=scope,
            reference=reference[scope],
            candidate=candidate[scope],
            delta=MemoryStatsDelta.between(reference[scope], candidate[scope]),
        )
        for scope in ALLOCATOR_SCOPES
    )
