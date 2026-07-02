"""Indexed allocator segment-range lookup."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .allocator_snapshot import normalize_pool_id


@dataclass(frozen=True)
class _PoolRange:
    start: int
    end: int
    pool_id: tuple[Any, ...]
    ordinal: int


@dataclass(frozen=True)
class _PoolRangeLayer:
    starts_by_device: Mapping[int | None, tuple[int, ...]]
    ranges_by_device: Mapping[int | None, tuple[_PoolRange, ...]]

    def find(self, device: int | None, address: int) -> tuple[Any, ...] | None:
        matches: list[_PoolRange] = []
        candidate_devices = (device,) if device is None else (device, None)
        for candidate_device in candidate_devices:
            starts = self.starts_by_device.get(candidate_device)
            ranges = self.ranges_by_device.get(candidate_device)
            if not starts or not ranges:
                continue
            index = bisect_right(starts, address) - 1
            if index < 0:
                continue
            candidate = ranges[index]
            if address < candidate.end:
                matches.append(candidate)
        if not matches:
            return None
        return min(matches, key=lambda item: item.ordinal).pool_id


@dataclass(frozen=True)
class PoolRangeIndex:
    """Address-to-pool index preserving segment-set preference."""

    layers: tuple[_PoolRangeLayer, ...]

    def find(self, device: int | None, address: int) -> tuple[Any, ...] | None:
        for layer in self.layers:
            pool_id = layer.find(device, address)
            if pool_id is not None:
                return pool_id
        return None


def build_pool_range_index(
    *segment_sets: Sequence[Mapping[str, Any]],
) -> PoolRangeIndex:
    """Index non-overlapping allocator segments separately by snapshot input."""

    layers: list[_PoolRangeLayer] = []
    for segments in segment_sets:
        ranges_by_device: defaultdict[int | None, list[_PoolRange]] = defaultdict(list)
        for ordinal, segment in enumerate(segments):
            address = segment.get("address")
            size = int(segment.get("total_size", 0) or 0)
            if address is None or size <= 0:
                continue
            raw_device = segment.get("device")
            device = int(raw_device) if raw_device is not None else None
            start = int(address)
            ranges_by_device[device].append(
                _PoolRange(
                    start=start,
                    end=start + size,
                    pool_id=normalize_pool_id(segment.get("segment_pool_id")),
                    ordinal=ordinal,
                )
            )
        ordered_ranges = {
            device: tuple(sorted(items, key=lambda item: (item.start, item.ordinal)))
            for device, items in ranges_by_device.items()
        }
        layers.append(
            _PoolRangeLayer(
                starts_by_device={
                    device: tuple(item.start for item in items)
                    for device, items in ordered_ranges.items()
                },
                ranges_by_device=ordered_ranges,
            )
        )
    return PoolRangeIndex(layers=tuple(layers))
