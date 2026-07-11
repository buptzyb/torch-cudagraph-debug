"""Memory timeline state models and construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from ._pool_identity import MemoryObservationKey, MemoryPoolKey
from .aggregation import ALLOCATOR_SCOPES, summarize_devices
from .attribution import MemoryAttributionOptions
from .comparison import (
    _compare_same_run_views,
    _load_interval_views,
    _MemoryStateView,
)
from .events import EventWindow
from .lifetimes import analyze_allocation_lifetimes
from .recording import MemoryRun
from .reports import MemoryPointComparison, MemoryTimeline
from .stats import (
    AllocatorScope,
    DeviceMemorySample,
    MemoryStats,
    MemoryStatsDelta,
)


@dataclass(frozen=True)
class MemoryPoolTimelineEntry:
    """One absolute device/pool state in a timeline."""

    point_index: int
    point_label: str
    key: MemoryPoolKey
    stats: MemoryStats
    delta: MemoryStatsDelta | None

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "key": {
                "device_index": self.key.device_index,
                "pool_id": list(self.key.pool_id),
            },
            "state": self.stats.to_dict(),
            "delta": self.delta.to_dict() if self.delta is not None else None,
        }

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "key": self.key.label,
            **{f"state_{key}": value for key, value in self.stats.to_dict().items()},
        }
        row.update(
            {
                f"delta_{key}": value
                for key, value in (
                    self.delta.to_dict() if self.delta is not None else {}
                ).items()
            }
        )
        return row


@dataclass(frozen=True)
class MemoryAllocatorScopeTimelineEntry:
    """One allocator-scope state in a timeline."""

    point_index: int
    point_label: str
    scope: AllocatorScope
    stats: MemoryStats
    delta: MemoryStatsDelta | None

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "scope": self.scope,
            "state": self.stats.to_dict(),
            "delta": self.delta.to_dict() if self.delta is not None else None,
        }

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "scope": self.scope,
            **{f"state_{key}": value for key, value in self.stats.to_dict().items()},
        }
        row.update(
            {
                f"delta_{key}": value
                for key, value in (
                    self.delta.to_dict() if self.delta is not None else {}
                ).items()
            }
        )
        return row


@dataclass(frozen=True)
class MemoryObservationTimelineEntry:
    """One absolute device/pool/stream state in a timeline."""

    point_index: int
    point_label: str
    key: MemoryObservationKey
    stats: MemoryStats
    delta: MemoryStatsDelta | None

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "key": {
                "device_index": self.key.device_index,
                "pool_id": list(self.key.pool_id),
                "stream": self.key.stream,
            },
            "state": self.stats.to_dict(),
            "delta": self.delta.to_dict() if self.delta is not None else None,
        }

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "key": self.key.label,
            **{f"state_{key}": value for key, value in self.stats.to_dict().items()},
        }
        row.update(
            {
                f"delta_{key}": value
                for key, value in (
                    self.delta.to_dict() if self.delta is not None else {}
                ).items()
            }
        )
        return row


@dataclass(frozen=True)
class MemoryDeviceTimelineEntry:
    """One device node state in a timeline: the optional device-wide CUDA
    Runtime sample plus this process's allocator rollup for that device.

    The sample comes from ``torch.cuda.mem_get_info`` and covers the whole
    device, including the CUDA context and every other process on the GPU;
    it is ``None`` when the point did not sample this device. Sample deltas
    compare against the previous point and are ``None`` together when either
    endpoint lacks the sample: a missing sample means unknown, never zero.
    The allocator rollup is always present (all-zero without pools); its
    delta is ``None`` only at the first point.
    """

    point_index: int
    point_label: str
    device_index: int
    sample: DeviceMemorySample | None
    stats: MemoryStats
    delta: MemoryStatsDelta | None
    delta_used_bytes: int | None
    delta_free_bytes: int | None
    delta_total_bytes: int | None
    delta_cuda_allocator_residual_bytes: int | None

    @property
    def allocator_reserved_bytes(self) -> int:
        return self.stats.reserved_bytes

    @property
    def delta_allocator_reserved_bytes(self) -> int | None:
        return None if self.delta is None else self.delta.reserved_bytes

    @property
    def cuda_allocator_residual_bytes(self) -> int | None:
        if self.sample is None:
            return None
        return self.sample.used_bytes - self.stats.reserved_bytes

    def to_dict(self) -> dict[str, object]:
        sample_deltas = (
            None
            if self.delta_used_bytes is None
            else {
                "used_bytes": self.delta_used_bytes,
                "free_bytes": self.delta_free_bytes,
                "total_bytes": self.delta_total_bytes,
                "cuda_allocator_residual_bytes": self.delta_cuda_allocator_residual_bytes,
            }
        )
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "device_index": self.device_index,
            "sample": self.sample.to_dict() if self.sample else None,
            "cuda_allocator_residual_bytes": self.cuda_allocator_residual_bytes,
            "allocator_state": self.stats.to_dict(),
            "allocator_delta": self.delta.to_dict() if self.delta else None,
            "sample_delta": sample_deltas,
        }

    def to_row(self) -> dict[str, object]:
        sample = self.sample
        delta_dict = self.delta.to_dict() if self.delta else None
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "device_index": self.device_index,
            "state_free_bytes": sample.free_bytes if sample else None,
            "state_total_bytes": sample.total_bytes if sample else None,
            "state_used_bytes": sample.used_bytes if sample else None,
            "state_cuda_allocator_residual_bytes": self.cuda_allocator_residual_bytes,
            **{
                f"state_allocator_{key}": value
                for key, value in self.stats.to_dict().items()
            },
            "delta_used_bytes": self.delta_used_bytes,
            "delta_free_bytes": self.delta_free_bytes,
            "delta_total_bytes": self.delta_total_bytes,
            "delta_cuda_allocator_residual_bytes": self.delta_cuda_allocator_residual_bytes,
            **{
                f"delta_allocator_{key}": (
                    None if delta_dict is None else delta_dict[key]
                )
                for key in self.stats.to_dict()
            },
        }


def _build_timeline(
    run: MemoryRun, options: MemoryAttributionOptions
) -> MemoryTimeline:
    allocator_scope_entries: list[MemoryAllocatorScopeTimelineEntry] = []
    pool_entries: list[MemoryPoolTimelineEntry] = []
    observation_entries: list[MemoryObservationTimelineEntry] = []
    device_entries: list[MemoryDeviceTimelineEntry] = []
    previous_pool_stats: Mapping[MemoryPoolKey, MemoryStats] | None = None
    previous_allocator_scope_stats: Mapping[AllocatorScope, MemoryStats] | None = None
    previous_observations: Mapping[MemoryObservationKey, MemoryStats] | None = None
    previous_device_memory: Mapping[int, DeviceMemorySample] | None = None
    previous_device_stats: dict[int, MemoryStats] | None = None
    for point in run.points:
        current_pool_stats = point.pool_stats
        current_allocator_scope_stats = point.allocator_scope_stats
        for scope in ALLOCATOR_SCOPES:
            stats = current_allocator_scope_stats[scope]
            allocator_scope_entries.append(
                MemoryAllocatorScopeTimelineEntry(
                    point_index=point.index,
                    point_label=point.label,
                    scope=scope,
                    stats=stats,
                    delta=(
                        None
                        if previous_allocator_scope_stats is None
                        else MemoryStatsDelta.between(
                            previous_allocator_scope_stats[scope], stats
                        )
                    ),
                )
            )
        pool_keys = set(current_pool_stats)
        if previous_pool_stats is not None:
            pool_keys.update(previous_pool_stats)
        for pool_key in sorted(pool_keys, key=lambda item: item.label):
            stats = current_pool_stats.get(pool_key, MemoryStats())
            pool_entries.append(
                MemoryPoolTimelineEntry(
                    point_index=point.index,
                    point_label=point.label,
                    key=pool_key,
                    stats=stats,
                    delta=(
                        None
                        if previous_pool_stats is None
                        else MemoryStatsDelta.between(
                            previous_pool_stats.get(pool_key, MemoryStats()), stats
                        )
                    ),
                )
            )
        observation_keys = set(point.observation_stats)
        if previous_observations is not None:
            observation_keys.update(previous_observations)
        for key in sorted(observation_keys, key=lambda item: item.label):
            stats = point.observation_stats.get(key, MemoryStats())
            observation_entries.append(
                MemoryObservationTimelineEntry(
                    point_index=point.index,
                    point_label=point.label,
                    key=key,
                    stats=stats,
                    delta=(
                        None
                        if previous_observations is None
                        else MemoryStatsDelta.between(
                            previous_observations.get(key, MemoryStats()), stats
                        )
                    ),
                )
            )
        current_device_stats = summarize_devices(current_pool_stats)
        empty = MemoryStats()
        device_indices = sorted(
            {
                *point.device_memory,
                *current_device_stats,
                *(previous_device_memory or ()),
                *(previous_device_stats if previous_device_stats is not None else ()),
            }
        )
        for device in device_indices:
            sample = point.device_memory.get(device)
            stats = current_device_stats.get(device, empty)
            stats_delta = (
                None
                if previous_device_stats is None
                else MemoryStatsDelta.between(
                    previous_device_stats.get(device, empty), stats
                )
            )
            previous_sample = (
                None
                if previous_device_memory is None
                else previous_device_memory.get(device)
            )
            if sample is None or previous_sample is None:
                delta_used = delta_free = delta_total = None
                delta_cuda_allocator_residual_bytes = None
            else:
                earlier_reserved = previous_device_stats.get(
                    device, empty
                ).reserved_bytes
                delta_used = sample.used_bytes - previous_sample.used_bytes
                delta_free = sample.free_bytes - previous_sample.free_bytes
                delta_total = sample.total_bytes - previous_sample.total_bytes
                delta_cuda_allocator_residual_bytes = (
                    sample.used_bytes - stats.reserved_bytes
                ) - (previous_sample.used_bytes - earlier_reserved)
            device_entries.append(
                MemoryDeviceTimelineEntry(
                    point_index=point.index,
                    point_label=point.label,
                    device_index=device,
                    sample=sample,
                    stats=stats,
                    delta=stats_delta,
                    delta_used_bytes=delta_used,
                    delta_free_bytes=delta_free,
                    delta_total_bytes=delta_total,
                    delta_cuda_allocator_residual_bytes=delta_cuda_allocator_residual_bytes,
                )
            )
        previous_pool_stats = current_pool_stats
        previous_allocator_scope_stats = current_allocator_scope_stats
        previous_observations = point.observation_stats
        previous_device_memory = point.device_memory
        previous_device_stats = current_device_stats

    allocation_lifetimes = None
    comparison_options = replace(options, lifetimes=False)
    interval_views: tuple[_MemoryStateView, ...] = ()
    if (options.stacks or options.events or options.lifetimes) and run.points:
        interval_views = _load_interval_views(run.points, comparison_options)
    shared_event_windows = (
        tuple(point._event_windows() for point in run.points[1:])
        if options.events and options.lifetimes
        else None
    )
    if options.lifetimes and run.points:
        allocation_lifetimes = analyze_allocation_lifetimes(
            run,
            start=run.points[0],
            end=run.points[-1],
            active_at=None,
            born_between=None,
            options=options.lifetime_options(),
            _allocator_states=tuple(view.raw for view in interval_views),
            _event_windows=shared_event_windows,
        )
    point_comparisons = (
        _build_point_comparisons(
            run,
            comparison_options,
            interval_views,
            event_windows=shared_event_windows,
        )
        if options.stacks or options.events
        else ()
    )
    return MemoryTimeline(
        run=run,
        allocator_scope_entries=tuple(allocator_scope_entries),
        pool_entries=tuple(pool_entries),
        observation_entries=tuple(observation_entries),
        device_entries=tuple(device_entries),
        point_comparisons=point_comparisons,
        allocation_lifetimes=allocation_lifetimes,
        display_stack_depth=options.display.stack_depth,
        display_limit=options.display.limit,
    )


def _build_point_comparisons(
    run: MemoryRun,
    options: MemoryAttributionOptions,
    interval_views: tuple[_MemoryStateView, ...],
    *,
    event_windows: Sequence[Sequence[EventWindow]] | None = None,
) -> tuple[MemoryPointComparison, ...]:
    if len(run.points) < 2:
        return ()
    rows = []
    if len(interval_views) != len(run.points):
        raise ValueError("timeline interval views must match run points")
    if event_windows is not None and len(event_windows) != len(run.points) - 1:
        raise ValueError("timeline event windows must match run intervals")
    reference_view = interval_views[0]
    for interval_index, candidate_view in enumerate(interval_views[1:]):
        rows.append(
            _compare_same_run_views(
                run,
                reference_view,
                candidate_view,
                options,
                interval_views=(reference_view, candidate_view),
                _event_windows=(
                    (event_windows[interval_index],)
                    if event_windows is not None
                    else None
                ),
            )
        )
        reference_view = candidate_view
    return tuple(rows)
