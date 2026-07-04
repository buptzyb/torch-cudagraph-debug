"""Memory timeline state models and construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ._pool_identity import PoolId, pool_id_label, stream_label
from .aggregation import ALLOCATOR_SCOPES, summarize_allocator_scopes
from .allocator_snapshot import MemoryObservationKey
from .attribution import MemoryAttributionOptions
from .comparison import (
    _MemoryStateView,
    _compare_same_run_views,
    _load_interval_views,
)
from .lifetimes import analyze_allocation_lifetimes
from .recording import MemoryRun
from .reports import MemoryPointComparison, MemoryTimeline
from .stats import AllocatorScope, MemoryStats, MemoryStatsDelta


@dataclass(frozen=True)
class MemoryPoolTimelineEntry:
    """One absolute pool state in a timeline."""

    point_index: int
    point_label: str
    pool_id: PoolId
    stats: MemoryStats
    delta: MemoryStatsDelta

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "pool_id": list(self.pool_id),
            "state": self.stats.to_dict(),
            "delta": self.delta.to_dict(),
        }

    def to_row(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "pool_id": pool_id_label(self.pool_id),
            **{f"state_{key}": value for key, value in self.stats.to_dict().items()},
            **{f"delta_{key}": value for key, value in self.delta.to_dict().items()},
        }


@dataclass(frozen=True)
class MemoryAllocatorScopeTimelineEntry:
    """One allocator-scope state in a timeline."""

    point_index: int
    point_label: str
    scope: AllocatorScope
    stats: MemoryStats
    delta: MemoryStatsDelta

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "scope": self.scope,
            "state": self.stats.to_dict(),
            "delta": self.delta.to_dict(),
        }

    def to_row(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "scope": self.scope,
            **{f"state_{key}": value for key, value in self.stats.to_dict().items()},
            **{f"delta_{key}": value for key, value in self.delta.to_dict().items()},
        }


@dataclass(frozen=True)
class MemoryObservationTimelineEntry:
    """One absolute pool/stream state in a timeline."""

    point_index: int
    point_label: str
    pool_id: PoolId
    stream: Any
    stats: MemoryStats
    delta: MemoryStatsDelta

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "pool_id": list(self.pool_id),
            "stream": self.stream,
            "state": self.stats.to_dict(),
            "delta": self.delta.to_dict(),
        }

    def to_row(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "pool_id": pool_id_label(self.pool_id),
            "stream": stream_label(self.stream),
            **{f"state_{key}": value for key, value in self.stats.to_dict().items()},
            **{f"delta_{key}": value for key, value in self.delta.to_dict().items()},
        }


def _build_timeline(
    run: MemoryRun, options: MemoryAttributionOptions
) -> MemoryTimeline:
    allocator_scope_entries: list[MemoryAllocatorScopeTimelineEntry] = []
    pool_entries: list[MemoryPoolTimelineEntry] = []
    observation_entries: list[MemoryObservationTimelineEntry] = []
    previous_pool_stats: Mapping[PoolId, MemoryStats] = {}
    previous_allocator_scope_stats = summarize_allocator_scopes(previous_pool_stats)
    previous_observations: Mapping[MemoryObservationKey, MemoryStats] = {}
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
                    delta=MemoryStatsDelta.between(
                        previous_allocator_scope_stats[scope], stats
                    ),
                )
            )
        for pool_id in sorted(
            set(previous_pool_stats) | set(current_pool_stats), key=pool_id_label
        ):
            reference_stats = previous_pool_stats.get(pool_id, MemoryStats())
            stats = current_pool_stats.get(pool_id, MemoryStats())
            pool_entries.append(
                MemoryPoolTimelineEntry(
                    point_index=point.index,
                    point_label=point.label,
                    pool_id=pool_id,
                    stats=stats,
                    delta=MemoryStatsDelta.between(reference_stats, stats),
                )
            )
        for key in sorted(
            set(previous_observations) | set(point.observation_stats),
            key=lambda item: (pool_id_label(item.pool_id), stream_label(item.stream)),
        ):
            reference_stats = previous_observations.get(key, MemoryStats())
            stats = point.observation_stats.get(key, MemoryStats())
            observation_entries.append(
                MemoryObservationTimelineEntry(
                    point_index=point.index,
                    point_label=point.label,
                    pool_id=key.pool_id,
                    stream=key.stream,
                    stats=stats,
                    delta=MemoryStatsDelta.between(reference_stats, stats),
                )
            )
        previous_pool_stats = current_pool_stats
        previous_allocator_scope_stats = current_allocator_scope_stats
        previous_observations = point.observation_stats

    allocation_lifetimes = None
    comparison_options = replace(options, lifetimes=False)
    interval_views: tuple[_MemoryStateView, ...] = ()
    if (options.stacks or options.events or options.lifetimes) and run.points:
        interval_views = _load_interval_views(run.points, comparison_options)
    if options.lifetimes and run.points:
        allocation_lifetimes = analyze_allocation_lifetimes(
            run,
            start=run.points[0],
            end=run.points[-1],
            active_at=None,
            born_between=None,
            options=options.lifetime_options(),
            _raw_snapshots=tuple(view.raw for view in interval_views),
        )
    point_comparisons = (
        _build_point_comparisons(run, comparison_options, interval_views)
        if options.stacks or options.events
        else ()
    )
    return MemoryTimeline(
        run=run,
        allocator_scope_entries=tuple(allocator_scope_entries),
        pool_entries=tuple(pool_entries),
        observation_entries=tuple(observation_entries),
        point_comparisons=point_comparisons,
        allocation_lifetimes=allocation_lifetimes,
    )


def _build_point_comparisons(
    run: MemoryRun,
    options: MemoryAttributionOptions,
    interval_views: tuple[_MemoryStateView, ...],
) -> tuple[MemoryPointComparison, ...]:
    if len(run.points) < 2:
        return ()
    rows = []
    if len(interval_views) != len(run.points):
        raise ValueError("timeline interval views must match run points")
    reference_view = interval_views[0]
    for candidate_view in interval_views[1:]:
        rows.append(
            _compare_same_run_views(
                run,
                reference_view,
                candidate_view,
                options,
                interval_views=(reference_view, candidate_view),
            )
        )
        reference_view = candidate_view
    return tuple(rows)
