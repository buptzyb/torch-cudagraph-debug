"""Two-point and phase memory comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ._pool_identity import PoolId
from .aggregation import (
    ALLOCATOR_SCOPES,
    compare_allocator_scopes,
    summarize_allocator_scopes,
)
from .allocator_snapshot import (
    DEFAULT_POOL_ID,
    AllocatorSnapshotData,
    compare_observation_lifecycle,
    normalize_pool_id,
    normalize_snapshot,
    normalize_trace_entries,
    pool_id_label,
    stream_label,
    trace_device_indices,
)
from .attribution import MemoryAttributionOptions
from .comparison_models import (
    MemoryLifecycleDelta,
    MemoryObservationComparison,
    MemoryPoolComparison,
)
from .errors import MemoryDebugError, MemoryHistoryError, MemoryOwnershipError
from .events import (
    extract_event_window,
    extract_event_window_from_snapshot,
    summarize_allocator_events,
)
from .lifetimes import (
    analyze_allocation_lifetimes,
    analyze_probe_snapshot_lifetimes,
)
from .recording import MemoryPoint, MemoryRange, MemoryRun
from .reports import (
    MemoryPhaseComparison,
    MemoryPointComparison,
    MemorySnapshotComparison,
)
from .snapshots import MemoryProbeSnapshot
from .stacks import (
    AllocationStackCoverage,
    AllocationStackDelta,
    _AllocationStackIndex,
    _build_allocation_stack_index,
    _compare_mapped_stack_indexes,
    _compare_stack_indexes,
)
from .stats import MemoryStats, MemoryStatsDelta

_MemoryState = MemoryPoint | MemoryProbeSnapshot


@dataclass(frozen=True)
class _MemoryStateView:
    state: _MemoryState
    raw: AllocatorSnapshotData
    segments: tuple[Mapping[str, Any], ...]
    stacks: _AllocationStackIndex | None


def _load_state(
    state: _MemoryState,
    options: MemoryAttributionOptions,
    *,
    raw: AllocatorSnapshotData | None = None,
) -> _MemoryStateView:
    snapshot = state.raw_snapshot() if raw is None else raw
    segments = normalize_snapshot(snapshot)
    return _MemoryStateView(
        state=state,
        raw=snapshot,
        segments=segments,
        stacks=(_build_allocation_stack_index(segments) if options.stacks else None),
    )


def compare_snapshots(
    reference: MemoryProbeSnapshot,
    candidate: MemoryProbeSnapshot,
    *,
    pool_mapping: Mapping[Any, Any] | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemorySnapshotComparison:
    """Compare Probe snapshots using same- or cross-Probe semantics."""

    options = attribution or MemoryAttributionOptions()
    same_probe = reference.probe_id == candidate.probe_id
    if same_probe and candidate.index <= reference.index:
        raise ValueError("candidate snapshot must follow reference snapshot")
    if same_probe and pool_mapping is not None:
        raise ValueError("pool_mapping is only valid for cross-Probe comparison")

    if same_probe:
        interval_views = _load_interval_views((reference, candidate), options)
        comparison = _compare_same_identity_views(
            interval_views[0],
            interval_views[-1],
            options,
            interval_views=interval_views,
            lifetime_run=None,
            match="same_probe",
        )
    else:
        comparison = _compare_independent_states(
            reference,
            candidate,
            pool_mapping=pool_mapping,
            options=options,
            independent_kind="probes",
        )
    assert isinstance(comparison, MemorySnapshotComparison)
    return comparison


def compare_points(
    reference: MemoryPoint,
    candidate: MemoryPoint,
    *,
    pool_mapping: Mapping[Any, Any] | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemoryPointComparison:
    """Compare points from independent runs without address/event identity."""

    if reference.run_id == candidate.run_id:
        raise MemoryOwnershipError(
            "compare_points requires independent runs; use MemoryRun.compare "
            "for points from the same run"
        )
    comparison = _compare_independent_states(
        reference,
        candidate,
        pool_mapping=pool_mapping,
        options=attribution or MemoryAttributionOptions(),
    )
    assert isinstance(comparison, MemoryPointComparison)
    return comparison


def _compare_independent_states(
    reference: _MemoryState,
    candidate: _MemoryState,
    *,
    pool_mapping: Mapping[Any, Any] | None,
    options: MemoryAttributionOptions,
    reference_stack_index: _AllocationStackIndex | None = None,
    candidate_stack_index: _AllocationStackIndex | None = None,
    independent_kind: Literal["runs", "probes"] = "runs",
) -> MemoryPointComparison | MemorySnapshotComparison:
    if options.events:
        raise MemoryDebugError(
            f"allocator events cannot be compared across independent {independent_kind}"
        )
    if options.lifetimes:
        raise MemoryDebugError(
            f"allocation lifetimes cannot be compared across independent {independent_kind}"
        )

    reference_pool_stats = reference.pool_stats
    candidate_pool_stats = candidate.pool_stats
    mapping = _validate_pool_mapping(
        reference_pool_stats, candidate_pool_stats, pool_mapping
    )
    pool_comparisons: list[MemoryPoolComparison] = []
    matched_reference: set[PoolId] = set()
    matched_candidate: set[PoolId] = set()
    for reference_pool, (candidate_pool, match) in mapping.items():
        matched_reference.add(reference_pool)
        matched_candidate.add(candidate_pool)
        reference_stats = reference_pool_stats[reference_pool]
        candidate_stats = candidate_pool_stats[candidate_pool]
        pool_comparisons.append(
            MemoryPoolComparison(
                reference_pool_id=reference_pool,
                candidate_pool_id=candidate_pool,
                match=match,
                reference=reference_stats,
                candidate=candidate_stats,
                delta=MemoryStatsDelta.between(reference_stats, candidate_stats),
                lifecycle=None,
            )
        )
    for pool_id in set(reference_pool_stats) - matched_reference:
        reference_stats = reference_pool_stats[pool_id]
        candidate_stats = MemoryStats()
        pool_comparisons.append(
            MemoryPoolComparison(
                reference_pool_id=pool_id,
                candidate_pool_id=None,
                match="reference_only",
                reference=reference_stats,
                candidate=candidate_stats,
                delta=MemoryStatsDelta.between(reference_stats, candidate_stats),
                lifecycle=None,
            )
        )
    for pool_id in set(candidate_pool_stats) - matched_candidate:
        reference_stats = MemoryStats()
        candidate_stats = candidate_pool_stats[pool_id]
        pool_comparisons.append(
            MemoryPoolComparison(
                reference_pool_id=None,
                candidate_pool_id=pool_id,
                match="candidate_only",
                reference=reference_stats,
                candidate=candidate_stats,
                delta=MemoryStatsDelta.between(reference_stats, candidate_stats),
                lifecycle=None,
            )
        )

    observation_comparisons = _unmatched_independent_observations(reference, candidate)
    warnings = [*reference.warnings, *candidate.warnings]
    unmatched_private = (
        set(reference_pool_stats) - matched_reference - {DEFAULT_POOL_ID}
    ) or (set(candidate_pool_stats) - matched_candidate - {DEFAULT_POOL_ID})
    if unmatched_private:
        warnings.append(
            f"private pools are unmatched across independent {independent_kind} "
            "unless pool_mapping explicitly pairs them"
        )

    reference_coverage: AllocationStackCoverage | None = None
    candidate_coverage: AllocationStackCoverage | None = None
    stack_deltas: tuple[AllocationStackDelta, ...] = ()
    if options.stacks:
        reference_index = reference_stack_index or _build_allocation_stack_index(
            normalize_snapshot(reference.raw_snapshot())
        )
        candidate_index = candidate_stack_index or _build_allocation_stack_index(
            normalize_snapshot(candidate.raw_snapshot())
        )
        reference_coverage = reference_index.coverage
        candidate_coverage = candidate_index.coverage
        _check_stack_coverage(
            reference_coverage,
            candidate_coverage,
            options,
            warnings,
        )
        stack_deltas = _compare_mapped_stack_indexes(
            reference_index,
            candidate_index,
            mapping,
        )
        if unmatched_private:
            relation = "cross-run" if independent_kind == "runs" else "cross-probe"
            warnings.append(
                "allocation stacks for unmatched private pools are omitted "
                f"from {relation} stack deltas"
            )

    result_type = _comparison_result_type(reference, candidate)
    return result_type(
        reference=reference,
        candidate=candidate,
        allocator_scope_comparisons=compare_allocator_scopes(
            summarize_allocator_scopes(reference_pool_stats),
            summarize_allocator_scopes(candidate_pool_stats),
        ),
        pool_comparisons=tuple(sorted(pool_comparisons, key=_pool_comparison_sort_key)),
        observation_comparisons=observation_comparisons,
        allocation_stack_comparisons=stack_deltas,
        reference_stack_coverage=reference_coverage,
        candidate_stack_coverage=candidate_coverage,
        lifecycle_available=False,
        display_stack_depth=options.stack_depth,
        display_limit=options.limit,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def compare_phases(
    baseline: MemoryRange,
    candidate: MemoryRange,
    *,
    pool_mapping: Mapping[Any, Any] | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemoryPhaseComparison:
    """Decompose candidate-vs-baseline phase memory using four points."""

    if baseline.run.run_id == candidate.run.run_id:
        raise ValueError(
            "baseline and candidate phases must come from independent runs"
        )
    options = attribution or MemoryAttributionOptions()
    baseline_states = (
        baseline.run.points[baseline.start.index : baseline.end.index + 1]
        if options.events or options.lifetimes
        else (baseline.start, baseline.end)
    )
    baseline_views = _load_interval_views(baseline_states, options)
    baseline_start_view = baseline_views[0]
    baseline_end_view = baseline_views[-1]
    baseline_change = _compare_same_run_views(
        baseline.run,
        baseline_start_view,
        baseline_end_view,
        options,
        interval_views=baseline_views,
    )
    baseline_start_stacks = baseline_start_view.stacks
    baseline_end_stacks = baseline_end_view.stacks
    del baseline_views, baseline_start_view, baseline_end_view

    candidate_states = (
        candidate.run.points[candidate.start.index : candidate.end.index + 1]
        if options.events or options.lifetimes
        else (candidate.start, candidate.end)
    )
    candidate_views = _load_interval_views(candidate_states, options)
    candidate_start_view = candidate_views[0]
    candidate_end_view = candidate_views[-1]
    candidate_change = _compare_same_run_views(
        candidate.run,
        candidate_start_view,
        candidate_end_view,
        options,
        interval_views=candidate_views,
    )
    candidate_start_stacks = candidate_start_view.stacks
    candidate_end_stacks = candidate_end_view.stacks
    del candidate_views, candidate_start_view, candidate_end_view

    cross_options = MemoryAttributionOptions(
        stacks=options.stacks,
        events=False,
        lifetimes=False,
        on_missing=options.on_missing,
        stack_depth=options.stack_depth,
        limit=options.limit,
    )
    start_gap = _compare_independent_states(
        baseline.start,
        candidate.start,
        pool_mapping=pool_mapping,
        options=cross_options,
        reference_stack_index=baseline_start_stacks,
        candidate_stack_index=candidate_start_stacks,
    )
    end_gap = _compare_independent_states(
        baseline.end,
        candidate.end,
        pool_mapping=pool_mapping,
        options=cross_options,
        reference_stack_index=baseline_end_stacks,
        candidate_stack_index=candidate_end_stacks,
    )
    pool_decomposition = _phase_decomposition(
        baseline_change,
        candidate_change,
        start_gap,
        end_gap,
    )
    allocator_scope_decomposition = _scope_phase_decomposition(
        baseline_change,
        candidate_change,
        start_gap,
        end_gap,
    )
    return MemoryPhaseComparison(
        baseline_name=baseline.run.name,
        candidate_name=candidate.run.name,
        baseline_change=baseline_change,
        candidate_change=candidate_change,
        start_gap=start_gap,
        end_gap=end_gap,
        allocator_scope_decomposition=allocator_scope_decomposition,
        pool_decomposition=pool_decomposition,
        display_stack_depth=options.stack_depth,
        display_limit=options.limit,
    )


def _compare_same_run(
    run: MemoryRun,
    reference: MemoryPoint,
    candidate: MemoryPoint,
    options: MemoryAttributionOptions,
) -> MemoryPointComparison:
    states = (
        run.points[reference.index : candidate.index + 1]
        if options.events or options.lifetimes
        else (reference, candidate)
    )
    interval_views = _load_interval_views(states, options)
    return _compare_same_run_views(
        run,
        interval_views[0],
        interval_views[-1],
        options,
        interval_views=interval_views,
    )


def _compare_same_run_views(
    run: MemoryRun,
    reference_view: _MemoryStateView,
    candidate_view: _MemoryStateView,
    options: MemoryAttributionOptions,
    *,
    interval_views: Sequence[_MemoryStateView] | None = None,
) -> MemoryPointComparison:
    reference = reference_view.state
    candidate = candidate_view.state
    assert isinstance(reference, MemoryPoint)
    assert isinstance(candidate, MemoryPoint)
    selected_views = (
        tuple(interval_views)
        if interval_views is not None
        else (reference_view, candidate_view)
    )
    comparison = _compare_same_identity_views(
        reference_view,
        candidate_view,
        options,
        interval_views=selected_views,
        lifetime_run=run,
        match="same_run",
    )
    assert isinstance(comparison, MemoryPointComparison)
    return comparison


def _compare_same_identity_views(
    reference_view: _MemoryStateView,
    candidate_view: _MemoryStateView,
    options: MemoryAttributionOptions,
    *,
    interval_views: Sequence[_MemoryStateView],
    lifetime_run: MemoryRun | None,
    match: Literal["same_run", "same_probe"],
) -> MemoryPointComparison | MemorySnapshotComparison:
    reference = reference_view.state
    candidate = candidate_view.state
    reference_segments = reference_view.segments
    candidate_segments = candidate_view.segments
    reference_pool_stats = reference.pool_stats
    candidate_pool_stats = candidate.pool_stats

    observation_lifecycle = compare_observation_lifecycle(
        reference_segments, candidate_segments
    )
    lifecycle_by_pool: dict[PoolId, list[MemoryLifecycleDelta]] = {}
    for key, lifecycle in observation_lifecycle.items():
        lifecycle_by_pool.setdefault(key.pool_id, []).append(lifecycle)
    pool_lifecycle = {
        pool_id: MemoryLifecycleDelta.combine(values)
        for pool_id, values in lifecycle_by_pool.items()
    }

    pool_comparisons = tuple(
        MemoryPoolComparison(
            reference_pool_id=pool_id,
            candidate_pool_id=pool_id,
            match=match,
            reference=reference_pool_stats.get(pool_id, MemoryStats()),
            candidate=candidate_pool_stats.get(pool_id, MemoryStats()),
            delta=MemoryStatsDelta.between(
                reference_pool_stats.get(pool_id, MemoryStats()),
                candidate_pool_stats.get(pool_id, MemoryStats()),
            ),
            lifecycle=pool_lifecycle.get(pool_id, MemoryLifecycleDelta()),
        )
        for pool_id in sorted(
            set(reference_pool_stats) | set(candidate_pool_stats), key=pool_id_label
        )
    )
    observation_comparisons = tuple(
        MemoryObservationComparison(
            reference_pool_id=key.pool_id,
            reference_stream=key.stream,
            candidate_pool_id=key.pool_id,
            candidate_stream=key.stream,
            match=match,
            reference=reference.observation_stats.get(key, MemoryStats()),
            candidate=candidate.observation_stats.get(key, MemoryStats()),
            delta=MemoryStatsDelta.between(
                reference.observation_stats.get(key, MemoryStats()),
                candidate.observation_stats.get(key, MemoryStats()),
            ),
            lifecycle=observation_lifecycle.get(key, MemoryLifecycleDelta()),
        )
        for key in sorted(
            set(reference.observation_stats) | set(candidate.observation_stats),
            key=lambda item: (pool_id_label(item.pool_id), stream_label(item.stream)),
        )
    )

    warnings = [*reference.warnings, *candidate.warnings]
    reference_coverage: AllocationStackCoverage | None = None
    candidate_coverage: AllocationStackCoverage | None = None
    stack_deltas: tuple[AllocationStackDelta, ...] = ()
    stack_detail_deltas: tuple[AllocationStackDelta, ...] = ()
    if options.stacks:
        reference_index = reference_view.stacks
        candidate_index = candidate_view.stacks
        assert reference_index is not None and candidate_index is not None
        reference_coverage = reference_index.coverage
        candidate_coverage = candidate_index.coverage
        _check_stack_coverage(
            reference_coverage,
            candidate_coverage,
            options,
            warnings,
        )
        stack_deltas = _compare_stack_indexes(
            reference_index,
            candidate_index,
            by_stream=False,
        )
        stack_detail_deltas = _compare_stack_indexes(
            reference_index,
            candidate_index,
            by_stream=True,
        )

    allocator_events = ()
    events_available = False
    events_complete = True
    if options.events:
        entries = []
        events_available = True
        event_devices = _segment_device_indices(reference_segments, candidate_segments)
        for reference_block, candidate_block in zip(interval_views, interval_views[1:]):
            reference_state = reference_block.state
            candidate_state = candidate_block.state
            current_snapshot = candidate_block.raw
            interval_devices = tuple(
                sorted(set(event_devices) | set(trace_device_indices(current_snapshot)))
            )
            if interval_devices:
                windows = tuple(
                    extract_event_window_from_snapshot(
                        current_snapshot,
                        device_index=device,
                        start_marker=reference_state.boundary_marker,
                        end_marker=candidate_state.boundary_marker,
                        start_label=f"{_state_label(reference_state)} on device {device}",
                    )
                    for device in interval_devices
                )
            else:
                windows = (
                    extract_event_window(
                        normalize_trace_entries(current_snapshot),
                        start_marker=reference_state.boundary_marker,
                        end_marker=candidate_state.boundary_marker,
                        start_label=_state_label(reference_state),
                    ),
                )
            for window in windows:
                entries.extend(window.entries)
                events_available = events_available and window.available
                events_complete = events_complete and window.complete
                warnings.extend(window.warnings)
        if not events_available or not events_complete:
            _missing(
                "allocator event history is unavailable or incomplete; "
                "enable torch.cuda.memory._record_memory_history(...) "
                "before the phase",
                options,
                warnings,
            )
        allocator_events = summarize_allocator_events(
            entries,
            reference_segments=reference_segments,
            candidate_segments=candidate_segments,
        )

    allocation_lifetimes = None
    if options.lifetimes:
        lifetime_options = options.lifetime_options()
        if lifetime_run is None:
            assert isinstance(reference, MemoryProbeSnapshot)
            assert isinstance(candidate, MemoryProbeSnapshot)
            allocation_lifetimes = analyze_probe_snapshot_lifetimes(
                reference,
                candidate,
                options=lifetime_options,
                _raw_snapshots=tuple(view.raw for view in interval_views),
            )
        else:
            assert isinstance(reference, MemoryPoint)
            assert isinstance(candidate, MemoryPoint)
            allocation_lifetimes = analyze_allocation_lifetimes(
                lifetime_run,
                start=reference,
                end=candidate,
                active_at=None,
                born_between=None,
                options=lifetime_options,
                _raw_snapshots=tuple(view.raw for view in interval_views),
            )
        warnings.extend(allocation_lifetimes.warnings)

    result_type = _comparison_result_type(reference, candidate)
    return result_type(
        reference=reference,
        candidate=candidate,
        allocator_scope_comparisons=compare_allocator_scopes(
            reference.allocator_scope_stats, candidate.allocator_scope_stats
        ),
        pool_comparisons=pool_comparisons,
        observation_comparisons=observation_comparisons,
        allocation_stack_comparisons=stack_deltas,
        allocation_stack_observation_comparisons=stack_detail_deltas,
        reference_stack_coverage=reference_coverage,
        candidate_stack_coverage=candidate_coverage,
        allocator_events=allocator_events,
        allocation_lifetimes=allocation_lifetimes,
        events_available=events_available,
        events_complete=events_complete,
        lifecycle_available=True,
        display_stack_depth=options.stack_depth,
        display_limit=options.limit,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _load_interval_views(
    states: Sequence[_MemoryState],
    options: MemoryAttributionOptions,
) -> tuple[_MemoryStateView, ...]:
    return tuple(_load_state(state, options) for state in states)


def _state_label(state: _MemoryState) -> str:
    if isinstance(state, MemoryProbeSnapshot):
        return f"{state.probe_name}@snapshot-{state.index}"
    return state.label


def _comparison_result_type(
    reference: _MemoryState,
    candidate: _MemoryState,
) -> type[MemoryPointComparison] | type[MemorySnapshotComparison]:
    reference_is_snapshot = isinstance(reference, MemoryProbeSnapshot)
    candidate_is_snapshot = isinstance(candidate, MemoryProbeSnapshot)
    if reference_is_snapshot != candidate_is_snapshot:
        raise TypeError("memory comparison endpoints must use the same workflow type")
    return MemorySnapshotComparison if reference_is_snapshot else MemoryPointComparison


def _segment_device_indices(
    *segment_sets: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    devices = {
        int(segment["device"])
        for segments in segment_sets
        for segment in segments
        if segment.get("device") is not None
    }
    return tuple(sorted(devices))


def _validate_pool_mapping(
    reference_pool_stats: Mapping[PoolId, MemoryStats],
    candidate_pool_stats: Mapping[PoolId, MemoryStats],
    pool_mapping: Mapping[Any, Any] | None,
) -> dict[PoolId, tuple[PoolId, Literal["default", "mapped"]]]:
    result: dict[PoolId, tuple[PoolId, Literal["default", "mapped"]]] = {}
    if (
        DEFAULT_POOL_ID in reference_pool_stats
        and DEFAULT_POOL_ID in candidate_pool_stats
    ):
        result[DEFAULT_POOL_ID] = (DEFAULT_POOL_ID, "default")

    candidate_pools: set[PoolId] = {
        candidate_pool for candidate_pool, _match in result.values()
    }
    for raw_reference, raw_candidate in (pool_mapping or {}).items():
        reference_pool = normalize_pool_id(raw_reference)
        candidate_pool = normalize_pool_id(raw_candidate)
        if reference_pool == DEFAULT_POOL_ID or candidate_pool == DEFAULT_POOL_ID:
            if reference_pool != DEFAULT_POOL_ID or candidate_pool != DEFAULT_POOL_ID:
                raise ValueError("the default pool may only map to the default pool")
            continue
        if reference_pool not in reference_pool_stats:
            raise ValueError(
                f"mapped reference pool {pool_id_label(reference_pool)} does not exist"
            )
        if candidate_pool not in candidate_pool_stats:
            raise ValueError(
                f"mapped candidate pool {pool_id_label(candidate_pool)} does not exist"
            )
        if reference_pool in result:
            raise ValueError(
                f"reference pool {pool_id_label(reference_pool)} is mapped more than once"
            )
        if candidate_pool in candidate_pools:
            raise ValueError(
                f"candidate pool {pool_id_label(candidate_pool)} is mapped more than once"
            )
        result[reference_pool] = (candidate_pool, "mapped")
        candidate_pools.add(candidate_pool)
    return result


def _unmatched_independent_observations(
    reference: _MemoryState,
    candidate: _MemoryState,
) -> tuple[MemoryObservationComparison, ...]:
    rows: list[MemoryObservationComparison] = []
    for key, reference_stats in reference.observation_stats.items():
        candidate_stats = MemoryStats()
        rows.append(
            MemoryObservationComparison(
                reference_pool_id=key.pool_id,
                reference_stream=key.stream,
                candidate_pool_id=None,
                candidate_stream=None,
                match="reference_only",
                reference=reference_stats,
                candidate=candidate_stats,
                delta=MemoryStatsDelta.between(reference_stats, candidate_stats),
                lifecycle=None,
            )
        )
    for key, candidate_stats in candidate.observation_stats.items():
        reference_stats = MemoryStats()
        rows.append(
            MemoryObservationComparison(
                reference_pool_id=None,
                reference_stream=None,
                candidate_pool_id=key.pool_id,
                candidate_stream=key.stream,
                match="candidate_only",
                reference=reference_stats,
                candidate=candidate_stats,
                delta=MemoryStatsDelta.between(reference_stats, candidate_stats),
                lifecycle=None,
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                _optional_pool_label(item.candidate_pool_id or item.reference_pool_id),
                stream_label(
                    item.candidate_stream
                    if item.candidate_pool_id is not None
                    else item.reference_stream
                ),
                item.match,
            ),
        )
    )


def _check_stack_coverage(
    reference: AllocationStackCoverage,
    candidate: AllocationStackCoverage,
    options: MemoryAttributionOptions,
    warnings: list[str],
) -> None:
    if reference.unattributed_bytes or candidate.unattributed_bytes:
        _missing(
            "allocation stack coverage is incomplete: "
            f"reference={reference.ratio:.1%}, candidate={candidate.ratio:.1%}; "
            "enable torch.cuda.memory._record_memory_history(...) "
            "before the allocations",
            options,
            warnings,
        )


def _missing(
    message: str,
    options: MemoryAttributionOptions,
    warnings: list[str],
) -> None:
    if options.on_missing == "error":
        raise MemoryHistoryError(message)
    warnings.append(message)


def _phase_decomposition(
    baseline_change: MemoryPointComparison,
    candidate_change: MemoryPointComparison,
    start_gap: MemoryPointComparison,
    end_gap: MemoryPointComparison,
) -> tuple[dict[str, object], ...]:
    baseline_change_by_pool = {
        item.reference_pool_id: item
        for item in baseline_change.pool_comparisons
        if item.reference_pool_id is not None
    }
    candidate_change_by_pool = {
        item.reference_pool_id: item
        for item in candidate_change.pool_comparisons
        if item.reference_pool_id is not None
    }
    end_gap_by_pair = {
        (item.reference_pool_id, item.candidate_pool_id): item
        for item in end_gap.pool_comparisons
    }
    rows: list[dict[str, object]] = []
    for start_gap_comparison in start_gap.pool_comparisons:
        if start_gap_comparison.match not in {"default", "mapped"}:
            continue
        baseline_pool = start_gap_comparison.reference_pool_id
        candidate_pool = start_gap_comparison.candidate_pool_id
        if baseline_pool is None or candidate_pool is None:
            continue
        baseline_change_comparison = baseline_change_by_pool.get(baseline_pool)
        candidate_change_comparison = candidate_change_by_pool.get(candidate_pool)
        end_gap_comparison = end_gap_by_pair.get((baseline_pool, candidate_pool))
        if (
            baseline_change_comparison is None
            or candidate_change_comparison is None
            or end_gap_comparison is None
        ):
            continue
        for metric in (
            "reserved_bytes",
            "allocated_bytes",
            "active_bytes",
            "inactive_bytes",
            "requested_bytes",
            "fragmentation_bytes",
        ):
            baseline_value = int(getattr(baseline_change_comparison.delta, metric))
            candidate_value = int(getattr(candidate_change_comparison.delta, metric))
            start_value = int(getattr(start_gap_comparison.delta, metric))
            end_value = int(getattr(end_gap_comparison.delta, metric))
            rows.append(
                {
                    "baseline_pool_id": pool_id_label(baseline_pool),
                    "candidate_pool_id": pool_id_label(candidate_pool),
                    "pool": (
                        f"{pool_id_label(baseline_pool)} -> "
                        f"{pool_id_label(candidate_pool)}"
                    ),
                    "metric": metric,
                    "start_gap_bytes": start_value,
                    "baseline_change_bytes": baseline_value,
                    "candidate_change_bytes": candidate_value,
                    "change_gap_bytes": candidate_value - baseline_value,
                    "end_gap_bytes": end_value,
                    "identity_holds": (
                        end_value == start_value + candidate_value - baseline_value
                    ),
                }
            )
    return tuple(rows)


def _scope_phase_decomposition(
    baseline_change: MemoryPointComparison,
    candidate_change: MemoryPointComparison,
    start_gap: MemoryPointComparison,
    end_gap: MemoryPointComparison,
) -> tuple[dict[str, object], ...]:
    baseline_change_by_scope = {
        item.scope: item for item in baseline_change.allocator_scope_comparisons
    }
    candidate_change_by_scope = {
        item.scope: item for item in candidate_change.allocator_scope_comparisons
    }
    start_gap_by_scope = {
        item.scope: item for item in start_gap.allocator_scope_comparisons
    }
    end_gap_by_scope = {
        item.scope: item for item in end_gap.allocator_scope_comparisons
    }
    rows = []
    for scope in ALLOCATOR_SCOPES:
        baseline_change_comparison = baseline_change_by_scope[scope]
        candidate_change_comparison = candidate_change_by_scope[scope]
        start_gap_comparison = start_gap_by_scope[scope]
        end_gap_comparison = end_gap_by_scope[scope]
        for metric in (
            "reserved_bytes",
            "allocated_bytes",
            "active_bytes",
            "inactive_bytes",
            "requested_bytes",
            "fragmentation_bytes",
        ):
            baseline_value = int(getattr(baseline_change_comparison.delta, metric))
            candidate_value = int(getattr(candidate_change_comparison.delta, metric))
            start_value = int(getattr(start_gap_comparison.delta, metric))
            end_value = int(getattr(end_gap_comparison.delta, metric))
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "start_gap_bytes": start_value,
                    "baseline_change_bytes": baseline_value,
                    "candidate_change_bytes": candidate_value,
                    "change_gap_bytes": candidate_value - baseline_value,
                    "end_gap_bytes": end_value,
                    "identity_holds": (
                        end_value == start_value + candidate_value - baseline_value
                    ),
                }
            )
    return tuple(rows)


def _pool_comparison_sort_key(
    item: MemoryPoolComparison,
) -> tuple[int, str, str]:
    match_order = {
        "default": 0,
        "mapped": 1,
        "same_run": 0,
        "same_probe": 0,
        "reference_only": 2,
        "candidate_only": 3,
    }
    return (
        match_order[item.match],
        _optional_pool_label(item.reference_pool_id),
        _optional_pool_label(item.candidate_pool_id),
    )


def _optional_pool_label(pool_id: PoolId | None) -> str:
    return "" if pool_id is None else pool_id_label(pool_id)
