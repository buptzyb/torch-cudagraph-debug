"""Two-point and phase memory comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from ._pool_identity import DEFAULT_POOL_ID, MemoryPoolKey
from .aggregation import (
    ALLOCATOR_SCOPES,
    compare_allocator_scopes,
    compare_device_memory,
    summarize_allocator_scopes,
)
from .allocator_snapshot import (
    AllocatorSnapshotData,
    compare_observation_lifecycle,
    normalize_snapshot,
    trace_device_indices,
)
from .attribution import (
    MemoryAttributionOptions,
    MemoryAttributionStatus,
    MemoryEvidenceStatus,
)
from .comparison_models import (
    DEVICE_MEMORY_METRICS,
    PHASE_METRICS,
    DeviceMatchKind,
    MemoryAllocatorScopePhaseDecomposition,
    MemoryDeviceComparison,
    MemoryDevicePhaseDecomposition,
    MemoryLifecycleDelta,
    MemoryObservationComparison,
    MemoryPhaseComponents,
    MemoryPoolComparison,
    MemoryPoolPhaseDecomposition,
)
from .errors import (
    MemoryDebugError,
    MemoryHistoryBoundaryError,
    MemoryHistoryDisabledError,
    MemoryHistoryTruncatedError,
    MemoryOwnershipError,
    MemoryReconciliationError,
)
from .events import (
    EventWindow,
    _extract_snapshot_event_windows,
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


def _lifecycle_identity_is_exact(segments: Sequence[Mapping[str, Any]]) -> bool:
    for segment in segments:
        if segment.get("address") is None:
            return False
        for block in segment.get("blocks", ()):
            if (
                str(block.get("state")).startswith("active_")
                and block.get("address") is None
            ):
                return False
    return True


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
    snapshot = (
        state.raw_snapshot()
        if raw is None and isinstance(state, MemoryProbeSnapshot)
        else state.allocator_state()
        if raw is None
        else raw
    )
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
    pool_mapping: Mapping[MemoryPoolKey, MemoryPoolKey] | None = None,
    device_mapping: Mapping[int, int] | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemorySnapshotComparison:
    """Compare Probe snapshots using same- or cross-Probe semantics.

    ``pool_mapping`` and ``device_mapping`` pair pools and devices across
    independent Probes; ``attribution`` selects stack, event, and lifetime
    evidence. Raises ValueError when a same-probe candidate does not follow
    the reference or mappings are passed for a same-probe pair,
    MemoryDebugError when events or lifetimes are requested across
    independent Probes, and the MemoryHistory* errors when requested
    same-probe event or lifetime evidence is unusable.
    """

    options = attribution or MemoryAttributionOptions()
    same_probe = reference.probe_id == candidate.probe_id
    if same_probe and candidate.snapshot_index <= reference.snapshot_index:
        raise ValueError("candidate snapshot must follow reference snapshot")
    if same_probe and (pool_mapping is not None or device_mapping is not None):
        raise ValueError(
            "pool_mapping and device_mapping are only valid for cross-Probe comparison"
        )

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
            device_mapping=device_mapping,
            options=options,
            independent_kind="probes",
        )
    assert isinstance(comparison, MemorySnapshotComparison)
    return comparison


def compare_points(
    reference: MemoryPoint,
    candidate: MemoryPoint,
    *,
    pool_mapping: Mapping[MemoryPoolKey, MemoryPoolKey] | None = None,
    device_mapping: Mapping[int, int] | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemoryPointComparison:
    """Compare points from independent runs without address/event identity.

    ``pool_mapping`` and ``device_mapping`` pair private pools and devices
    across the runs; ``attribution`` may request stacks only. Raises
    MemoryOwnershipError for points from the same run (use
    ``MemoryRun.compare`` instead) and MemoryDebugError when events or
    lifetimes attribution is requested.
    """

    if reference.run_id == candidate.run_id:
        raise MemoryOwnershipError(
            "compare_points requires independent runs; use MemoryRun.compare "
            "for points from the same run"
        )
    comparison = _compare_independent_states(
        reference,
        candidate,
        pool_mapping=pool_mapping,
        device_mapping=device_mapping,
        options=attribution or MemoryAttributionOptions(),
    )
    assert isinstance(comparison, MemoryPointComparison)
    return comparison


def _compare_independent_states(
    reference: _MemoryState,
    candidate: _MemoryState,
    *,
    pool_mapping: Mapping[MemoryPoolKey, MemoryPoolKey] | None,
    device_mapping: Mapping[int, int] | None,
    options: MemoryAttributionOptions,
    reference_pool_universe: Mapping[MemoryPoolKey, MemoryStats] | None = None,
    candidate_pool_universe: Mapping[MemoryPoolKey, MemoryStats] | None = None,
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
    resolved_reference_pools = (
        reference_pool_stats
        if reference_pool_universe is None
        else reference_pool_universe
    )
    resolved_candidate_pools = (
        candidate_pool_stats
        if candidate_pool_universe is None
        else candidate_pool_universe
    )
    device_pairs = _resolve_device_pairs(
        reference,
        candidate,
        resolved_reference_pools,
        resolved_candidate_pools,
        pool_mapping,
        device_mapping,
    )
    mapping = _validate_pool_mapping(
        resolved_reference_pools,
        resolved_candidate_pools,
        pool_mapping,
        device_pairs,
    )
    pool_comparisons: list[MemoryPoolComparison] = []
    matched_reference: set[MemoryPoolKey] = set()
    matched_candidate: set[MemoryPoolKey] = set()
    for reference_key, (candidate_key, match) in mapping.items():
        matched_reference.add(reference_key)
        matched_candidate.add(candidate_key)
        reference_stats = reference_pool_stats.get(reference_key, MemoryStats())
        candidate_stats = candidate_pool_stats.get(candidate_key, MemoryStats())
        pool_comparisons.append(
            MemoryPoolComparison(
                reference_key=reference_key,
                candidate_key=candidate_key,
                match=match,
                reference=reference_stats,
                candidate=candidate_stats,
                delta=MemoryStatsDelta.between(reference_stats, candidate_stats),
                lifecycle=None,
            )
        )
    for pool_key in set(reference_pool_stats) - matched_reference:
        reference_stats = reference_pool_stats[pool_key]
        candidate_stats = MemoryStats()
        pool_comparisons.append(
            MemoryPoolComparison(
                reference_key=pool_key,
                candidate_key=None,
                match="reference_only",
                reference=reference_stats,
                candidate=candidate_stats,
                delta=MemoryStatsDelta.between(reference_stats, candidate_stats),
                lifecycle=None,
            )
        )
    for pool_key in set(candidate_pool_stats) - matched_candidate:
        reference_stats = MemoryStats()
        candidate_stats = candidate_pool_stats[pool_key]
        pool_comparisons.append(
            MemoryPoolComparison(
                reference_key=None,
                candidate_key=pool_key,
                match="candidate_only",
                reference=reference_stats,
                candidate=candidate_stats,
                delta=MemoryStatsDelta.between(reference_stats, candidate_stats),
                lifecycle=None,
            )
        )

    observation_comparisons = _unmatched_independent_observations(reference, candidate)
    warnings = [*reference.warnings, *candidate.warnings]
    unmatched_reference = set(reference_pool_stats) - matched_reference
    unmatched_candidate = set(candidate_pool_stats) - matched_candidate
    unmatched_private = any(
        key.pool_id != DEFAULT_POOL_ID
        for key in unmatched_reference | unmatched_candidate
    )
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
            normalize_snapshot(reference.allocator_state())
        )
        candidate_index = candidate_stack_index or _build_allocation_stack_index(
            normalize_snapshot(candidate.allocator_state())
        )
        reference_coverage = reference_index.coverage
        candidate_coverage = candidate_index.coverage
        _check_stack_coverage(reference_coverage, candidate_coverage, warnings)
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
        device_comparisons=compare_device_memory(
            reference.device_memory,
            candidate.device_memory,
            reference_pool_stats,
            candidate_pool_stats,
            device_pairs=device_pairs,
        ),
        allocation_stack_comparisons=stack_deltas,
        reference_stack_coverage=reference_coverage,
        candidate_stack_coverage=candidate_coverage,
        attribution_status=_attribution_status(
            options,
            reference_coverage=reference_coverage,
            candidate_coverage=candidate_coverage,
            events_available=False,
            events_complete=False,
            allocation_lifetimes=None,
        ),
        lifecycle_available=False,
        lifecycle_confidence="unavailable",
        display_stack_depth=options.display.stack_depth,
        display_limit=options.display.limit,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def compare_phases(
    baseline: MemoryRange,
    candidate: MemoryRange,
    *,
    pool_mapping: Mapping[MemoryPoolKey, MemoryPoolKey] | None = None,
    device_mapping: Mapping[int, int] | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemoryPhaseComparison:
    """Decompose candidate-vs-baseline phase memory using four points.

    Raises ValueError when both phases come from the same run. The cross-run
    ``start_gap``/``end_gap`` legs silently drop requested ``events`` and
    ``lifetimes`` attribution; only the within-run change legs carry them.
    """

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
        display=options.display,
    )
    baseline_pool_universe = {
        **baseline.start.pool_stats,
        **baseline.end.pool_stats,
    }
    candidate_pool_universe = {
        **candidate.start.pool_stats,
        **candidate.end.pool_stats,
    }
    start_gap = _compare_independent_states(
        baseline.start,
        candidate.start,
        pool_mapping=pool_mapping,
        device_mapping=device_mapping,
        options=cross_options,
        reference_pool_universe=baseline_pool_universe,
        candidate_pool_universe=candidate_pool_universe,
        reference_stack_index=baseline_start_stacks,
        candidate_stack_index=candidate_start_stacks,
    )
    end_gap = _compare_independent_states(
        baseline.end,
        candidate.end,
        pool_mapping=pool_mapping,
        device_mapping=device_mapping,
        options=cross_options,
        reference_pool_universe=baseline_pool_universe,
        candidate_pool_universe=candidate_pool_universe,
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
    device_decomposition = _device_phase_decomposition(
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
        device_decomposition=device_decomposition,
        display_stack_depth=options.display.stack_depth,
        display_limit=options.display.limit,
    )


def _compare_same_run(
    run: MemoryRun,
    reference: MemoryPoint,
    candidate: MemoryPoint,
    options: MemoryAttributionOptions,
) -> MemoryPointComparison:
    states = (
        run.points[reference.index : candidate.index + 1]
        if options.lifetimes
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
    _event_windows: Sequence[Sequence[EventWindow]] | None = None,
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
        _event_windows=_event_windows,
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
    _event_windows: Sequence[Sequence[EventWindow]] | None = None,
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
    lifecycle_by_pool: dict[MemoryPoolKey, list[MemoryLifecycleDelta]] = {}
    for key, lifecycle in observation_lifecycle.items():
        lifecycle_by_pool.setdefault(key.pool_key, []).append(lifecycle)
    pool_lifecycle = {
        pool_key: MemoryLifecycleDelta.combine(values)
        for pool_key, values in lifecycle_by_pool.items()
    }

    pool_comparisons = tuple(
        MemoryPoolComparison(
            reference_key=pool_key,
            candidate_key=pool_key,
            match=match,
            reference=reference_pool_stats.get(pool_key, MemoryStats()),
            candidate=candidate_pool_stats.get(pool_key, MemoryStats()),
            delta=MemoryStatsDelta.between(
                reference_pool_stats.get(pool_key, MemoryStats()),
                candidate_pool_stats.get(pool_key, MemoryStats()),
            ),
            lifecycle=pool_lifecycle.get(pool_key, MemoryLifecycleDelta()),
        )
        for pool_key in sorted(
            set(reference_pool_stats) | set(candidate_pool_stats),
            key=lambda item: item.label,
        )
    )
    observation_comparisons = tuple(
        MemoryObservationComparison(
            reference_key=key,
            candidate_key=key,
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
            key=lambda item: item.label,
        )
    )

    warnings = [*reference.warnings, *candidate.warnings]
    lifecycle_confidence: Literal["approximate", "exact"] = (
        "exact"
        if _lifecycle_identity_is_exact(reference_segments)
        and _lifecycle_identity_is_exact(candidate_segments)
        else "approximate"
    )
    if lifecycle_confidence == "approximate":
        warnings.append(
            "address lifecycle is approximate because allocator addresses are missing"
        )
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
        _check_stack_coverage(reference_coverage, candidate_coverage, warnings)
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

    interval_event_windows = _event_windows
    allocator_events = ()
    events_available = False
    events_complete = True
    if options.events:
        entries = []
        events_available = True
        event_causes: set[str] = set()
        if interval_event_windows is None and lifetime_run is None:
            event_devices = _segment_device_indices(
                reference_segments, candidate_segments
            )
            computed_windows = []
            for reference_block, candidate_block in zip(
                interval_views, interval_views[1:]
            ):
                reference_state = reference_block.state
                candidate_state = candidate_block.state
                assert isinstance(reference_state, MemoryProbeSnapshot)
                assert isinstance(candidate_state, MemoryProbeSnapshot)
                current_snapshot = candidate_block.raw
                interval_devices = tuple(
                    sorted(
                        set(event_devices) | set(trace_device_indices(current_snapshot))
                    )
                )
                computed_windows.append(
                    _extract_snapshot_event_windows(
                        current_snapshot,
                        devices=interval_devices,
                        previous_boundary_recorded=reference_state._boundary_recorded,
                        current_boundary_recorded=candidate_state._boundary_recorded,
                        start_marker=reference_state.boundary_marker,
                        end_marker=candidate_state.boundary_marker,
                        start_label=_state_label(reference_state),
                        end_label=_state_label(candidate_state),
                        start_index=reference_state.snapshot_index,
                        end_index=candidate_state.snapshot_index,
                    )
                )
            interval_event_windows = tuple(computed_windows)
        elif interval_event_windows is None:
            assert isinstance(reference, MemoryPoint)
            assert isinstance(candidate, MemoryPoint)
            interval_event_windows = tuple(
                point._event_windows()
                for point in lifetime_run.points[
                    reference.index + 1 : candidate.index + 1
                ]
            )
        assert interval_event_windows is not None
        for windows in interval_event_windows:
            if not windows:
                event_causes.add("disabled")
                events_available = False
                events_complete = False
                warnings.append("allocator event history is unavailable")
            for window in windows:
                entries.extend(window.entries)
                events_available = events_available and window.available
                events_complete = events_complete and window.complete
                if window.cause is not None:
                    event_causes.add(window.cause)
                warnings.extend(window.warnings)
        if "invalid_boundary_order" in event_causes:
            raise MemoryReconciliationError(
                "allocator event boundary order is inconsistent for the compared "
                "interval; the recorded markers were interleaved or the input snapshots are corrupted"
            )
        if "boundary_unavailable" in event_causes:
            raise MemoryHistoryBoundaryError(
                "an allocator event boundary could not be recorded for the compared "
                "interval; record points outside CUDA Graph capture or use a PyTorch "
                "build with memory metadata support"
            )
        if "disabled" in event_causes or not events_available:
            raise MemoryHistoryDisabledError(
                "allocator event history is unavailable for the compared "
                "interval; enable torch.cuda.memory._record_memory_history() "
                "before the allocations of interest"
            )
        if "truncated" in event_causes or not events_complete:
            raise MemoryHistoryTruncatedError(
                "allocator event history was truncated for the compared "
                "interval; enable _record_memory_history() before the first "
                "analyzed point and keep it enabled, raise "
                "_record_memory_history(max_entries=...), or record points "
                "more frequently"
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
                _event_windows=interval_event_windows,
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
                _allocator_states=tuple(view.raw for view in interval_views),
                _event_windows=interval_event_windows,
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
        device_comparisons=compare_device_memory(
            reference.device_memory,
            candidate.device_memory,
            reference.pool_stats,
            candidate.pool_stats,
            default_match=match,
        ),
        allocation_stack_comparisons=stack_deltas,
        allocation_stack_observation_comparisons=stack_detail_deltas,
        reference_stack_coverage=reference_coverage,
        candidate_stack_coverage=candidate_coverage,
        allocator_events=allocator_events,
        allocation_lifetimes=allocation_lifetimes,
        attribution_status=_attribution_status(
            options,
            reference_coverage=reference_coverage,
            candidate_coverage=candidate_coverage,
            events_available=events_available,
            events_complete=events_complete,
            allocation_lifetimes=allocation_lifetimes,
        ),
        lifecycle_available=True,
        lifecycle_confidence=lifecycle_confidence,
        display_stack_depth=options.display.stack_depth,
        display_limit=options.display.limit,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _load_interval_views(
    states: Sequence[_MemoryState],
    options: MemoryAttributionOptions,
) -> tuple[_MemoryStateView, ...]:
    return tuple(_load_state(state, options) for state in states)


def _state_label(state: _MemoryState) -> str:
    if isinstance(state, MemoryProbeSnapshot):
        return f"{state.probe_name}@snapshot-{state.snapshot_index}"
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


def _resolve_device_pairs(
    reference: _MemoryState,
    candidate: _MemoryState,
    reference_pools: Mapping[MemoryPoolKey, MemoryStats],
    candidate_pools: Mapping[MemoryPoolKey, MemoryStats],
    pool_mapping: Mapping[MemoryPoolKey, MemoryPoolKey] | None,
    device_mapping: Mapping[int, int] | None,
) -> tuple[tuple[int | None, int | None, DeviceMatchKind], ...]:
    reference_devices = {
        *reference.device_memory,
        *(key.device_index for key in reference_pools),
    }
    candidate_devices = {
        *candidate.device_memory,
        *(key.device_index for key in candidate_pools),
    }
    pairs: dict[int, tuple[int, DeviceMatchKind]] = {}
    claimed_candidates: dict[int, int] = {}

    def add_pair(
        reference_device: int,
        candidate_device: int,
        match: DeviceMatchKind,
        *,
        source: str,
    ) -> None:
        existing = pairs.get(reference_device)
        if existing is not None and existing[0] != candidate_device:
            raise ValueError(
                f"{source} maps reference device {reference_device} to candidate "
                f"device {candidate_device}, conflicting with device {existing[0]}"
            )
        claimed = claimed_candidates.get(candidate_device)
        if claimed is not None and claimed != reference_device:
            raise ValueError(
                f"{source} maps candidate device {candidate_device} from reference "
                f"device {reference_device}, conflicting with device {claimed}"
            )
        if existing is None:
            pairs[reference_device] = (candidate_device, match)
            claimed_candidates[candidate_device] = reference_device

    for reference_device, candidate_device in (device_mapping or {}).items():
        if type(reference_device) is not int or reference_device < 0:
            raise TypeError("device_mapping keys must be non-negative integers")
        if type(candidate_device) is not int or candidate_device < 0:
            raise TypeError("device_mapping values must be non-negative integers")
        if reference_device not in reference_devices:
            raise ValueError(
                f"mapped reference device {reference_device} does not exist"
            )
        if candidate_device not in candidate_devices:
            raise ValueError(
                f"mapped candidate device {candidate_device} does not exist"
            )
        add_pair(
            reference_device,
            candidate_device,
            "mapped",
            source="device_mapping",
        )

    for reference_pool, candidate_pool in (pool_mapping or {}).items():
        if not isinstance(reference_pool, MemoryPoolKey) or not isinstance(
            candidate_pool, MemoryPoolKey
        ):
            raise TypeError("pool_mapping keys and values must be MemoryPoolKey")
        if reference_pool not in reference_pools:
            raise ValueError(
                f"mapped reference pool {reference_pool.label} does not exist"
            )
        if candidate_pool not in candidate_pools:
            raise ValueError(
                f"mapped candidate pool {candidate_pool.label} does not exist"
            )
        if reference_pool.device_index != candidate_pool.device_index:
            add_pair(
                reference_pool.device_index,
                candidate_pool.device_index,
                "pool_mapping",
                source="pool_mapping",
            )

    for device in sorted(reference_devices & candidate_devices):
        if device not in pairs and device not in claimed_candidates:
            add_pair(device, device, "same_index", source="same-index pairing")

    result: list[tuple[int | None, int | None, DeviceMatchKind]] = [
        (reference_device, candidate_device, match)
        for reference_device, (candidate_device, match) in sorted(pairs.items())
    ]
    result.extend(
        (device, None, "reference_only")
        for device in sorted(reference_devices - set(pairs))
    )
    result.extend(
        (None, device, "candidate_only")
        for device in sorted(candidate_devices - set(claimed_candidates))
    )
    return tuple(result)


def _validate_pool_mapping(
    reference_pool_stats: Mapping[MemoryPoolKey, MemoryStats],
    candidate_pool_stats: Mapping[MemoryPoolKey, MemoryStats],
    pool_mapping: Mapping[MemoryPoolKey, MemoryPoolKey] | None,
    device_pairs: Sequence[tuple[int | None, int | None, DeviceMatchKind]],
) -> dict[MemoryPoolKey, tuple[MemoryPoolKey, Literal["default", "mapped"]]]:
    paired_devices = {
        reference_device: candidate_device
        for reference_device, candidate_device, _match in device_pairs
        if reference_device is not None and candidate_device is not None
    }
    result: dict[MemoryPoolKey, tuple[MemoryPoolKey, Literal["default", "mapped"]]] = {}
    for reference_device, candidate_device in paired_devices.items():
        reference_pool = MemoryPoolKey(reference_device, DEFAULT_POOL_ID)
        candidate_pool = MemoryPoolKey(candidate_device, DEFAULT_POOL_ID)
        if (
            reference_pool in reference_pool_stats
            and candidate_pool in candidate_pool_stats
        ):
            result[reference_pool] = (candidate_pool, "default")

    candidate_pools = {candidate_pool for candidate_pool, _match in result.values()}
    for reference_pool, candidate_pool in (pool_mapping or {}).items():
        if (
            reference_pool.pool_id == DEFAULT_POOL_ID
            or candidate_pool.pool_id == DEFAULT_POOL_ID
        ) and (
            reference_pool.pool_id != DEFAULT_POOL_ID
            or candidate_pool.pool_id != DEFAULT_POOL_ID
        ):
            raise ValueError("the default pool may only map to a default pool")
        if (
            paired_devices.get(reference_pool.device_index)
            != candidate_pool.device_index
        ):
            raise ValueError(
                f"pool mapping {reference_pool.label} -> {candidate_pool.label} "
                "conflicts with the resolved device mapping"
            )
        existing = result.get(reference_pool)
        if existing is not None:
            if existing[0] == candidate_pool and (
                reference_pool.pool_id == DEFAULT_POOL_ID
                and candidate_pool.pool_id == DEFAULT_POOL_ID
            ):
                continue
            raise ValueError(
                f"reference pool {reference_pool.label} is mapped more than once"
            )
        if candidate_pool in candidate_pools:
            raise ValueError(
                f"candidate pool {candidate_pool.label} is mapped more than once"
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
                reference_key=key,
                candidate_key=None,
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
                reference_key=None,
                candidate_key=key,
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
                (item.candidate_key or item.reference_key).label,
                item.match,
            ),
        )
    )


def _attribution_status(
    options: MemoryAttributionOptions,
    *,
    reference_coverage: AllocationStackCoverage | None,
    candidate_coverage: AllocationStackCoverage | None,
    events_available: bool,
    events_complete: bool,
    allocation_lifetimes: object | None,
) -> MemoryAttributionStatus:
    if options.stacks:
        coverages = tuple(
            item
            for item in (reference_coverage, candidate_coverage)
            if item is not None
        )
        stacks_available = len(coverages) == 2 and all(
            item.active_bytes == 0 or item.attributed_bytes > 0 for item in coverages
        )
        stacks_complete = stacks_available and all(
            item.unattributed_bytes == 0 for item in coverages
        )
        stack_status = MemoryEvidenceStatus(
            requested=True,
            available=stacks_available,
            complete=stacks_complete,
        )
    else:
        stack_status = MemoryEvidenceStatus.not_requested()

    event_status = (
        MemoryEvidenceStatus(
            requested=True,
            available=events_available,
            complete=events_available and events_complete,
        )
        if options.events
        else MemoryEvidenceStatus.not_requested()
    )
    if options.lifetimes:
        assert allocation_lifetimes is not None
        lifetime_status = MemoryEvidenceStatus(
            requested=True,
            available=True,
            complete=bool(
                getattr(allocation_lifetimes, "history_available")
                and getattr(allocation_lifetimes, "history_complete")
            ),
        )
    else:
        lifetime_status = MemoryEvidenceStatus.not_requested()
    return MemoryAttributionStatus(
        stacks=stack_status,
        events=event_status,
        lifetimes=lifetime_status,
    )


def _check_stack_coverage(
    reference: AllocationStackCoverage,
    candidate: AllocationStackCoverage,
    warnings: list[str],
) -> None:
    unavailable = [
        label
        for label, coverage in (
            ("reference", reference),
            ("candidate", candidate),
        )
        if coverage.active_bytes > 0 and coverage.attributed_bytes == 0
    ]
    if unavailable:
        raise MemoryHistoryDisabledError(
            "allocation stack history is unavailable for active bytes at: "
            + ", ".join(unavailable)
            + "; "
            "enable torch.cuda.memory._record_memory_history(...) with stack "
            "context before the allocations of interest"
        )
    if reference.unattributed_bytes or candidate.unattributed_bytes:
        warnings.append(
            "allocation stack coverage is incomplete: "
            f"reference={reference.ratio:.1%} "
            f"({reference.unattributed_bytes} unattributed bytes), "
            f"candidate={candidate.ratio:.1%} "
            f"({candidate.unattributed_bytes} unattributed bytes); "
            "attributed rows remain exact and unframed active bytes are grouped "
            "under <unattributed>"
        )


def _device_phase_decomposition(
    baseline_change: MemoryPointComparison,
    candidate_change: MemoryPointComparison,
    start_gap: MemoryPointComparison,
    end_gap: MemoryPointComparison,
) -> tuple[MemoryDevicePhaseDecomposition, ...]:
    baseline_by_device = {
        (item.reference_device_index, item.candidate_device_index): item
        for item in baseline_change.device_comparisons
    }
    candidate_by_device = {
        (item.reference_device_index, item.candidate_device_index): item
        for item in candidate_change.device_comparisons
    }
    end_gap_by_pair = {
        (item.reference_device_index, item.candidate_device_index): item
        for item in end_gap.device_comparisons
    }
    rows: list[MemoryDevicePhaseDecomposition] = []
    for start in start_gap.device_comparisons:
        baseline_device = start.reference_device_index
        candidate_device = start.candidate_device_index
        if baseline_device is None or candidate_device is None:
            continue
        comparisons = (
            baseline_by_device.get((baseline_device, baseline_device)),
            candidate_by_device.get((candidate_device, candidate_device)),
            start,
            end_gap_by_pair.get((baseline_device, candidate_device)),
        )
        if any(item is None for item in comparisons):
            continue
        complete = tuple(cast(MemoryDeviceComparison, item) for item in comparisons)
        for metric in DEVICE_MEMORY_METRICS:
            # Sample-derived metrics report None deltas when any leg lacks a
            # device sample; allocator rollups stay available regardless.
            deltas = tuple(getattr(item, f"delta_{metric}") for item in complete)
            if any(value is None for value in deltas):
                continue
            baseline_delta, candidate_delta, start_delta, end_delta = (
                int(value) for value in deltas
            )
            rows.append(
                MemoryDevicePhaseDecomposition(
                    baseline_device_index=baseline_device,
                    candidate_device_index=candidate_device,
                    components=MemoryPhaseComponents(
                        metric=metric,
                        start_gap_bytes=start_delta,
                        baseline_change_bytes=baseline_delta,
                        candidate_change_bytes=candidate_delta,
                        end_gap_bytes=end_delta,
                    ),
                )
            )
    return tuple(rows)


def _phase_decomposition(
    baseline_change: MemoryPointComparison,
    candidate_change: MemoryPointComparison,
    start_gap: MemoryPointComparison,
    end_gap: MemoryPointComparison,
) -> tuple[MemoryPoolPhaseDecomposition, ...]:
    baseline_change_by_pool = {
        item.reference_key: item
        for item in baseline_change.pool_comparisons
        if item.reference_key is not None
    }
    candidate_change_by_pool = {
        item.reference_key: item
        for item in candidate_change.pool_comparisons
        if item.reference_key is not None
    }
    end_gap_by_pair = {
        (item.reference_key, item.candidate_key): item
        for item in end_gap.pool_comparisons
    }
    rows: list[MemoryPoolPhaseDecomposition] = []
    for start_gap_comparison in start_gap.pool_comparisons:
        if start_gap_comparison.match not in {"default", "mapped"}:
            continue
        baseline_key = start_gap_comparison.reference_key
        candidate_key = start_gap_comparison.candidate_key
        if baseline_key is None or candidate_key is None:
            continue
        baseline_item = baseline_change_by_pool.get(baseline_key)
        candidate_item = candidate_change_by_pool.get(candidate_key)
        end_item = end_gap_by_pair.get((baseline_key, candidate_key))
        if baseline_item is None or candidate_item is None or end_item is None:
            continue
        for metric in PHASE_METRICS:
            rows.append(
                MemoryPoolPhaseDecomposition(
                    baseline_key=baseline_key,
                    candidate_key=candidate_key,
                    components=MemoryPhaseComponents(
                        metric=metric,
                        start_gap_bytes=int(
                            getattr(start_gap_comparison.delta, metric)
                        ),
                        baseline_change_bytes=int(getattr(baseline_item.delta, metric)),
                        candidate_change_bytes=int(
                            getattr(candidate_item.delta, metric)
                        ),
                        end_gap_bytes=int(getattr(end_item.delta, metric)),
                    ),
                )
            )
    return tuple(rows)


def _scope_phase_decomposition(
    baseline_change: MemoryPointComparison,
    candidate_change: MemoryPointComparison,
    start_gap: MemoryPointComparison,
    end_gap: MemoryPointComparison,
) -> tuple[MemoryAllocatorScopePhaseDecomposition, ...]:
    baseline_by_scope = {
        item.scope: item for item in baseline_change.allocator_scope_comparisons
    }
    candidate_by_scope = {
        item.scope: item for item in candidate_change.allocator_scope_comparisons
    }
    start_by_scope = {
        item.scope: item for item in start_gap.allocator_scope_comparisons
    }
    end_by_scope = {item.scope: item for item in end_gap.allocator_scope_comparisons}
    rows: list[MemoryAllocatorScopePhaseDecomposition] = []
    for scope in ALLOCATOR_SCOPES:
        for metric in PHASE_METRICS:
            rows.append(
                MemoryAllocatorScopePhaseDecomposition(
                    scope=scope,
                    components=MemoryPhaseComponents(
                        metric=metric,
                        start_gap_bytes=int(
                            getattr(start_by_scope[scope].delta, metric)
                        ),
                        baseline_change_bytes=int(
                            getattr(baseline_by_scope[scope].delta, metric)
                        ),
                        candidate_change_bytes=int(
                            getattr(candidate_by_scope[scope].delta, metric)
                        ),
                        end_gap_bytes=int(getattr(end_by_scope[scope].delta, metric)),
                    ),
                )
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
        item.reference_key.label if item.reference_key is not None else "",
        item.candidate_key.label if item.candidate_key is not None else "",
    )
