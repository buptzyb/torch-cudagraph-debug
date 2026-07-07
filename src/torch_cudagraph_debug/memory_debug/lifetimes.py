"""Offline allocation-cohort lifetime analysis for memory run bundles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING, TypeVar

from ._pool_ranges import PoolRangeIndex, build_pool_range_index
from ._stack_trace import (
    normalize_stack_frames,
    stack_fingerprint as _stack_fingerprint,
    stack_key,
)
from .errors import MemoryHistoryError
from .events import (
    extract_event_window,
    extract_event_window_from_snapshot,
)
from .allocator_snapshot import (
    ALLOCATION_LIFETIME_ACTIONS,
    ACTIVE_STATES,
    AWAITING_FREE_STATES,
    AllocatorTraceEntry,
    KNOWN_TRACE_ACTIONS,
    OWNER_ACTIVE_STATES,
    normalize_pool_id,
    normalize_snapshot,
    normalize_trace_entries,
    trace_device_indices,
)

if TYPE_CHECKING:
    from .attribution import MemoryLifetimeOptions
    from .recording import MemoryPoint, MemoryRun
    from .snapshots import MemoryProbeSnapshot
    from .reports import MemoryAllocationLifetimeAnalysis


LifetimeConfidence = Literal["event_exact", "snapshot_inferred"]
LifetimeTerminalState = Literal[
    "owner_active", "awaiting_free", "free_completed", "unknown"
]


def _cohort_id(
    device: int | None,
    pool_id: tuple[Any, ...],
    stack_fingerprint: str,
    unattributed_size: tuple[int, int] | None,
) -> str:
    payload = [device, list(pool_id), stack_fingerprint, unattributed_size]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return f"cohort-{hashlib.sha256(encoded).hexdigest()[:16]}"


@dataclass(frozen=True)
class CohortPointState:
    """Owner-active and stream-waiting state at one memory point."""

    point_index: int
    point_label: str
    owner_active_bytes: int
    owner_requested_bytes: int
    owner_active_count: int
    awaiting_free_bytes: int
    awaiting_free_requested_bytes: int
    awaiting_free_count: int

    @property
    def active_bytes(self) -> int:
        return self.owner_active_bytes + self.awaiting_free_bytes

    @property
    def requested_bytes(self) -> int:
        return self.owner_requested_bytes + self.awaiting_free_requested_bytes

    @property
    def block_count(self) -> int:
        return self.owner_active_count + self.awaiting_free_count

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "owner_active_bytes": self.owner_active_bytes,
            "owner_requested_bytes": self.owner_requested_bytes,
            "owner_active_count": self.owner_active_count,
            "awaiting_free_bytes": self.awaiting_free_bytes,
            "awaiting_free_requested_bytes": self.awaiting_free_requested_bytes,
            "awaiting_free_count": self.awaiting_free_count,
            "active_bytes": self.active_bytes,
            "requested_bytes": self.requested_bytes,
            "block_count": self.block_count,
        }

    def to_row(self, cohort_id: str) -> dict[str, object]:
        return {"cohort_id": cohort_id, **self.to_dict()}


@dataclass(frozen=True)
class CohortSizeBucket:
    """Unique allocation-instance sizes within one cohort."""

    size_bytes: int
    requested_bytes: int
    count: int
    total_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "size_bytes": self.size_bytes,
            "requested_bytes": self.requested_bytes,
            "count": self.count,
            "total_bytes": self.total_bytes,
        }

    def to_row(self, cohort_id: str) -> dict[str, object]:
        return {"cohort_id": cohort_id, **self.to_dict()}


@dataclass(frozen=True)
class CohortSizeOutcome:
    """Allocation sizes correlated with their state at the analysis end."""

    size_bytes: int
    requested_bytes: int
    terminal_state: LifetimeTerminalState
    count: int
    total_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "size_bytes": self.size_bytes,
            "requested_bytes": self.requested_bytes,
            "terminal_state": self.terminal_state,
            "count": self.count,
            "total_bytes": self.total_bytes,
        }

    def to_row(self, cohort_id: str) -> dict[str, object]:
        return {"cohort_id": cohort_id, **self.to_dict()}


@dataclass(frozen=True)
class _CohortTransition:
    start_index: int
    end_index: int
    start_label: str
    end_label: str
    stack_frames: tuple[Mapping[str, Any], ...]
    stack_fallback: str
    confidence: LifetimeConfidence
    size_bytes: int
    count: int

    @property
    def stack_key(self) -> str:
        return stack_key(self.stack_frames, fallback=self.stack_fallback)

    def display_stack(self, depth: int) -> str:
        return stack_key(self.stack_frames, depth=depth, fallback=self.stack_fallback)

    def to_dict(self) -> dict[str, object]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_label": self.start_label,
            "end_label": self.end_label,
            "stack_key": self.stack_key,
            "confidence": self.confidence,
            "stack_frames": [dict(frame) for frame in self.stack_frames],
            "size_bytes": self.size_bytes,
            "count": self.count,
        }

    def to_row(self, cohort_id: str) -> dict[str, object]:
        return {"cohort_id": cohort_id, **self.to_dict()}


@dataclass(frozen=True)
class CohortBirth(_CohortTransition):
    """Allocation births grouped by marker interval and allocation stack."""


@dataclass(frozen=True)
class CohortFreeRequest(_CohortTransition):
    """Storage free requests grouped by interval and request stack."""


@dataclass(frozen=True)
class CohortFreeCompletion(_CohortTransition):
    """Allocator reuse-ready transitions grouped by marker interval."""


@dataclass(frozen=True)
class AllocationCohort:
    """Allocation instances sharing a device, pool, and allocation stack."""

    cohort_id: str
    display_rank: int
    device: int | None
    pool_id: tuple[Any, ...]
    stack_fingerprint: str
    stack_frames: tuple[Mapping[str, Any], ...]
    streams: tuple[Any, ...]
    points: tuple[CohortPointState, ...]
    size_histogram: tuple[CohortSizeBucket, ...]
    size_outcomes: tuple[CohortSizeOutcome, ...]
    births: tuple[CohortBirth, ...]
    free_requests: tuple[CohortFreeRequest, ...]
    free_completions: tuple[CohortFreeCompletion, ...]
    peak_active_bytes: int
    peak_owner_active_bytes: int
    peak_awaiting_free_bytes: int
    event_owner_peak_bytes: int
    event_unreusable_peak_bytes: int
    peak_block_count: int
    snapshot_active_span_bytes: int
    first_seen_index: int
    first_seen_label: str
    last_seen_index: int
    last_seen_label: str
    born_bytes: int
    born_count: int
    event_exact_birth_bytes: int
    event_exact_birth_count: int
    snapshot_inferred_birth_bytes: int
    snapshot_inferred_birth_count: int
    event_exact_free_requested_bytes: int
    event_exact_free_requested_count: int
    snapshot_inferred_free_requested_bytes: int
    snapshot_inferred_free_requested_count: int
    event_exact_free_completed_bytes: int
    event_exact_free_completed_count: int
    snapshot_inferred_free_completed_bytes: int
    snapshot_inferred_free_completed_count: int
    owner_active_at_end_bytes: int
    owner_active_at_end_count: int
    awaiting_free_at_end_bytes: int
    awaiting_free_at_end_count: int

    @property
    def stack_key(self) -> str:
        return stack_key(self.stack_frames)

    def display_stack(self, depth: int) -> str:
        return stack_key(self.stack_frames, depth=depth)

    @property
    def free_requested_bytes(self) -> int:
        return (
            self.event_exact_free_requested_bytes
            + self.snapshot_inferred_free_requested_bytes
        )

    @property
    def free_completed_bytes(self) -> int:
        return (
            self.event_exact_free_completed_bytes
            + self.snapshot_inferred_free_completed_bytes
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cohort_id": self.cohort_id,
            "display_rank": self.display_rank,
            "device": self.device,
            "pool_id": list(self.pool_id),
            "stack_key": self.stack_key,
            "stack_fingerprint": self.stack_fingerprint,
            "stack_frames": [dict(frame) for frame in self.stack_frames],
            "streams": list(self.streams),
            "peak_active_bytes": self.peak_active_bytes,
            "peak_owner_active_bytes": self.peak_owner_active_bytes,
            "peak_awaiting_free_bytes": self.peak_awaiting_free_bytes,
            "event_owner_peak_bytes": self.event_owner_peak_bytes,
            "event_unreusable_peak_bytes": self.event_unreusable_peak_bytes,
            "peak_block_count": self.peak_block_count,
            "snapshot_active_span_bytes": self.snapshot_active_span_bytes,
            "first_seen_index": self.first_seen_index,
            "first_seen_label": self.first_seen_label,
            "last_seen_index": self.last_seen_index,
            "last_seen_label": self.last_seen_label,
            "born_bytes": self.born_bytes,
            "born_count": self.born_count,
            "event_exact_birth_bytes": self.event_exact_birth_bytes,
            "event_exact_birth_count": self.event_exact_birth_count,
            "snapshot_inferred_birth_bytes": self.snapshot_inferred_birth_bytes,
            "snapshot_inferred_birth_count": self.snapshot_inferred_birth_count,
            "free_requested_bytes": self.free_requested_bytes,
            "event_exact_free_requested_bytes": self.event_exact_free_requested_bytes,
            "event_exact_free_requested_count": self.event_exact_free_requested_count,
            "snapshot_inferred_free_requested_bytes": self.snapshot_inferred_free_requested_bytes,
            "snapshot_inferred_free_requested_count": self.snapshot_inferred_free_requested_count,
            "free_completed_bytes": self.free_completed_bytes,
            "event_exact_free_completed_bytes": self.event_exact_free_completed_bytes,
            "event_exact_free_completed_count": self.event_exact_free_completed_count,
            "snapshot_inferred_free_completed_bytes": self.snapshot_inferred_free_completed_bytes,
            "snapshot_inferred_free_completed_count": self.snapshot_inferred_free_completed_count,
            "owner_active_at_end_bytes": self.owner_active_at_end_bytes,
            "owner_active_at_end_count": self.owner_active_at_end_count,
            "awaiting_free_at_end_bytes": self.awaiting_free_at_end_bytes,
            "awaiting_free_at_end_count": self.awaiting_free_at_end_count,
            "points": [item.to_dict() for item in self.points],
            "size_histogram": [item.to_dict() for item in self.size_histogram],
            "size_outcomes": [item.to_dict() for item in self.size_outcomes],
            "births": [item.to_dict() for item in self.births],
            "free_requests": [item.to_dict() for item in self.free_requests],
            "free_completions": [item.to_dict() for item in self.free_completions],
        }

    def to_row(self) -> dict[str, object]:
        return {
            "cohort_id": self.cohort_id,
            "display_rank": self.display_rank,
            "device": "unknown" if self.device is None else self.device,
            "pool_id": "pool[" + ",".join(str(item) for item in self.pool_id) + "]",
            "streams": ";".join(f"stream[{item}]" for item in self.streams),
            "stack_key": self.stack_key,
            "stack_fingerprint": self.stack_fingerprint,
            "peak_active_bytes": self.peak_active_bytes,
            "peak_owner_active_bytes": self.peak_owner_active_bytes,
            "peak_awaiting_free_bytes": self.peak_awaiting_free_bytes,
            "event_owner_peak_bytes": self.event_owner_peak_bytes,
            "event_unreusable_peak_bytes": self.event_unreusable_peak_bytes,
            "peak_block_count": self.peak_block_count,
            "snapshot_active_span_bytes": self.snapshot_active_span_bytes,
            "first_seen_index": self.first_seen_index,
            "first_seen_label": self.first_seen_label,
            "last_seen_index": self.last_seen_index,
            "last_seen_label": self.last_seen_label,
            "born_bytes": self.born_bytes,
            "born_count": self.born_count,
            "event_exact_birth_bytes": self.event_exact_birth_bytes,
            "event_exact_birth_count": self.event_exact_birth_count,
            "snapshot_inferred_birth_bytes": self.snapshot_inferred_birth_bytes,
            "snapshot_inferred_birth_count": self.snapshot_inferred_birth_count,
            "free_requested_bytes": self.free_requested_bytes,
            "event_exact_free_requested_bytes": self.event_exact_free_requested_bytes,
            "event_exact_free_requested_count": self.event_exact_free_requested_count,
            "snapshot_inferred_free_requested_bytes": self.snapshot_inferred_free_requested_bytes,
            "snapshot_inferred_free_requested_count": self.snapshot_inferred_free_requested_count,
            "free_completed_bytes": self.free_completed_bytes,
            "event_exact_free_completed_bytes": self.event_exact_free_completed_bytes,
            "event_exact_free_completed_count": self.event_exact_free_completed_count,
            "snapshot_inferred_free_completed_bytes": self.snapshot_inferred_free_completed_bytes,
            "snapshot_inferred_free_completed_count": self.snapshot_inferred_free_completed_count,
            "owner_active_at_end_bytes": self.owner_active_at_end_bytes,
            "owner_active_at_end_count": self.owner_active_at_end_count,
            "awaiting_free_at_end_bytes": self.awaiting_free_at_end_bytes,
            "awaiting_free_at_end_count": self.awaiting_free_at_end_count,
        }


@dataclass(frozen=True)
class _BlockObservation:
    point_index: int
    point_label: str
    ordinal: int
    device: int | None
    pool_id: tuple[Any, ...]
    stream: Any
    address: int | None
    size_bytes: int
    requested_bytes: int
    state: str
    stack_frames: tuple[Mapping[str, Any], ...]


@dataclass
class _AllocationInstance:
    device: int | None
    pool_id: tuple[Any, ...]
    stream: Any
    address: int | None
    size_bytes: int
    requested_bytes: int
    stack_frames: tuple[Mapping[str, Any], ...]
    observations: dict[int, _BlockObservation] = field(default_factory=dict)
    birth: CohortBirth | None = None
    free_request: CohortFreeRequest | None = None
    free_completion: CohortFreeCompletion | None = None
    birth_order: int | None = None
    free_request_order: int | None = None
    free_completion_order: int | None = None


@dataclass(frozen=True)
class _IntervalHistory:
    start: Any
    end: Any
    entries: tuple[AllocatorTraceEntry, ...]
    pool_ranges: PoolRangeIndex | None
    available: bool
    complete: bool
    warnings: tuple[str, ...]


def analyze_allocation_lifetimes(
    run: MemoryRun,
    *,
    start: MemoryPoint,
    end: MemoryPoint,
    active_at: MemoryPoint | None,
    born_between: tuple[MemoryPoint, MemoryPoint] | None,
    options: MemoryLifetimeOptions,
    _raw_snapshots: Sequence[Any] | None = None,
) -> MemoryAllocationLifetimeAnalysis:
    """Build allocation cohorts for a range in one recorded run."""

    return _analyze_allocation_lifetimes(
        run.points[start.index : end.index + 1],
        source_kind="run",
        source_id=run.run_id,
        source_name=run.name,
        start=start,
        end=end,
        active_at=active_at,
        born_between=born_between,
        options=options,
        raw_snapshots=_raw_snapshots,
    )


def analyze_probe_snapshot_lifetimes(
    reference: MemoryProbeSnapshot,
    candidate: MemoryProbeSnapshot,
    *,
    options: MemoryLifetimeOptions,
    _raw_snapshots: Sequence[Any] | None = None,
) -> MemoryAllocationLifetimeAnalysis:
    """Build allocation cohorts for two ordered snapshots from one Probe."""

    if reference.probe_id != candidate.probe_id:
        raise ValueError("probe lifetime endpoints must belong to one MemoryProbe")
    if candidate.index <= reference.index:
        raise ValueError("candidate snapshot must follow reference snapshot")
    return _analyze_allocation_lifetimes(
        (reference, candidate),
        source_kind="probe",
        source_id=reference.probe_id,
        source_name=reference.probe_name,
        start=reference,
        end=candidate,
        active_at=None,
        born_between=None,
        options=options,
        raw_snapshots=_raw_snapshots,
    )


def _analyze_allocation_lifetimes(
    states: Sequence[Any],
    *,
    source_kind: Literal["run", "probe"],
    source_id: str,
    source_name: str,
    start: Any,
    end: Any,
    active_at: Any | None,
    born_between: tuple[Any, Any] | None,
    options: MemoryLifetimeOptions,
    raw_snapshots: Sequence[Any] | None,
) -> MemoryAllocationLifetimeAnalysis:
    """Build an offline allocation-cohort lifetime report."""

    from .reports import MemoryAllocationLifetimeAnalysis

    points = tuple(states)
    observations, histories = _scan_points(
        points,
        events=options.events,
        raw_snapshots=raw_snapshots,
    )
    warnings = [warning for point in points for warning in point.warnings]
    if options.events:
        for history in histories:
            warnings.extend(history.warnings)
            unknown_actions = sorted(
                {entry.action for entry in history.entries} - KNOWN_TRACE_ACTIONS
            )
            if unknown_actions:
                warnings.append(f"unknown allocator actions: {unknown_actions}")
        if any(not item.available or not item.complete for item in histories):
            message = (
                "allocator event history is unavailable or incomplete; "
                "allocation lifetimes include snapshot-inferred transitions"
            )
            if options.on_missing == "error":
                raise MemoryHistoryError(message)
            warnings.append(message)

    instances, replay_warnings = _track_instances(points, observations, histories)
    warnings.extend(replay_warnings)
    if active_at is not None:
        instances = [item for item in instances if active_at.index in item.observations]
    elif born_between is not None:
        born_start, born_end = born_between
        instances = [
            item
            for item in instances
            if item.birth is not None
            and item.birth.start_index >= born_start.index
            and item.birth.end_index <= born_end.index
        ]
        if not options.events:
            warnings.append(
                "born-between analysis is snapshot-inferred because allocator "
                "events were disabled; transient allocations may be missing"
            )
        elif any(not item.available or not item.complete for item in histories):
            warnings.append(
                "born-between analysis fell back to snapshot-inferred births where "
                "event history was unavailable or incomplete; transient allocations "
                "may be missing"
            )
    total_instance_bytes = sum(item.size_bytes for item in instances)
    attributed_instance_bytes = sum(
        item.size_bytes for item in instances if item.stack_frames
    )
    if attributed_instance_bytes < total_instance_bytes:
        message = (
            "allocation stack coverage is incomplete for lifetime cohorts: "
            f"{attributed_instance_bytes}/{total_instance_bytes} bytes attributed; "
            "enable torch.cuda.memory._record_memory_history(...) before allocations"
        )
        if options.on_missing == "error":
            raise MemoryHistoryError(message)
        warnings.append(message)

    cohorts = _cohorts(
        instances,
        points,
        active_at=active_at,
        born_between=born_between,
    )
    return MemoryAllocationLifetimeAnalysis(
        source_kind=source_kind,
        source_id=source_id,
        source_name=source_name,
        start=start,
        end=end,
        active_at=active_at,
        born_between=born_between,
        cohorts=cohorts,
        history_requested=options.events,
        history_available=options.events and all(item.available for item in histories),
        history_complete=options.events and all(item.complete for item in histories),
        total_instance_bytes=total_instance_bytes,
        attributed_instance_bytes=attributed_instance_bytes,
        display_stack_depth=options.stack_depth,
        display_limit=options.limit,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _state_label(state: Any) -> str:
    label = getattr(state, "label", None)
    if label is not None:
        return str(label)
    return f"{state.probe_name}@snapshot-{state.index}"


def _scan_points(
    points: Sequence[Any],
    *,
    events: bool,
    raw_snapshots: Sequence[Any] | None,
) -> tuple[
    dict[int, tuple[_BlockObservation, ...]],
    tuple[_IntervalHistory, ...],
]:
    observations: dict[int, tuple[_BlockObservation, ...]] = {}
    histories: list[_IntervalHistory] = []
    previous: Any | None = None
    previous_segments: tuple[Mapping[str, Any], ...] | None = None

    if raw_snapshots is not None and len(raw_snapshots) != len(points):
        raise ValueError("raw snapshot count must match lifetime points")
    snapshots = (
        iter(raw_snapshots)
        if raw_snapshots is not None
        else (point.raw_snapshot() for point in points)
    )
    for point, snapshot in zip(points, snapshots):
        segments = normalize_snapshot(snapshot)
        observations[point.index] = _active_blocks_from_segments(point, segments)
        if previous is None or previous_segments is None:
            previous = point
            previous_segments = segments
            continue

        if events:
            devices = sorted(
                {
                    int(segment["device"])
                    for segment in (*previous_segments, *segments)
                    if segment.get("device") is not None
                }
                | set(trace_device_indices(snapshot))
            )
            if devices:
                windows = [
                    extract_event_window_from_snapshot(
                        snapshot,
                        device_index=device,
                        start_marker=previous.boundary_marker,
                        end_marker=point.boundary_marker,
                        start_label=f"{_state_label(previous)} on device {device}",
                    )
                    for device in devices
                ]
            else:
                windows = [
                    extract_event_window(
                        normalize_trace_entries(snapshot),
                        start_marker=previous.boundary_marker,
                        end_marker=point.boundary_marker,
                        start_label=_state_label(previous),
                    )
                ]
            entries = tuple(entry for window in windows for entry in window.entries)
            pool_ranges = build_pool_range_index(segments, previous_segments)
            available = bool(windows) and all(window.available for window in windows)
            complete = bool(windows) and all(window.complete for window in windows)
            warnings = tuple(
                warning for window in windows for warning in window.warnings
            )
        else:
            entries = ()
            pool_ranges = None
            available = False
            complete = False
            warnings = ()

        histories.append(
            _IntervalHistory(
                start=previous,
                end=point,
                entries=entries,
                pool_ranges=pool_ranges,
                available=available,
                complete=complete,
                warnings=warnings,
            )
        )
        previous = point
        previous_segments = segments

    return observations, tuple(histories)


def _active_blocks_from_segments(
    point: Any,
    segments: Sequence[Mapping[str, Any]],
) -> tuple[_BlockObservation, ...]:
    rows: list[_BlockObservation] = []
    ordinal = 0
    for segment in segments:
        device_value = segment.get("device")
        device = int(device_value) if device_value is not None else None
        pool_id = normalize_pool_id(segment.get("segment_pool_id"))
        stream = segment.get("stream")
        for block in segment.get("blocks", ()):
            if str(block.get("state")) not in ACTIVE_STATES:
                continue
            address_value = block.get("address")
            address = int(address_value) if address_value is not None else None
            rows.append(
                _BlockObservation(
                    point_index=point.index,
                    point_label=_state_label(point),
                    ordinal=ordinal,
                    device=device,
                    pool_id=pool_id,
                    stream=stream,
                    address=address,
                    size_bytes=int(block.get("size", 0) or 0),
                    requested_bytes=int(block.get("requested_size", 0) or 0),
                    state=str(block.get("state")),
                    stack_frames=normalize_stack_frames(block.get("frames") or ()),
                )
            )
            ordinal += 1
    return tuple(rows)


def _track_instances(
    points: Sequence[Any],
    observations: Mapping[int, Sequence[_BlockObservation]],
    histories: Sequence[_IntervalHistory],
) -> tuple[list[_AllocationInstance], tuple[str, ...]]:
    instances: list[_AllocationInstance] = []
    current: dict[tuple[int | None, int], _AllocationInstance] = {}
    missing_address: list[_AllocationInstance] = []
    warnings: list[str] = []
    event_order = 0

    def create_from_block(
        block: _BlockObservation,
        *,
        birth: CohortBirth | None = None,
        birth_order: int | None = None,
    ) -> _AllocationInstance:
        item = _AllocationInstance(
            device=block.device,
            pool_id=block.pool_id,
            stream=block.stream,
            address=block.address,
            size_bytes=block.size_bytes,
            requested_bytes=block.requested_bytes,
            stack_frames=block.stack_frames,
            observations={block.point_index: block},
            birth=birth,
            birth_order=birth_order,
        )
        instances.append(item)
        return item

    def infer_free_request(
        history: _IntervalHistory, instance: _AllocationInstance
    ) -> None:
        nonlocal event_order
        if instance.free_request is not None:
            return
        event_order += 1
        instance.free_request = _transition_from_snapshot(
            CohortFreeRequest, history, instance
        )
        instance.free_request_order = event_order

    def infer_free_completion(
        history: _IntervalHistory, instance: _AllocationInstance
    ) -> None:
        nonlocal event_order
        infer_free_request(history, instance)
        if instance.free_completion is not None:
            return
        event_order += 1
        instance.free_completion = _transition_from_snapshot(
            CohortFreeCompletion, history, instance
        )
        instance.free_completion_order = event_order

    for block in observations.get(points[0].index, ()):
        item = create_from_block(block)
        if block.address is None:
            missing_address.append(item)
        else:
            current[(block.device, block.address)] = item

    for history in histories:
        for entry in history.entries:
            event_order += 1
            if entry.addr is None or entry.action not in ALLOCATION_LIFETIME_ACTIONS:
                continue
            if entry.action == "alloc" and entry.size_bytes == 0:
                continue
            key, instance = _find_address_instance(
                current, entry.device_index, entry.addr
            )
            if entry.action == "free_requested":
                if instance is None or not _event_size_matches(entry, instance):
                    continue
                if instance.free_request is None:
                    instance.free_request = _transition_from_event(
                        CohortFreeRequest, history, instance, entry
                    )
                    instance.free_request_order = event_order
                continue
            if entry.action == "free_completed":
                if instance is None or not _event_size_matches(entry, instance):
                    warnings.append(
                        "free_completed could not be matched to an active allocation "
                        f"on device {entry.device_index} at address {entry.addr}"
                    )
                    continue
                if instance.free_request is None:
                    instance.free_request = _transition_from_snapshot(
                        CohortFreeRequest, history, instance
                    )
                    instance.free_request_order = event_order
                    event_order += 1
                    warnings.append(
                        "free_completed had no matching free_requested; inferred the request"
                    )
                if instance.free_completion is None:
                    instance.free_completion = _transition_from_event(
                        CohortFreeCompletion, history, instance, entry
                    )
                    instance.free_completion_order = event_order
                assert key is not None
                current.pop(key, None)
                continue

            if instance is not None:
                infer_free_completion(history, instance)
                warnings.append(
                    "allocation address was reused before a matching free_completed "
                    f"event was observed on device {entry.device_index} at "
                    f"address {entry.addr}"
                )
                assert key is not None
                current.pop(key, None)
            size = abs(entry.size_bytes)
            pool_id = _event_pool_id(entry, history.pool_ranges)
            frames = normalize_stack_frames(entry.frames)
            birth = CohortBirth(
                start_index=history.start.index,
                end_index=history.end.index,
                start_label=_state_label(history.start),
                end_label=_state_label(history.end),
                stack_frames=frames,
                stack_fallback="<unattributed>",
                confidence="event_exact",
                size_bytes=size,
                count=1,
            )
            item = _AllocationInstance(
                device=entry.device_index,
                pool_id=pool_id,
                stream=entry.stream,
                address=entry.addr,
                size_bytes=size,
                requested_bytes=size,
                stack_frames=frames,
                birth=birth,
                birth_order=event_order,
            )
            instances.append(item)
            current[(entry.device_index, entry.addr)] = item

        end_blocks = list(observations.get(history.end.index, ()))
        exact_blocks, fallback_blocks = _block_indexes(end_blocks)
        consumed_after: set[int] = set()
        next_current: dict[tuple[int | None, int], _AllocationInstance] = {}
        for _key, instance in current.items():
            block_index = _matching_block_index(
                exact_blocks,
                fallback_blocks,
                instance,
                consumed_after,
            )
            if block_index is None:
                infer_free_completion(history, instance)
                continue
            block = end_blocks[block_index]
            previous = _latest_observation(instance)
            generation_reused = (
                previous is not None
                and previous.state in AWAITING_FREE_STATES
                and block.state in OWNER_ACTIVE_STATES
            ) or (
                instance.free_request is not None and block.state in OWNER_ACTIVE_STATES
            )
            if generation_reused:
                infer_free_completion(history, instance)
                continue
            consumed_after.add(block_index)
            if block.state in AWAITING_FREE_STATES and instance.free_request is None:
                infer_free_request(history, instance)
            instance.observations[block.point_index] = block
            _refine_instance_from_block(instance, block)
            assert block.address is not None
            next_current[(block.device, block.address)] = instance

        for instance in missing_address:
            infer_free_completion(history, instance)
        missing_address = []

        for block_index, block in enumerate(end_blocks):
            if block_index in consumed_after:
                continue
            event_order += 1
            birth = CohortBirth(
                start_index=history.start.index,
                end_index=history.end.index,
                start_label=_state_label(history.start),
                end_label=_state_label(history.end),
                stack_frames=block.stack_frames,
                stack_fallback="<unattributed>",
                confidence="snapshot_inferred",
                size_bytes=block.size_bytes,
                count=1,
            )
            item = create_from_block(
                block,
                birth=birth,
                birth_order=event_order,
            )
            if block.state in AWAITING_FREE_STATES:
                infer_free_request(history, item)
            if block.address is None:
                missing_address.append(item)
            else:
                next_current[(block.device, block.address)] = item
        current = next_current
    return instances, tuple(dict.fromkeys(warnings))


def _find_address_instance(
    current: Mapping[tuple[int | None, int], _AllocationInstance],
    device: int | None,
    address: int,
) -> tuple[tuple[int | None, int] | None, _AllocationInstance | None]:
    exact = (device, address)
    if exact in current:
        return exact, current[exact]
    unknown_device = (None, address)
    if unknown_device in current:
        return unknown_device, current[unknown_device]
    return None, None


def _block_indexes(
    blocks: Sequence[_BlockObservation],
) -> tuple[
    Mapping[tuple[int | None, int, int], tuple[int, ...]],
    Mapping[tuple[int, int], tuple[int, ...]],
]:
    exact: defaultdict[tuple[int | None, int, int], list[int]] = defaultdict(list)
    fallback: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for index, block in enumerate(blocks):
        if block.address is None:
            continue
        exact[(block.device, block.address, block.size_bytes)].append(index)
        fallback[(block.address, block.size_bytes)].append(index)
    return (
        {key: tuple(indices) for key, indices in exact.items()},
        {key: tuple(indices) for key, indices in fallback.items()},
    )


def _matching_block_index(
    exact_blocks: Mapping[tuple[int | None, int, int], Sequence[int]],
    fallback_blocks: Mapping[tuple[int, int], Sequence[int]],
    instance: _AllocationInstance,
    consumed: set[int],
) -> int | None:
    if instance.address is None:
        return None
    exact = (
        index
        for index in exact_blocks.get(
            (instance.device, instance.address, instance.size_bytes), ()
        )
        if index not in consumed
    )
    exact_index = next(exact, None)
    if exact_index is not None:
        return exact_index
    fallback = tuple(
        index
        for index in fallback_blocks.get((instance.address, instance.size_bytes), ())
        if index not in consumed
    )
    return fallback[0] if len(fallback) == 1 else None


def _event_size_matches(
    entry: AllocatorTraceEntry, instance: _AllocationInstance
) -> bool:
    return abs(entry.size_bytes) in {
        0,
        instance.size_bytes,
        instance.requested_bytes,
    }


def _latest_observation(
    instance: _AllocationInstance,
) -> _BlockObservation | None:
    return max(
        instance.observations.values(), key=lambda item: item.point_index, default=None
    )


def _refine_instance_from_block(
    instance: _AllocationInstance, block: _BlockObservation
) -> None:
    if instance.pool_id == ("unknown",):
        instance.pool_id = block.pool_id
    if not instance.stack_frames and block.stack_frames:
        instance.stack_frames = block.stack_frames
    instance.device = block.device
    instance.stream = block.stream
    instance.address = block.address
    instance.size_bytes = block.size_bytes
    instance.requested_bytes = block.requested_bytes


def _event_pool_id(
    entry: AllocatorTraceEntry,
    pool_ranges: PoolRangeIndex | None,
) -> tuple[Any, ...]:
    if entry.pool_id is not None:
        return entry.pool_id
    if entry.addr is None or pool_ranges is None:
        return ("unknown",)
    return pool_ranges.find(entry.device_index, entry.addr) or ("unknown",)


def _transition_from_event(
    kind: type[TransitionT],
    history: _IntervalHistory,
    instance: _AllocationInstance,
    entry: AllocatorTraceEntry,
) -> TransitionT:
    frames = normalize_stack_frames(entry.frames)
    return kind(
        start_index=history.start.index,
        end_index=history.end.index,
        start_label=_state_label(history.start),
        end_label=_state_label(history.end),
        stack_frames=frames,
        stack_fallback="<unavailable>",
        confidence="event_exact",
        size_bytes=instance.size_bytes,
        count=1,
    )


def _transition_from_snapshot(
    kind: type[TransitionT],
    history: _IntervalHistory,
    instance: _AllocationInstance,
) -> TransitionT:
    return kind(
        start_index=history.start.index,
        end_index=history.end.index,
        start_label=_state_label(history.start),
        end_label=_state_label(history.end),
        stack_frames=(),
        stack_fallback="<unavailable>",
        confidence="snapshot_inferred",
        size_bytes=instance.size_bytes,
        count=1,
    )


def _cohorts(
    instances: Sequence[_AllocationInstance],
    points: Sequence[Any],
    *,
    active_at: Any | None,
    born_between: tuple[Any, Any] | None,
) -> tuple[AllocationCohort, ...]:
    grouped: dict[tuple[object, ...], list[_AllocationInstance]] = defaultdict(list)
    for instance in instances:
        fingerprint = _stack_fingerprint(instance.stack_frames)
        unattributed_size = (
            (instance.size_bytes, instance.requested_bytes)
            if not instance.stack_frames
            else None
        )
        grouped[
            (instance.device, instance.pool_id, fingerprint, unattributed_size)
        ].append(instance)

    pending = []
    last_index = points[-1].index
    for (
        device,
        pool_id,
        stack_fingerprint,
        unattributed_size,
    ), members in grouped.items():
        stack_frames = members[0].stack_frames
        point_states = []
        for point in points:
            observed = [
                member.observations[point.index]
                for member in members
                if point.index in member.observations
            ]
            owner_active = [
                item for item in observed if item.state in OWNER_ACTIVE_STATES
            ]
            awaiting_free = [
                item for item in observed if item.state in AWAITING_FREE_STATES
            ]
            point_states.append(
                CohortPointState(
                    point_index=point.index,
                    point_label=_state_label(point),
                    owner_active_bytes=sum(item.size_bytes for item in owner_active),
                    owner_requested_bytes=sum(
                        item.requested_bytes for item in owner_active
                    ),
                    owner_active_count=len(owner_active),
                    awaiting_free_bytes=sum(item.size_bytes for item in awaiting_free),
                    awaiting_free_requested_bytes=sum(
                        item.requested_bytes for item in awaiting_free
                    ),
                    awaiting_free_count=len(awaiting_free),
                )
            )

        nonzero = [item for item in point_states if item.block_count]
        sizes: Counter[tuple[int, int]] = Counter(
            (member.size_bytes, member.requested_bytes) for member in members
        )
        outcomes: Counter[tuple[int, int, LifetimeTerminalState]] = Counter(
            (
                member.size_bytes,
                member.requested_bytes,
                _terminal_state(member, last_index),
            )
            for member in members
        )
        births = _aggregate_births(
            member.birth for member in members if member.birth is not None
        )
        free_requests = _aggregate_free_requests(
            member.free_request for member in members if member.free_request is not None
        )
        free_completions = _aggregate_free_completions(
            member.free_completion
            for member in members
            if member.free_completion is not None
        )
        exact_births = _instances_by_confidence(members, "birth", "event_exact")
        inferred_births = _instances_by_confidence(
            members, "birth", "snapshot_inferred"
        )
        exact_requests = _instances_by_confidence(
            members, "free_request", "event_exact"
        )
        inferred_requests = _instances_by_confidence(
            members, "free_request", "snapshot_inferred"
        )
        exact_completions = _instances_by_confidence(
            members, "free_completion", "event_exact"
        )
        inferred_completions = _instances_by_confidence(
            members, "free_completion", "snapshot_inferred"
        )
        owner_active_at_end = [
            member
            for member in members
            if _terminal_state(member, last_index) == "owner_active"
        ]
        awaiting_free_at_end = [
            member
            for member in members
            if _terminal_state(member, last_index) == "awaiting_free"
        ]
        active_values = [item.active_bytes for item in point_states]
        born_bytes = sum(member.size_bytes for member in members if member.birth)
        owner_peak, unreusable_peak = _event_peaks(members, points[0].index)
        peak_active = max(active_values)
        if born_between is not None:
            anchor_bytes = born_bytes
        elif active_at is not None:
            anchor_bytes = next(
                item.active_bytes
                for item in point_states
                if item.point_index == active_at.index
            )
        else:
            anchor_bytes = max(peak_active, unreusable_peak)
        if nonzero:
            first_seen_index = nonzero[0].point_index
            first_seen_label = nonzero[0].point_label
            last_seen_index = nonzero[-1].point_index
            last_seen_label = nonzero[-1].point_label
        else:
            member_births = [member.birth for member in members if member.birth]
            assert member_births
            first_birth = min(member_births, key=lambda item: item.end_index)
            last_boundary = max(_instance_last_boundary(member) for member in members)
            point_labels = {point.index: _state_label(point) for point in points}
            first_seen_index = first_birth.end_index
            first_seen_label = first_birth.end_label
            last_seen_index = last_boundary
            last_seen_label = point_labels.get(last_boundary, first_birth.end_label)
        pending.append(
            {
                "device": device,
                "pool_id": pool_id,
                "stack_fingerprint": stack_fingerprint,
                "stack_frames": stack_frames,
                "unattributed_size": unattributed_size,
                "members": members,
                "points": tuple(point_states),
                "sizes": tuple(
                    CohortSizeBucket(
                        size_bytes=size,
                        requested_bytes=requested,
                        count=count,
                        total_bytes=size * count,
                    )
                    for (size, requested), count in sorted(
                        sizes.items(),
                        key=lambda item: (-item[0][0] * item[1], -item[1]),
                    )
                ),
                "outcomes": tuple(
                    CohortSizeOutcome(
                        size_bytes=size,
                        requested_bytes=requested,
                        terminal_state=terminal_state,
                        count=count,
                        total_bytes=size * count,
                    )
                    for (size, requested, terminal_state), count in sorted(
                        outcomes.items(),
                        key=lambda item: (
                            -item[0][0] * item[1],
                            item[0][2],
                            -item[1],
                        ),
                    )
                ),
                "births": births,
                "free_requests": free_requests,
                "free_completions": free_completions,
                "exact_births": exact_births,
                "inferred_births": inferred_births,
                "exact_requests": exact_requests,
                "inferred_requests": inferred_requests,
                "exact_completions": exact_completions,
                "inferred_completions": inferred_completions,
                "owner_active_at_end": owner_active_at_end,
                "awaiting_free_at_end": awaiting_free_at_end,
                "anchor_bytes": anchor_bytes,
                "born_bytes": born_bytes,
                "owner_peak": owner_peak,
                "unreusable_peak": unreusable_peak,
                "first_seen_index": first_seen_index,
                "first_seen_label": first_seen_label,
                "last_seen_index": last_seen_index,
                "last_seen_label": last_seen_label,
            }
        )

    pending.sort(
        key=lambda item: (
            -int(item["anchor_bytes"]),
            -max(point.active_bytes for point in item["points"]),
            str(item["device"]),
            str(item["pool_id"]),
            str(item["stack_fingerprint"]),
            str(item["unattributed_size"]),
        )
    )
    result = []
    for display_rank, item in enumerate(pending, start=1):
        point_states = item["points"]
        members = item["members"]
        exact_births = item["exact_births"]
        inferred_births = item["inferred_births"]
        exact_requests = item["exact_requests"]
        inferred_requests = item["inferred_requests"]
        exact_completions = item["exact_completions"]
        inferred_completions = item["inferred_completions"]
        owner_active_at_end = item["owner_active_at_end"]
        awaiting_free_at_end = item["awaiting_free_at_end"]
        active_values = [point.active_bytes for point in point_states]
        result.append(
            AllocationCohort(
                cohort_id=_cohort_id(
                    item["device"],
                    item["pool_id"],
                    item["stack_fingerprint"],
                    item["unattributed_size"],
                ),
                display_rank=display_rank,
                device=item["device"],
                pool_id=item["pool_id"],
                stack_fingerprint=item["stack_fingerprint"],
                stack_frames=item["stack_frames"],
                streams=tuple(sorted({member.stream for member in members}, key=str)),
                points=point_states,
                size_histogram=item["sizes"],
                size_outcomes=item["outcomes"],
                births=item["births"],
                free_requests=item["free_requests"],
                free_completions=item["free_completions"],
                peak_active_bytes=max(active_values),
                peak_owner_active_bytes=max(
                    point.owner_active_bytes for point in point_states
                ),
                peak_awaiting_free_bytes=max(
                    point.awaiting_free_bytes for point in point_states
                ),
                event_owner_peak_bytes=item["owner_peak"],
                event_unreusable_peak_bytes=item["unreusable_peak"],
                peak_block_count=max(point.block_count for point in point_states),
                snapshot_active_span_bytes=max(active_values) - min(active_values),
                first_seen_index=item["first_seen_index"],
                first_seen_label=item["first_seen_label"],
                last_seen_index=item["last_seen_index"],
                last_seen_label=item["last_seen_label"],
                born_bytes=item["born_bytes"],
                born_count=len(exact_births) + len(inferred_births),
                event_exact_birth_bytes=_instance_bytes(exact_births),
                event_exact_birth_count=len(exact_births),
                snapshot_inferred_birth_bytes=_instance_bytes(inferred_births),
                snapshot_inferred_birth_count=len(inferred_births),
                event_exact_free_requested_bytes=_instance_bytes(exact_requests),
                event_exact_free_requested_count=len(exact_requests),
                snapshot_inferred_free_requested_bytes=_instance_bytes(
                    inferred_requests
                ),
                snapshot_inferred_free_requested_count=len(inferred_requests),
                event_exact_free_completed_bytes=_instance_bytes(exact_completions),
                event_exact_free_completed_count=len(exact_completions),
                snapshot_inferred_free_completed_bytes=_instance_bytes(
                    inferred_completions
                ),
                snapshot_inferred_free_completed_count=len(inferred_completions),
                owner_active_at_end_bytes=_instance_bytes(owner_active_at_end),
                owner_active_at_end_count=len(owner_active_at_end),
                awaiting_free_at_end_bytes=_instance_bytes(awaiting_free_at_end),
                awaiting_free_at_end_count=len(awaiting_free_at_end),
            )
        )
    return tuple(result)


def _terminal_state(
    instance: _AllocationInstance, last_index: int
) -> LifetimeTerminalState:
    observation = instance.observations.get(last_index)
    if observation is not None:
        if observation.state in OWNER_ACTIVE_STATES:
            return "owner_active"
        if observation.state in AWAITING_FREE_STATES:
            return "awaiting_free"
    if instance.free_completion is not None:
        return "free_completed"
    return "unknown"


def _instances_by_confidence(
    members: Sequence[_AllocationInstance],
    attribute: Literal["birth", "free_request", "free_completion"],
    confidence: LifetimeConfidence,
) -> list[_AllocationInstance]:
    return [
        member
        for member in members
        if (transition := getattr(member, attribute)) is not None
        and transition.confidence == confidence
    ]


def _instance_bytes(instances: Sequence[_AllocationInstance]) -> int:
    return sum(instance.size_bytes for instance in instances)


def _instance_last_boundary(instance: _AllocationInstance) -> int:
    if instance.free_completion is not None:
        return instance.free_completion.end_index
    if instance.free_request is not None:
        return instance.free_request.end_index
    assert instance.birth is not None
    return instance.birth.end_index


def _event_peaks(
    members: Sequence[_AllocationInstance], start_index: int
) -> tuple[int, int]:
    owner_live = 0
    unreusable = 0
    for member in members:
        if member.birth is not None:
            continue
        observation = member.observations.get(start_index)
        if observation is None:
            continue
        unreusable += member.size_bytes
        if observation.state in OWNER_ACTIVE_STATES:
            owner_live += member.size_bytes
    owner_peak = owner_live
    unreusable_peak = unreusable
    transitions: list[tuple[int, int, int]] = []
    for member in members:
        if member.birth_order is not None:
            transitions.append(
                (member.birth_order, member.size_bytes, member.size_bytes)
            )
        if member.free_request_order is not None:
            transitions.append((member.free_request_order, -member.size_bytes, 0))
        if member.free_completion_order is not None:
            transitions.append((member.free_completion_order, 0, -member.size_bytes))
    for _order, owner_delta, unreusable_delta in sorted(transitions):
        owner_live += owner_delta
        unreusable += unreusable_delta
        owner_peak = max(owner_peak, owner_live)
        unreusable_peak = max(unreusable_peak, unreusable)
    return owner_peak, unreusable_peak


TransitionT = TypeVar("TransitionT", bound=_CohortTransition)


def _aggregate_transitions(
    transitions: Sequence[TransitionT],
    kind: type[TransitionT],
) -> tuple[TransitionT, ...]:
    totals: dict[tuple[object, ...], list[int]] = defaultdict(lambda: [0, 0])
    examples: dict[tuple[object, ...], TransitionT] = {}
    for transition in transitions:
        key = (
            transition.start_index,
            transition.end_index,
            transition.start_label,
            transition.end_label,
            _stack_fingerprint(transition.stack_frames),
            transition.stack_fallback,
            transition.confidence,
        )
        totals[key][0] += transition.size_bytes
        totals[key][1] += transition.count
        examples[key] = transition
    rows = []
    for key, (size_bytes, count) in totals.items():
        example = examples[key]
        rows.append(
            kind(
                start_index=example.start_index,
                end_index=example.end_index,
                start_label=example.start_label,
                end_label=example.end_label,
                stack_frames=example.stack_frames,
                stack_fallback=example.stack_fallback,
                confidence=example.confidence,
                size_bytes=size_bytes,
                count=count,
            )
        )
    rows.sort(
        key=lambda item: (
            item.end_index,
            -item.size_bytes,
            item.confidence,
            item.stack_key,
        )
    )
    return tuple(rows)


def _aggregate_births(births: Sequence[CohortBirth]) -> tuple[CohortBirth, ...]:
    return _aggregate_transitions(births, CohortBirth)


def _aggregate_free_requests(
    requests: Sequence[CohortFreeRequest],
) -> tuple[CohortFreeRequest, ...]:
    return _aggregate_transitions(requests, CohortFreeRequest)


def _aggregate_free_completions(
    completions: Sequence[CohortFreeCompletion],
) -> tuple[CohortFreeCompletion, ...]:
    return _aggregate_transitions(completions, CohortFreeCompletion)
