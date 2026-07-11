"""Allocation-cohort lifetime analysis for memory runs and probe snapshots."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from ._pool_identity import pool_id_label
from ._stack_trace import (
    display_stack,
    normalize_stack_frames,
    stack_frames_json,
    stack_key,
)
from ._stack_trace import (
    stack_fingerprint as _stack_fingerprint,
)
from .allocator_snapshot import (
    ACTIVE_STATES,
    ALLOCATION_LIFETIME_ACTIONS,
    AWAITING_FREE_STATES,
    KNOWN_TRACE_ACTIONS,
    OWNER_ACTIVE_STATES,
    AllocatorTraceEntry,
    normalize_pool_id,
    normalize_snapshot,
    trace_device_indices,
)
from .errors import (
    MemoryHistoryBoundaryError,
    MemoryHistoryDisabledError,
    MemoryHistoryTruncatedError,
    MemoryReconciliationError,
)
from .events import EventWindow, _extract_snapshot_event_windows

if TYPE_CHECKING:
    from .attribution import MemoryLifetimeOptions
    from .recording import MemoryPoint, MemoryRun
    from .reports import MemoryAllocationLifetimeAnalysis
    from .snapshots import MemoryProbeSnapshot


TransitionOrigin = Literal["event", "range_boundary"]
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
    origin: TransitionOrigin
    size_bytes: int
    count: int

    @property
    def stack_key(self) -> str:
        return stack_key(self.stack_frames, fallback=self.stack_fallback)

    def display_stack(self, depth: int) -> str:
        return display_stack(
            self.stack_frames,
            depth=depth,
            fallback=self.stack_fallback,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_label": self.start_label,
            "end_label": self.end_label,
            "stack_key": self.stack_key,
            "origin": self.origin,
            "stack_frames": [dict(frame) for frame in self.stack_frames],
            "size_bytes": self.size_bytes,
            "count": self.count,
        }

    def to_row(self, cohort_id: str) -> dict[str, object]:
        row = self.to_dict()
        row.pop("stack_frames")
        row["stack_frames_json"] = stack_frames_json(self.stack_frames)
        return {"cohort_id": cohort_id, **row}


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
    """Allocation instances sharing a device, pool, and allocation identity.

    Framed allocations group by their complete normalized allocation stack.
    Unframed allocations also include block and requested sizes in the identity
    so unrelated same-pool allocations are not merged.
    """

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
    free_requested_bytes: int
    free_requested_count: int
    free_completed_bytes: int
    free_completed_count: int
    owner_active_at_end_bytes: int
    owner_active_at_end_count: int
    awaiting_free_at_end_bytes: int
    awaiting_free_at_end_count: int

    @property
    def pool_label(self) -> str:
        """Human-readable pool label.

        Transient cohorts whose pool cannot be attributed carry the
        ``("unknown",)`` sentinel, which must render explicitly instead of
        being coerced into a two-integer pool ID.
        """

        if self.pool_id == ("unknown",):
            return "pool[unknown]"
        return pool_id_label(self.pool_id)

    @property
    def stack_key(self) -> str:
        return stack_key(self.stack_frames)

    def display_stack(self, depth: int) -> str:
        return display_stack(self.stack_frames, depth=depth)

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
            "free_requested_bytes": self.free_requested_bytes,
            "free_requested_count": self.free_requested_count,
            "free_completed_bytes": self.free_completed_bytes,
            "free_completed_count": self.free_completed_count,
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
            "pool_id": self.pool_label,
            "streams": ";".join(f"stream[{item}]" for item in self.streams),
            "stack_key": self.stack_key,
            "stack_fingerprint": self.stack_fingerprint,
            "stack_frames_json": stack_frames_json(self.stack_frames),
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
            "free_requested_bytes": self.free_requested_bytes,
            "free_requested_count": self.free_requested_count,
            "free_completed_bytes": self.free_completed_bytes,
            "free_completed_count": self.free_completed_count,
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
    snapshot_confirmed: bool
    observations: dict[int, _BlockObservation] = field(default_factory=dict)
    birth: CohortBirth | None = None
    free_request: CohortFreeRequest | None = None
    free_completion: CohortFreeCompletion | None = None
    birth_order: int | None = None
    free_request_order: int | None = None
    free_completion_order: int | None = None
    birth_history: "_IntervalHistory | None" = None
    birth_entry_index: int | None = None


@dataclass(frozen=True)
class _IntervalHistory:
    start: Any
    end: Any
    entries: tuple[AllocatorTraceEntry, ...]
    available: bool
    complete: bool
    warnings: tuple[str, ...]
    causes: tuple[str, ...] = ()
    start_segments: tuple[Mapping[str, Any], ...] = ()
    end_segments: tuple[Mapping[str, Any], ...] = ()


_REPLAY_WARNING_EXAMPLE_LIMIT = 3


@dataclass
class _ReplayWarningGroup:
    count: int = 0
    example_addresses: list[int] = field(default_factory=list)

    def add(self, address: int) -> None:
        self.count += 1
        if (
            address not in self.example_addresses
            and len(self.example_addresses) < _REPLAY_WARNING_EXAMPLE_LIMIT
        ):
            self.example_addresses.append(address)


def analyze_allocation_lifetimes(
    run: MemoryRun,
    *,
    start: MemoryPoint,
    end: MemoryPoint,
    active_at: MemoryPoint | None,
    born_between: tuple[MemoryPoint, MemoryPoint] | None,
    options: MemoryLifetimeOptions,
    _allocator_states: Sequence[Any] | None = None,
    _event_windows: Sequence[Sequence[EventWindow]] | None = None,
) -> MemoryAllocationLifetimeAnalysis:
    """Build allocation cohorts for a range in one recorded run."""

    return _analyze_allocation_lifetimes(
        run.points[_state_index(start) : _state_index(end) + 1],
        source_kind="run",
        source_id=run.run_id,
        source_name=run.name,
        start=start,
        end=end,
        active_at=active_at,
        born_between=born_between,
        options=options,
        raw_snapshots=_allocator_states,
        event_windows=_event_windows,
    )


def analyze_probe_snapshot_lifetimes(
    reference: MemoryProbeSnapshot,
    candidate: MemoryProbeSnapshot,
    *,
    options: MemoryLifetimeOptions,
    _raw_snapshots: Sequence[Any] | None = None,
    _event_windows: Sequence[Sequence[EventWindow]] | None = None,
) -> MemoryAllocationLifetimeAnalysis:
    """Build allocation cohorts for two ordered snapshots from one Probe."""

    if reference.probe_id != candidate.probe_id:
        raise ValueError("probe lifetime endpoints must belong to one MemoryProbe")
    if candidate.snapshot_index <= reference.snapshot_index:
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
        raw_snapshots=(
            tuple(state.raw_snapshot() for state in (reference, candidate))
            if _raw_snapshots is None
            else _raw_snapshots
        ),
        event_windows=_event_windows,
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
    event_windows: Sequence[Sequence[EventWindow]] | None,
) -> MemoryAllocationLifetimeAnalysis:
    """Build an offline allocation-cohort lifetime report."""

    from .reports import MemoryAllocationLifetimeAnalysis

    points = tuple(states)
    observations, histories = _scan_points(
        points,
        source_kind=source_kind,
        events=True,
        raw_snapshots=raw_snapshots,
        event_windows=event_windows,
    )
    warnings = [warning for point in points for warning in point.warnings]
    for history in histories:
        warnings.extend(history.warnings)
        unknown_actions = sorted(
            {entry.action for entry in history.entries} - KNOWN_TRACE_ACTIONS
        )
        if unknown_actions:
            warnings.append(f"unknown allocator actions: {unknown_actions}")
    _raise_for_history_gaps(histories)

    instances, contradictions = _track_instances(points, observations, histories)
    if contradictions:
        raise MemoryReconciliationError(
            "allocator event history is complete but could not be reconciled "
            "with the snapshots:\n- "
            + "\n- ".join(contradictions)
            + "\nPossible causes include allocator activity during the "
            "non-atomic marker/snapshot window, corrupted input data, or a "
            "torch-cudagraph-debug bug. Quiesce concurrent allocator activity "
            "around collection; if the error persists, please report it together "
            "with this message."
        )
    if active_at is not None:
        instances = [
            item for item in instances if _state_index(active_at) in item.observations
        ]
    elif born_between is not None:
        born_start, born_end = born_between
        instances = [
            item
            for item in instances
            if item.birth is not None
            and item.birth.start_index >= _state_index(born_start)
            and item.birth.end_index <= _state_index(born_end)
        ]
    total_instance_bytes = sum(item.size_bytes for item in instances)
    attributed_instance_bytes = sum(
        item.size_bytes for item in instances if item.stack_frames
    )
    if attributed_instance_bytes < total_instance_bytes:
        warnings.append(
            "allocation stack coverage is incomplete for lifetime cohorts: "
            f"{attributed_instance_bytes}/{total_instance_bytes} bytes attributed; "
            "enable torch.cuda.memory._record_memory_history(...) with stack "
            "context before the allocations of interest"
        )

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
        history_requested=True,
        history_available=True,
        history_complete=True,
        total_instance_bytes=total_instance_bytes,
        attributed_instance_bytes=attributed_instance_bytes,
        display_stack_depth=options.display.stack_depth,
        display_limit=options.display.limit,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _state_label(state: Any) -> str:
    label = getattr(state, "label", None)
    if label is not None:
        return str(label)
    return f"{state.probe_name}@snapshot-{state.snapshot_index}"


def _state_index(state: Any) -> int:
    snapshot_index = getattr(state, "snapshot_index", None)
    return int(state.index if snapshot_index is None else snapshot_index)


def _scan_points(
    points: Sequence[Any],
    *,
    source_kind: Literal["run", "probe"],
    events: bool,
    raw_snapshots: Sequence[Any] | None,
    event_windows: Sequence[Sequence[EventWindow]] | None,
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
    interval_count = max(len(points) - 1, 0)
    if event_windows is not None and len(event_windows) != interval_count:
        raise ValueError("event window count must match lifetime intervals")
    interval_index = 0

    snapshots = (
        iter(raw_snapshots)
        if raw_snapshots is not None
        else (point.allocator_state() for point in points)
    )
    for point, snapshot in zip(points, snapshots):
        segments = normalize_snapshot(snapshot)
        observations[_state_index(point)] = _active_blocks_from_segments(
            point, segments
        )
        if previous is None or previous_segments is None:
            previous = point
            previous_segments = segments
            continue

        if events:
            if event_windows is not None:
                windows = list(event_windows[interval_index])
            elif source_kind == "run":
                windows = list(point._event_windows())
            else:
                devices = sorted(
                    {
                        int(segment["device"])
                        for segment in (*previous_segments, *segments)
                        if segment.get("device") is not None
                    }
                    | set(trace_device_indices(snapshot))
                )
                windows = list(
                    _extract_snapshot_event_windows(
                        snapshot,
                        devices=devices,
                        previous_boundary_recorded=previous._boundary_recorded,
                        current_boundary_recorded=point._boundary_recorded,
                        start_marker=previous.boundary_marker,
                        end_marker=point.boundary_marker,
                        start_label=_state_label(previous),
                        end_label=_state_label(point),
                        start_index=_state_index(previous),
                        end_index=_state_index(point),
                    )
                )
            entries = tuple(entry for window in windows for entry in window.entries)
            available = bool(windows) and all(window.available for window in windows)
            complete = bool(windows) and all(window.complete for window in windows)
            warnings = tuple(
                warning for window in windows for warning in window.warnings
            )
            causes = tuple(
                dict.fromkeys(
                    window.cause for window in windows if window.cause is not None
                )
            )
            if not windows:
                warnings = ("allocator event history is unavailable",)
                causes = ("disabled",)
        else:
            entries = ()
            available = False
            complete = False
            warnings = ()
            causes = ("disabled",)

        histories.append(
            _IntervalHistory(
                start=previous,
                end=point,
                entries=entries,
                available=available,
                complete=complete,
                warnings=warnings,
                causes=causes,
                start_segments=tuple(previous_segments),
                end_segments=tuple(segments),
            )
        )
        previous = point
        previous_segments = segments
        interval_index += 1

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
                    point_index=_state_index(point),
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
    contradiction_groups: dict[tuple[str, int | None], _ReplayWarningGroup] = {}
    event_order = 0

    def record_contradiction(
        kind: str,
        *,
        device: int | None,
        address: int,
    ) -> None:
        contradiction_groups.setdefault((kind, device), _ReplayWarningGroup()).add(
            address
        )

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
            snapshot_confirmed=True,
            observations={block.point_index: block},
            birth=birth,
            birth_order=birth_order,
        )
        instances.append(item)
        return item

    def boundary_free_request(
        start_state: Any, end_state: Any, instance: _AllocationInstance
    ) -> None:
        nonlocal event_order
        if instance.free_request is not None:
            return
        event_order += 1
        instance.free_request = _transition_at_boundary(
            CohortFreeRequest, start_state, end_state, instance
        )
        instance.free_request_order = event_order

    for point_blocks in observations.values():
        seen: set[tuple[int | None, int]] = set()
        for block in point_blocks:
            if block.address is None:
                record_contradiction(
                    "block_without_address", device=block.device, address=0
                )
                continue
            key = (block.device, block.address)
            if key in seen:
                record_contradiction(
                    "duplicate_active_address",
                    device=block.device,
                    address=block.address,
                )
            seen.add(key)

    for block in observations.get(_state_index(points[0]), ()):
        if block.address is None:
            continue
        key = (block.device, block.address)
        if key in current:
            continue
        item = create_from_block(block)
        if block.state in AWAITING_FREE_STATES:
            # The free request predates the analysis range; record it as a
            # range-boundary fact instead of guessing its interval.
            boundary_free_request(points[0], points[0], item)
        current[key] = item

    for history in histories:
        for entry_index, entry in enumerate(history.entries):
            event_order += 1
            if entry.addr is None or entry.action not in ALLOCATION_LIFETIME_ACTIONS:
                continue
            if entry.action == "alloc" and entry.size_bytes == 0:
                continue
            key, instance = _find_address_instance(
                current, entry.device_index, entry.addr
            )
            if entry.action == "free_requested":
                if instance is None:
                    record_contradiction(
                        "free_requested_without_allocation",
                        device=entry.device_index,
                        address=entry.addr,
                    )
                    continue
                if not _event_size_matches(entry, instance):
                    record_contradiction(
                        "event_size_contradiction",
                        device=entry.device_index,
                        address=entry.addr,
                    )
                    continue
                if instance.free_request is None:
                    instance.free_request = _transition_from_event(
                        CohortFreeRequest, history, instance, entry
                    )
                    instance.free_request_order = event_order
                else:
                    record_contradiction(
                        "duplicate_free_requested",
                        device=entry.device_index,
                        address=entry.addr,
                    )
                continue
            if entry.action == "free_completed":
                if instance is None:
                    record_contradiction(
                        "free_completed_without_allocation",
                        device=entry.device_index,
                        address=entry.addr,
                    )
                    continue
                if not _event_size_matches(entry, instance):
                    record_contradiction(
                        "event_size_contradiction",
                        device=entry.device_index,
                        address=entry.addr,
                    )
                    continue
                if instance.free_request is None:
                    # A complete window must contain the request event for any
                    # allocation that entered the range owner-active.
                    record_contradiction(
                        "free_completed_without_request",
                        device=entry.device_index,
                        address=entry.addr,
                    )
                    boundary_free_request(history.start, history.end, instance)
                if instance.free_completion is None:
                    instance.free_completion = _transition_from_event(
                        CohortFreeCompletion, history, instance, entry
                    )
                    instance.free_completion_order = event_order
                assert key is not None
                current.pop(key, None)
                continue

            if instance is not None:
                record_contradiction(
                    "allocation_address_reused",
                    device=entry.device_index,
                    address=entry.addr,
                )
                assert key is not None
                current.pop(key, None)
            size = abs(entry.size_bytes)
            pool_id = entry.pool_id if entry.pool_id is not None else ("unknown",)
            frames = normalize_stack_frames(entry.frames)
            birth = CohortBirth(
                start_index=_state_index(history.start),
                end_index=_state_index(history.end),
                start_label=_state_label(history.start),
                end_label=_state_label(history.end),
                stack_frames=frames,
                stack_fallback="<unattributed>",
                origin="event",
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
                snapshot_confirmed=False,
                birth=birth,
                birth_order=event_order,
                birth_history=history,
                birth_entry_index=entry_index,
            )
            instances.append(item)
            current[(entry.device_index, entry.addr)] = item

        end_blocks = list(observations.get(_state_index(history.end), ()))
        blocks_by_address = _block_indexes(end_blocks)
        consumed_after: set[int] = set()
        next_current: dict[tuple[int | None, int], _AllocationInstance] = {}
        for _key, instance in current.items():
            block_index = _matching_block_index(
                blocks_by_address,
                instance,
                consumed_after,
            )
            if block_index is None:
                record_contradiction(
                    "unexplained_disappearance",
                    device=instance.device,
                    address=instance.address or 0,
                )
                continue
            block = end_blocks[block_index]
            consumed_after.add(block_index)
            previous = _latest_observation(instance)
            generation_reused = (
                previous is not None
                and previous.state in AWAITING_FREE_STATES
                and block.state in OWNER_ACTIVE_STATES
            ) or (
                instance.free_request is not None and block.state in OWNER_ACTIVE_STATES
            )
            if generation_reused:
                # A generational swap must be witnessed by free_completed and
                # alloc events inside a complete window.
                record_contradiction(
                    "allocation_address_reused",
                    device=instance.device,
                    address=instance.address or 0,
                )
                continue
            identity_contradictions = _instance_block_contradictions(instance, block)
            if identity_contradictions:
                for kind in identity_contradictions:
                    record_contradiction(
                        kind,
                        device=instance.device,
                        address=instance.address or 0,
                    )
                continue
            if block.state in AWAITING_FREE_STATES and instance.free_request is None:
                record_contradiction(
                    "free_request_event_missing",
                    device=instance.device,
                    address=instance.address or 0,
                )
                boundary_free_request(history.start, history.end, instance)
            instance.observations[block.point_index] = block
            _refine_instance_from_block(instance, block)
            assert block.address is not None
            next_current[(block.device, block.address)] = instance

        for block_index, block in enumerate(end_blocks):
            if block_index in consumed_after:
                continue
            record_contradiction(
                "unexplained_birth",
                device=block.device,
                address=block.address or 0,
            )
            event_order += 1
            birth = _transition_at_boundary(
                CohortBirth,
                history.start,
                history.end,
                size_bytes=block.size_bytes,
                stack_frames=block.stack_frames,
                stack_fallback="<unattributed>",
            )
            item = create_from_block(
                block,
                birth=birth,
                birth_order=event_order,
            )
            if block.state in AWAITING_FREE_STATES:
                boundary_free_request(history.start, history.end, item)
            if block.address is not None:
                next_current[(block.device, block.address)] = item
        current = next_current

    # Snapshot-observed instances take their pool from block ground truth;
    # transients were never observed, so anchor their pool temporally to the
    # endpoint snapshots of their birth interval.
    pending_attribution: dict[
        int, tuple[_IntervalHistory, list[_AllocationInstance]]
    ] = {}
    for instance in instances:
        if instance.pool_id != ("unknown",) or instance.observations:
            continue
        if instance.birth_history is None or instance.birth_entry_index is None:
            continue
        history = instance.birth_history
        pending_attribution.setdefault(id(history), (history, []))[1].append(instance)

    for history, pending in pending_attribution.values():
        run_indexes = (
            _build_run_range_index(history.start_segments),
            _build_run_range_index(history.end_segments),
        )
        action_evidence = _segment_action_evidence(
            history,
            tuple(
                (
                    instance.birth_entry_index,
                    history.entries[instance.birth_entry_index],
                )
                for instance in pending
                if instance.birth_entry_index is not None
            ),
        )
        witness_cache: dict[tuple[str, tuple[Any, ...]], int | None] = {}
        for instance in pending:
            assert instance.birth_entry_index is not None
            entry_index = instance.birth_entry_index
            entry = history.entries[entry_index]
            pool_id, pool_contradictions = _attribute_event_pool(
                entry,
                entry_index,
                history,
                witness_cache,
                action_evidence[entry_index],
                run_indexes,
            )
            for kind in pool_contradictions:
                record_contradiction(
                    kind, device=entry.device_index, address=entry.addr
                )
            instance.pool_id = pool_id

    return instances, _format_contradictions(contradiction_groups)


_CONTRADICTION_TEMPLATES = {
    "free_requested_without_allocation": (
        "{count} free_requested {label} referenced addresses with no tracked "
        "allocation on {device}"
    ),
    "free_completed_without_allocation": (
        "{count} free_completed {label} referenced addresses with no tracked "
        "allocation on {device}"
    ),
    "event_size_contradiction": (
        "{count} free {label} on {device} carried sizes incompatible with the "
        "tracked allocation"
    ),
    "free_completed_without_request": (
        "{count} free_completed {label} on {device} arrived without any "
        "free_requested event for an in-range allocation"
    ),
    "allocation_address_reused": (
        "{count} allocation {label} on {device} reused an address whose "
        "previous allocation has no free_completed event"
    ),
    "block_size_contradiction": (
        "{count} tracked allocation(s) on {device} changed allocator-rounded size"
    ),
    "unexplained_disappearance": (
        "{count} tracked allocation(s) on {device} disappeared from the "
        "snapshot without any free event"
    ),
    "unexplained_birth": (
        "{count} snapshot block(s) on {device} appeared without any alloc event"
    ),
    "free_request_event_missing": (
        "{count} block(s) on {device} entered an awaiting-free state without "
        "a free_requested event"
    ),
    "duplicate_free_requested": (
        "{count} duplicate free_requested {label} referenced an allocation "
        "that was already awaiting free on {device}"
    ),
    "duplicate_active_address": (
        "{count} duplicate active block address(es) appeared within one "
        "snapshot on {device}"
    ),
    "requested_size_contradiction": (
        "{count} tracked allocation(s) on {device} changed requested size"
    ),
    "pool_contradiction": (
        "{count} tracked allocation(s) on {device} changed allocator pool"
    ),
    "stream_contradiction": (
        "{count} tracked allocation(s) on {device} changed allocation stream"
    ),
    "allocation_stack_contradiction": (
        "{count} tracked allocation(s) on {device} changed allocation stack"
    ),
    "pool_identity_contradiction": (
        "{count} transient allocation(s) on {device} saw both interval "
        "endpoints claim their address with conflicting pools despite no "
        "covering segment churn"
    ),
    "unexplained_mapping_gap": (
        "{count} transient allocation(s) on {device} were born at addresses "
        "with no covering segment at an interval endpoint and no covering "
        "segment event explaining the gap"
    ),
    "block_without_address": ("{count} active block(s) on {device} carried no address"),
}


def _format_contradictions(
    groups: Mapping[tuple[str, int | None], _ReplayWarningGroup],
) -> tuple[str, ...]:
    messages: list[str] = []
    for (kind, device), group in groups.items():
        device_label = f"device {device}" if device is not None else "an unknown device"
        count_label = "event" if group.count == 1 else "events"
        template = _CONTRADICTION_TEMPLATES.get(kind)
        if template is None:  # pragma: no cover - callers use the closed set.
            raise AssertionError(f"unknown contradiction kind: {kind}")
        message = template.format(
            count=group.count, label=count_label, device=device_label
        )
        addresses = ", ".join(hex(address) for address in group.example_addresses)
        if addresses:
            message = f"{message}; example addresses: {addresses}"
        messages.append(message)
    return tuple(messages)


def _raise_for_history_gaps(histories: Sequence[_IntervalHistory]) -> None:
    disabled: list[str] = []
    boundary_unavailable: list[str] = []
    truncated: list[str] = []
    inconsistent: list[str] = []
    for history in histories:
        label = f"{_state_label(history.start)} -> {_state_label(history.end)}"
        if "invalid_boundary_order" in history.causes:
            inconsistent.append(label)
        elif "boundary_unavailable" in history.causes:
            boundary_unavailable.append(label)
        elif "disabled" in history.causes or not history.available:
            disabled.append(label)
        elif "truncated" in history.causes or not history.complete:
            truncated.append(label)
    if inconsistent:
        raise MemoryReconciliationError(
            "allocator event boundary order is inconsistent for interval(s): "
            + ", ".join(dict.fromkeys(inconsistent))
            + "; the recorded markers were interleaved or the input snapshots are corrupted"
        )
    if boundary_unavailable:
        raise MemoryHistoryBoundaryError(
            "allocator event boundaries could not be recorded for interval(s): "
            + ", ".join(dict.fromkeys(boundary_unavailable))
            + "; record points outside CUDA Graph capture or use a PyTorch build "
            "with memory metadata support"
        )
    if disabled:
        raise MemoryHistoryDisabledError(
            "allocator event history is unavailable for interval(s): "
            + ", ".join(dict.fromkeys(disabled))
            + "; enable torch.cuda.memory._record_memory_history() before the "
            "allocations of interest on every analyzed device"
        )
    if truncated:
        raise MemoryHistoryTruncatedError(
            "allocator event history was truncated for interval(s): "
            + ", ".join(dict.fromkeys(truncated))
            + "; enable _record_memory_history() before the first analyzed "
            "point and keep it enabled, raise "
            "_record_memory_history(max_entries=...), or record points more "
            "frequently so each interval fits the ring buffer"
        )


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
) -> Mapping[tuple[int | None, int], tuple[int, ...]]:
    by_address: defaultdict[tuple[int | None, int], list[int]] = defaultdict(list)
    for index, block in enumerate(blocks):
        if block.address is None:
            continue
        by_address[(block.device, block.address)].append(index)
    return {key: tuple(indices) for key, indices in by_address.items()}


def _matching_block_index(
    blocks_by_address: Mapping[tuple[int | None, int], Sequence[int]],
    instance: _AllocationInstance,
    consumed: set[int],
) -> int | None:
    """Return the unconsumed endpoint block at an instance's exact address."""

    if instance.address is None:
        return None
    candidates = [
        index
        for index in blocks_by_address.get((instance.device, instance.address), ())
        if index not in consumed
    ]
    return candidates[0] if candidates else None


def _instance_block_contradictions(
    instance: _AllocationInstance,
    block: _BlockObservation,
) -> tuple[str, ...]:
    """Return immutable-identity conflicts for one exact-address match.

    An allocator event knows the requested size but not necessarily the
    allocator-rounded block size or pool. Its first snapshot match confirms
    those fields. Snapshot-confirmed values must not drift without a witnessed
    free-complete/alloc generation change.
    """

    contradictions: list[str] = []
    if instance.requested_bytes != block.requested_bytes:
        contradictions.append("requested_size_contradiction")
    if instance.snapshot_confirmed and instance.size_bytes != block.size_bytes:
        contradictions.append("block_size_contradiction")
    if instance.pool_id != ("unknown",) and instance.pool_id != block.pool_id:
        contradictions.append("pool_contradiction")
    if instance.stream is not None and instance.stream != block.stream:
        contradictions.append("stream_contradiction")
    if (
        instance.snapshot_confirmed
        and instance.stack_frames
        and instance.stack_frames != block.stack_frames
    ):
        contradictions.append("allocation_stack_contradiction")
    return tuple(contradictions)


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
    if instance.stream is None:
        instance.stream = block.stream
    if not instance.snapshot_confirmed:
        if block.stack_frames:
            instance.stack_frames = block.stack_frames
        instance.size_bytes = block.size_bytes
        instance.snapshot_confirmed = True
    elif not instance.stack_frames and block.stack_frames:
        instance.stack_frames = block.stack_frames
    # Transition rows recorded before refinement carry the event-requested
    # size; re-stamp them so every row of an observed instance reports the
    # same allocator-rounded basis as the cohort byte totals. Transient
    # instances are never refined and stay on the requested basis throughout.
    if instance.birth is not None and instance.birth.size_bytes != block.size_bytes:
        instance.birth = replace(instance.birth, size_bytes=block.size_bytes)
    if (
        instance.free_request is not None
        and instance.free_request.size_bytes != block.size_bytes
    ):
        instance.free_request = replace(
            instance.free_request, size_bytes=block.size_bytes
        )


_SEGMENT_SCOPE_ACTIONS = frozenset(
    {"segment_alloc", "segment_free", "segment_map", "segment_unmap"}
)
_HARD_CHURN_ACTIONS = frozenset({"segment_alloc", "segment_free"})
_START_GAP_EXPLANATIONS = frozenset({"segment_map", "segment_alloc"})
_END_GAP_EXPLANATIONS = frozenset({"segment_unmap", "segment_free"})


@dataclass(frozen=True)
class _RunRangeIndex:
    starts_by_device: Mapping[int | None, tuple[int, ...]]
    runs_by_device: Mapping[int | None, tuple[Mapping[str, Any], ...]]

    def find(self, device: int | None, address: int) -> Mapping[str, Any] | None:
        starts = self.starts_by_device.get(device)
        runs = self.runs_by_device.get(device)
        if not starts or not runs:
            return None
        index = bisect_right(starts, address) - 1
        if index < 0:
            return None
        run = runs[index]
        size = run.get("total_size")
        if not isinstance(size, int) or address >= starts[index] + size:
            return None
        return run


def _build_run_range_index(
    segments: Sequence[Mapping[str, Any]],
) -> _RunRangeIndex:
    runs_by_device: defaultdict[
        int | None, list[tuple[int, int, Mapping[str, Any]]]
    ] = defaultdict(list)
    for ordinal, run in enumerate(segments):
        base = run.get("address")
        size = run.get("total_size")
        if not isinstance(base, int) or not isinstance(size, int) or size <= 0:
            continue
        runs_by_device[run.get("device")].append((base, ordinal, run))
    ordered = {
        device: tuple(item[2] for item in sorted(items))
        for device, items in runs_by_device.items()
    }
    return _RunRangeIndex(
        starts_by_device={
            device: tuple(int(run["address"]) for run in runs)
            for device, runs in ordered.items()
        },
        runs_by_device=ordered,
    )


def _segment_action_evidence(
    history: _IntervalHistory,
    queries: Sequence[tuple[int, AllocatorTraceEntry]],
) -> dict[int, tuple[frozenset[str], frozenset[str]]]:
    """Index segment actions covering each queried birth before and after it.

    Queries are answered offline per device and action. A Fenwick difference
    tree range-adds event coverage over the sorted birth addresses, avoiding a
    complete trace scan for every transient allocation.
    """

    evidence: dict[int, tuple[set[str], set[str]]] = {
        entry_index: (set(), set()) for entry_index, _entry in queries
    }
    queries_by_device: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for entry_index, entry in queries:
        assert entry.addr is not None
        queries_by_device[entry.device_index].append((entry_index, entry.addr))

    events_by_device_action: defaultdict[
        tuple[int, str], list[tuple[int, AllocatorTraceEntry]]
    ] = defaultdict(list)
    for event_index, entry in enumerate(history.entries):
        if entry.action in _SEGMENT_SCOPE_ACTIONS:
            events_by_device_action[(entry.device_index, entry.action)].append(
                (event_index, entry)
            )

    for device, device_queries in queries_by_device.items():
        addresses = sorted({address for _index, address in device_queries})
        address_index = {address: index for index, address in enumerate(addresses)}
        ascending_queries = sorted(device_queries)
        descending_queries = list(reversed(ascending_queries))

        def add_event(bit: list[int], entry: AllocatorTraceEntry) -> None:
            if entry.addr is None or entry.size_bytes <= 0:
                low, high = 0, len(addresses)
            else:
                low = bisect_left(addresses, entry.addr)
                high = bisect_left(addresses, entry.addr + entry.size_bytes)
            if low >= high:
                return
            _fenwick_add(bit, low, 1)
            _fenwick_add(bit, high, -1)

        for action in _SEGMENT_SCOPE_ACTIONS:
            events = events_by_device_action.get((device, action), ())
            if not events:
                continue

            bit = [0] * (len(addresses) + 2)
            event_cursor = 0
            for entry_index, address in ascending_queries:
                while (
                    event_cursor < len(events) and events[event_cursor][0] < entry_index
                ):
                    add_event(bit, events[event_cursor][1])
                    event_cursor += 1
                if _fenwick_point(bit, address_index[address]) > 0:
                    evidence[entry_index][0].add(action)

            bit = [0] * (len(addresses) + 2)
            event_cursor = len(events) - 1
            for entry_index, address in descending_queries:
                while event_cursor >= 0 and events[event_cursor][0] > entry_index:
                    add_event(bit, events[event_cursor][1])
                    event_cursor -= 1
                if _fenwick_point(bit, address_index[address]) > 0:
                    evidence[entry_index][1].add(action)

    return {
        entry_index: (frozenset(before), frozenset(after))
        for entry_index, (before, after) in evidence.items()
    }


def _fenwick_add(bit: list[int], index: int, delta: int) -> None:
    index += 1
    while index < len(bit):
        bit[index] += delta
        index += index & -index


def _fenwick_point(bit: Sequence[int], index: int) -> int:
    total = 0
    index += 1
    while index:
        total += bit[index]
        index -= index & -index
    return total


def _run_witness_key(run: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        run.get("device"),
        run.get("stream"),
        normalize_pool_id(run.get("segment_pool_id")),
        run.get("segment_type"),
    )


def _clip_to_universe(
    universe: Sequence[tuple[int, int]], low: int, high: int
) -> list[tuple[int, int]]:
    clipped = []
    for begin, end in universe:
        lo = max(begin, low)
        hi = min(end, high)
        if lo < hi:
            clipped.append((lo, hi))
    return clipped


def _subtract_interval(
    mapped: list[tuple[int, int]], low: int, high: int
) -> list[tuple[int, int]]:
    result = []
    for begin, end in mapped:
        if high <= begin or end <= low:
            result.append((begin, end))
            continue
        if begin < low:
            result.append((begin, low))
        if high < end:
            result.append((high, end))
    return result


def _add_intervals(
    mapped: list[tuple[int, int]], pieces: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    merged = sorted([*mapped, *pieces])
    result: list[tuple[int, int]] = []
    for begin, end in merged:
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((begin, end))
    return result


def _witness_death_index(
    history: _IntervalHistory,
    side: str,
    key: tuple[Any, ...],
) -> int | None:
    """First event index at which the reservation's witnessed footprint is
    empty, walking forward from the start snapshot or backward from the end.

    The witness universe is every expandable mapped range sharing the anchor
    run's (device, stream, pool, segment_type) key. The native allocator
    normally grows one reservation per key, so a surviving same-key range is
    useful liveness evidence. Under extreme fragmentation it may create more
    than one reservation with the same key, while snapshots expose no
    reservation ID; this witness can then conflate them. Bytes mapped outside
    the universe cannot be credited, so an empty witness remains a
    conservative death verdict.
    """

    device = key[0]
    segments = history.start_segments if side == "start" else history.end_segments
    universe = []
    for run in segments:
        if not run.get("is_expandable"):
            continue
        if _run_witness_key(run) != key:
            continue
        base = run.get("address")
        size = run.get("total_size")
        if isinstance(base, int) and isinstance(size, int) and size > 0:
            universe.append((base, base + size))
    if not universe:
        return None
    universe = _add_intervals([], universe)
    mapped = list(universe)

    indices: Sequence[int]
    if side == "start":
        indices = range(len(history.entries))
    else:
        indices = range(len(history.entries) - 1, -1, -1)
    for index in indices:
        entry = history.entries[index]
        if entry.action not in ("segment_map", "segment_unmap"):
            continue
        if entry.device_index != device:
            continue
        if entry.addr is None or entry.size_bytes <= 0:
            return index
        low, high = entry.addr, entry.addr + entry.size_bytes
        pieces = _clip_to_universe(universe, low, high)
        if not pieces:
            continue
        # Walking backward, undoing a forward map removes the region and
        # undoing a forward unmap restores it.
        removes = entry.action == "segment_unmap"
        if side == "end":
            removes = not removes
        if removes:
            mapped = _subtract_interval(mapped, low, high)
        else:
            mapped = _add_intervals(mapped, pieces)
        if not mapped:
            return index
    return None


def _attribute_event_pool(
    entry: AllocatorTraceEntry,
    entry_index: int,
    history: _IntervalHistory,
    witness_cache: dict[tuple[str, tuple[Any, ...]], int | None],
    action_evidence: tuple[frozenset[str], frozenset[str]],
    run_indexes: tuple[_RunRangeIndex, _RunRangeIndex],
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """Attribute a pool to an event-born allocation by temporal anchoring.

    An endpoint snapshot can testify about the allocation's birth mapping
    only when no covering segment churn separates them; expandable coverage
    additionally requires the reservation's witnessed footprint to stay
    alive across the span. Trace entries that carry their own pool ID
    short-circuit everything.
    """

    if entry.pool_id is not None:
        return entry.pool_id, ()
    if entry.addr is None:
        return ("unknown",), ()
    device = entry.device_index
    address = entry.addr

    def side_state(side: str) -> tuple[str, tuple[Any, ...] | None]:
        covering_actions = action_evidence[0 if side == "start" else 1]
        run = run_indexes[0 if side == "start" else 1].find(device, address)
        if run is None:
            explanations = (
                _START_GAP_EXPLANATIONS if side == "start" else _END_GAP_EXPLANATIONS
            )
            if covering_actions & explanations:
                return "abstain", None
            return "contradiction", None
        if covering_actions & _HARD_CHURN_ACTIONS:
            return "invalid", None
        pool = normalize_pool_id(run.get("segment_pool_id"))
        if run.get("is_expandable"):
            key = _run_witness_key(run)
            cache_key = (side, key)
            if cache_key not in witness_cache:
                witness_cache[cache_key] = _witness_death_index(history, side, key)
            death = witness_cache[cache_key]
            if death is not None and (
                death < entry_index if side == "start" else death > entry_index
            ):
                return "invalid", None
            return "valid", pool
        if covering_actions:
            # map/unmap over a non-expandable run cannot occur in honest
            # evidence; degrade conservatively instead of guessing.
            return "invalid", None
        return "valid", pool

    start_state, start_pool = side_state("start")
    end_state, end_pool = side_state("end")
    contradictions = []
    if "contradiction" in (start_state, end_state):
        contradictions.append("unexplained_mapping_gap")
    if start_state == "valid" and end_state == "valid":
        if start_pool == end_pool:
            assert start_pool is not None
            return start_pool, tuple(contradictions)
        contradictions.append("pool_identity_contradiction")
        return ("unknown",), tuple(contradictions)
    if start_state == "valid":
        assert start_pool is not None
        return start_pool, tuple(contradictions)
    if end_state == "valid":
        assert end_pool is not None
        return end_pool, tuple(contradictions)
    return ("unknown",), tuple(contradictions)


def _transition_from_event(
    kind: type[TransitionT],
    history: _IntervalHistory,
    instance: _AllocationInstance,
    entry: AllocatorTraceEntry,
) -> TransitionT:
    frames = normalize_stack_frames(entry.frames)
    return kind(
        start_index=_state_index(history.start),
        end_index=_state_index(history.end),
        start_label=_state_label(history.start),
        end_label=_state_label(history.end),
        stack_frames=frames,
        stack_fallback="<unavailable>",
        origin="event",
        size_bytes=instance.size_bytes,
        count=1,
    )


def _transition_at_boundary(
    kind: type[TransitionT],
    start_state: Any,
    end_state: Any,
    instance: _AllocationInstance | None = None,
    *,
    size_bytes: int | None = None,
    stack_frames: tuple[Mapping[str, Any], ...] = (),
    stack_fallback: str = "<unavailable>",
) -> TransitionT:
    """Record a transition whose trigger lies outside the analyzed range."""

    resolved_size = instance.size_bytes if instance is not None else size_bytes
    assert resolved_size is not None
    return kind(
        start_index=_state_index(start_state),
        end_index=_state_index(end_state),
        start_label=_state_label(start_state),
        end_label=_state_label(end_state),
        stack_frames=stack_frames,
        stack_fallback=stack_fallback,
        origin="range_boundary",
        size_bytes=resolved_size,
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
    last_index = _state_index(points[-1])
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
                member.observations[_state_index(point)]
                for member in members
                if _state_index(point) in member.observations
            ]
            owner_active = [
                item for item in observed if item.state in OWNER_ACTIVE_STATES
            ]
            awaiting_free = [
                item for item in observed if item.state in AWAITING_FREE_STATES
            ]
            point_states.append(
                CohortPointState(
                    point_index=_state_index(point),
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
        born_members = _instances_with(members, "birth")
        requested_members = _instances_with(members, "free_request")
        completed_members = _instances_with(members, "free_completion")
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
        owner_peak, unreusable_peak = _event_peaks(members, _state_index(points[0]))
        peak_active = max(active_values)
        if born_between is not None:
            anchor_bytes = born_bytes
        elif active_at is not None:
            anchor_bytes = next(
                item.active_bytes
                for item in point_states
                if item.point_index == _state_index(active_at)
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
            point_labels = {
                _state_index(point): _state_label(point) for point in points
            }
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
                "born_members": born_members,
                "requested_members": requested_members,
                "completed_members": completed_members,
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
        born_members = item["born_members"]
        requested_members = item["requested_members"]
        completed_members = item["completed_members"]
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
                born_count=len(born_members),
                free_requested_bytes=_instance_bytes(requested_members),
                free_requested_count=len(requested_members),
                free_completed_bytes=_instance_bytes(completed_members),
                free_completed_count=len(completed_members),
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


def _instances_with(
    members: Sequence[_AllocationInstance],
    attribute: Literal["birth", "free_request", "free_completion"],
) -> list[_AllocationInstance]:
    return [member for member in members if getattr(member, attribute) is not None]


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
            # A free request only balances an owner contribution the member
            # actually made: an in-range birth or an owner-active range-start
            # state. Blocks already awaiting free at the range start never
            # entered owner_live, so their boundary free request must not
            # subtract from it.
            observation = (
                member.observations.get(start_index) if member.birth is None else None
            )
            held_owner_active = member.birth_order is not None or (
                observation is not None and observation.state in OWNER_ACTIVE_STATES
            )
            transitions.append(
                (
                    member.free_request_order,
                    -member.size_bytes if held_owner_active else 0,
                    0,
                )
            )
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
            transition.origin,
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
                origin=example.origin,
                size_bytes=size_bytes,
                count=count,
            )
        )
    rows.sort(
        key=lambda item: (
            item.end_index,
            -item.size_bytes,
            item.origin,
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
