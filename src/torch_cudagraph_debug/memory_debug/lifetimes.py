"""Offline allocation-cohort lifetime analysis for memory run bundles."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING, TypeVar

from ._ranges import PoolRangeIndex, build_pool_range_index
from .errors import MemoryHistoryError
from .events import (
    extract_event_window,
    extract_event_window_from_snapshot,
)
from .summary import (
    ACTIVE_STATES,
    TraceEntry,
    normalize_pool_id,
    normalize_snapshot,
    normalize_trace_entries,
    stack_key_from_frames,
    trace_device_indices,
)

if TYPE_CHECKING:
    from .core import AttributionOptions, MemoryPoint, MemoryRun
    from .reports import AllocationLifetimeReport


LifetimeConfidence = Literal["event_exact", "snapshot_inferred"]


@dataclass(frozen=True)
class CohortPointState:
    """Active allocation state for one cohort at one memory point."""

    point_index: int
    point_label: str
    active_bytes: int
    requested_bytes: int
    block_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
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
class _CohortTransition:
    before_index: int
    after_index: int
    before_label: str
    after_label: str
    stack_key: str
    confidence: LifetimeConfidence
    size_bytes: int
    count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "before_index": self.before_index,
            "after_index": self.after_index,
            "before_label": self.before_label,
            "after_label": self.after_label,
            "stack_key": self.stack_key,
            "confidence": self.confidence,
            "size_bytes": self.size_bytes,
            "count": self.count,
        }

    def to_row(self, cohort_id: str) -> dict[str, object]:
        return {"cohort_id": cohort_id, **self.to_dict()}


@dataclass(frozen=True)
class CohortRelease(_CohortTransition):
    """Owner-release observations grouped by interval and free stack."""


@dataclass(frozen=True)
class CohortBirth(_CohortTransition):
    """Allocation births grouped by marker interval and allocation stack."""


@dataclass(frozen=True)
class AllocationCohort:
    """Allocation instances sharing a device, pool, and allocation stack."""

    cohort_id: str
    device: int | None
    pool_id: tuple[Any, ...]
    stack_key: str
    streams: tuple[Any, ...]
    points: tuple[CohortPointState, ...]
    size_histogram: tuple[CohortSizeBucket, ...]
    births: tuple[CohortBirth, ...]
    releases: tuple[CohortRelease, ...]
    peak_active_bytes: int
    peak_live_bytes: int
    peak_block_count: int
    impact_bytes: int
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
    event_exact_release_bytes: int
    event_exact_release_count: int
    snapshot_inferred_release_bytes: int
    snapshot_inferred_release_count: int
    still_active_bytes: int
    still_active_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "cohort_id": self.cohort_id,
            "device": self.device,
            "pool_id": list(self.pool_id),
            "stack_key": self.stack_key,
            "streams": list(self.streams),
            "peak_active_bytes": self.peak_active_bytes,
            "peak_live_bytes": self.peak_live_bytes,
            "peak_block_count": self.peak_block_count,
            "impact_bytes": self.impact_bytes,
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
            "event_exact_release_bytes": self.event_exact_release_bytes,
            "event_exact_release_count": self.event_exact_release_count,
            "snapshot_inferred_release_bytes": self.snapshot_inferred_release_bytes,
            "snapshot_inferred_release_count": self.snapshot_inferred_release_count,
            "still_active_bytes": self.still_active_bytes,
            "still_active_count": self.still_active_count,
            "points": [item.to_dict() for item in self.points],
            "size_histogram": [item.to_dict() for item in self.size_histogram],
            "births": [item.to_dict() for item in self.births],
            "releases": [item.to_dict() for item in self.releases],
        }

    def to_row(self) -> dict[str, object]:
        return {
            "cohort_id": self.cohort_id,
            "device": "unknown" if self.device is None else self.device,
            "pool_id": "pool[" + ",".join(str(item) for item in self.pool_id) + "]",
            "streams": ";".join(f"stream[{item}]" for item in self.streams),
            "stack_key": self.stack_key,
            "peak_active_bytes": self.peak_active_bytes,
            "peak_live_bytes": self.peak_live_bytes,
            "peak_block_count": self.peak_block_count,
            "impact_bytes": self.impact_bytes,
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
            "event_exact_release_bytes": self.event_exact_release_bytes,
            "event_exact_release_count": self.event_exact_release_count,
            "snapshot_inferred_release_bytes": self.snapshot_inferred_release_bytes,
            "snapshot_inferred_release_count": self.snapshot_inferred_release_count,
            "still_active_bytes": self.still_active_bytes,
            "still_active_count": self.still_active_count,
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
    stack_key: str

@dataclass
class _AllocationInstance:
    device: int | None
    pool_id: tuple[Any, ...]
    stream: Any
    address: int | None
    size_bytes: int
    requested_bytes: int
    stack_key: str
    observations: dict[int, _BlockObservation] = field(default_factory=dict)
    birth: CohortBirth | None = None
    release: CohortRelease | None = None
    birth_order: int | None = None
    release_order: int | None = None


@dataclass(frozen=True)
class _IntervalHistory:
    before: Any
    after: Any
    entries: tuple[TraceEntry, ...]
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
    options: AttributionOptions,
) -> AllocationLifetimeReport:
    """Build an offline allocation-cohort lifetime report."""

    from .reports import AllocationLifetimeReport

    points = run.points[start.index : end.index + 1]
    observations, histories = _scan_points(
        points,
        events=options.events,
        stack_depth=options.stack_depth,
    )
    warnings = [warning for point in points for warning in point.warnings]
    if options.events:
        for history in histories:
            warnings.extend(history.warnings)
        if any(not item.available or not item.complete for item in histories):
            message = (
                "allocator event history is unavailable or incomplete; "
                "allocation lifetimes include snapshot-inferred transitions"
            )
            if options.on_missing == "error":
                raise MemoryHistoryError(message)
            warnings.append(message)

    instances = _track_instances(points, observations, histories, options.stack_depth)
    if active_at is not None:
        instances = [item for item in instances if active_at.index in item.observations]
    elif born_between is not None:
        born_after, born_through = born_between
        instances = [
            item
            for item in instances
            if item.birth is not None
            and item.birth.before_index >= born_after.index
            and item.birth.after_index <= born_through.index
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
    else:
        # Preserve the existing all-cohort behavior: event-only transient
        # generations are exposed only by explicit born-between analysis.
        instances = [item for item in instances if item.observations]

    total_instance_bytes = sum(item.size_bytes for item in instances)
    attributed_instance_bytes = sum(
        item.size_bytes for item in instances if item.stack_key != "<unattributed>"
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
        limit=options.limit,
    )
    return AllocationLifetimeReport(
        run=run,
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
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _scan_points(
    points: Sequence[MemoryPoint],
    *,
    events: bool,
    stack_depth: int,
) -> tuple[
    dict[int, tuple[_BlockObservation, ...]],
    tuple[_IntervalHistory, ...],
]:
    observations: dict[int, tuple[_BlockObservation, ...]] = {}
    histories: list[_IntervalHistory] = []
    previous: MemoryPoint | None = None
    previous_segments: tuple[Mapping[str, Any], ...] | None = None

    for point in points:
        snapshot = point.raw_snapshot()
        segments = normalize_snapshot(snapshot)
        observations[point.index] = _active_blocks_from_segments(
            point,
            segments,
            stack_depth=stack_depth,
        )
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
                        start_label=f"{previous.label} on device {device}",
                    )
                    for device in devices
                ]
            else:
                windows = [
                    extract_event_window(
                        normalize_trace_entries(snapshot),
                        start_marker=previous.boundary_marker,
                        end_marker=point.boundary_marker,
                        start_label=previous.label,
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
                before=previous,
                after=point,
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
    point: MemoryPoint,
    segments: Sequence[Mapping[str, Any]],
    *,
    stack_depth: int,
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
                    point_label=point.label,
                    ordinal=ordinal,
                    device=device,
                    pool_id=pool_id,
                    stream=stream,
                    address=address,
                    size_bytes=int(block.get("size", 0) or 0),
                    requested_bytes=int(block.get("requested_size", 0) or 0),
                    state=str(block.get("state")),
                    stack_key=stack_key_from_frames(
                        block.get("frames") or (), depth=stack_depth
                    ),
                )
            )
            ordinal += 1
    return tuple(rows)


def _track_instances(
    points: Sequence[MemoryPoint],
    observations: Mapping[int, Sequence[_BlockObservation]],
    histories: Sequence[_IntervalHistory],
    stack_depth: int,
) -> list[_AllocationInstance]:
    instances: list[_AllocationInstance] = []
    current: dict[tuple[int | None, int], _AllocationInstance] = {}
    missing_address: list[_AllocationInstance] = []
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
            stack_key=block.stack_key,
            observations={block.point_index: block},
            birth=birth,
            birth_order=birth_order,
        )
        instances.append(item)
        return item

    for block in observations.get(points[0].index, ()):
        item = create_from_block(block)
        if block.address is None:
            missing_address.append(item)
        else:
            current[(block.device, block.address)] = item

    for history in histories:
        for entry in history.entries:
            event_order += 1
            if entry.addr is None or entry.action not in {"alloc", "free_requested"}:
                continue
            if entry.action == "alloc" and entry.size_bytes == 0:
                continue
            key, instance = _find_address_instance(
                current, entry.device_index, entry.addr
            )
            if entry.action == "free_requested":
                if instance is None or not _event_size_matches(entry, instance):
                    continue
                instance.release = _release_from_event(
                    history, instance, entry, stack_depth=stack_depth
                )
                instance.release_order = event_order
                assert key is not None
                current.pop(key, None)
                continue
            if instance is not None:
                if instance.release is None:
                    instance.release = _snapshot_release(history, instance)
                    instance.release_order = event_order
                assert key is not None
                current.pop(key, None)
            size = abs(entry.size_bytes)
            pool_id = _event_pool_id(entry, history.pool_ranges)
            birth = CohortBirth(
                before_index=history.before.index,
                after_index=history.after.index,
                before_label=history.before.label,
                after_label=history.after.label,
                stack_key=stack_key_from_frames(entry.frames, depth=stack_depth),
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
                stack_key=birth.stack_key,
                birth=birth,
                birth_order=event_order,
            )
            instances.append(item)
            current[(entry.device_index, entry.addr)] = item

        after_blocks = list(observations.get(history.after.index, ()))
        exact_blocks, fallback_blocks = _block_indexes(after_blocks)
        consumed_after: set[int] = set()
        next_current: dict[tuple[int | None, int], _AllocationInstance] = {}
        for key, instance in current.items():
            block_index = _matching_block_index(
                exact_blocks,
                fallback_blocks,
                instance,
                consumed_after,
            )
            if block_index is None:
                if instance.release is None:
                    event_order += 1
                    instance.release = _snapshot_release(history, instance)
                    instance.release_order = event_order
                continue
            block = after_blocks[block_index]
            consumed_after.add(block_index)
            instance.observations[block.point_index] = block
            _refine_instance_from_block(instance, block)
            assert block.address is not None
            next_current[(block.device, block.address)] = instance

        for instance in missing_address:
            if instance.release is None:
                event_order += 1
                instance.release = _snapshot_release(history, instance)
                instance.release_order = event_order
        missing_address = []

        for block_index, block in enumerate(after_blocks):
            if block_index in consumed_after:
                continue
            event_order += 1
            birth = CohortBirth(
                before_index=history.before.index,
                after_index=history.after.index,
                before_label=history.before.label,
                after_label=history.after.label,
                stack_key=block.stack_key,
                confidence="snapshot_inferred",
                size_bytes=block.size_bytes,
                count=1,
            )
            item = create_from_block(
                block,
                birth=birth,
                birth_order=event_order,
            )
            if block.address is None:
                missing_address.append(item)
            else:
                next_current[(block.device, block.address)] = item
        current = next_current
    return instances


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


def _event_size_matches(entry: TraceEntry, instance: _AllocationInstance) -> bool:
    return abs(entry.size_bytes) in {
        0,
        instance.size_bytes,
        instance.requested_bytes,
    }


def _refine_instance_from_block(
    instance: _AllocationInstance, block: _BlockObservation
) -> None:
    if instance.pool_id == ("unknown",):
        instance.pool_id = block.pool_id
    if instance.stack_key == "<unattributed>" and block.stack_key != "<unattributed>":
        instance.stack_key = block.stack_key
    instance.device = block.device
    instance.stream = block.stream
    instance.address = block.address
    instance.size_bytes = block.size_bytes
    instance.requested_bytes = block.requested_bytes


def _event_pool_id(
    entry: TraceEntry,
    pool_ranges: PoolRangeIndex | None,
) -> tuple[Any, ...]:
    if entry.pool_id is not None:
        return entry.pool_id
    if entry.addr is None or pool_ranges is None:
        return ("unknown",)
    return pool_ranges.find(entry.device_index, entry.addr) or ("unknown",)


def _release_from_event(
    history: _IntervalHistory,
    instance: _AllocationInstance,
    entry: TraceEntry,
    *,
    stack_depth: int,
) -> CohortRelease:
    return CohortRelease(
        before_index=history.before.index,
        after_index=history.after.index,
        before_label=history.before.label,
        after_label=history.after.label,
        stack_key=stack_key_from_frames(entry.frames, depth=stack_depth),
        confidence="event_exact",
        size_bytes=instance.size_bytes,
        count=1,
    )


def _snapshot_release(
    history: _IntervalHistory, instance: _AllocationInstance
) -> CohortRelease:
    return CohortRelease(
        before_index=history.before.index,
        after_index=history.after.index,
        before_label=history.before.label,
        after_label=history.after.label,
        stack_key="<unavailable>",
        confidence="snapshot_inferred",
        size_bytes=instance.size_bytes,
        count=1,
    )


def _cohorts(
    instances: Sequence[_AllocationInstance],
    points: Sequence[MemoryPoint],
    *,
    active_at: MemoryPoint | None,
    born_between: tuple[MemoryPoint, MemoryPoint] | None,
    limit: int,
) -> tuple[AllocationCohort, ...]:
    grouped: dict[tuple[object, ...], list[_AllocationInstance]] = defaultdict(list)
    for instance in instances:
        grouped[(instance.device, instance.pool_id, instance.stack_key)].append(
            instance
        )

    pending = []
    for (device, pool_id, stack_key), members in grouped.items():
        point_states = []
        for point in points:
            active = [
                member.observations[point.index]
                for member in members
                if point.index in member.observations
            ]
            point_states.append(
                CohortPointState(
                    point_index=point.index,
                    point_label=point.label,
                    active_bytes=sum(item.size_bytes for item in active),
                    requested_bytes=sum(item.requested_bytes for item in active),
                    block_count=len(active),
                )
            )

        nonzero = [item for item in point_states if item.block_count]
        sizes: Counter[tuple[int, int]] = Counter(
            (member.size_bytes, member.requested_bytes) for member in members
        )
        births = _aggregate_births(
            member.birth for member in members if member.birth is not None
        )
        releases = _aggregate_releases(
            member.release for member in members if member.release is not None
        )
        exact_births = [
            member
            for member in members
            if member.birth and member.birth.confidence == "event_exact"
        ]
        inferred_births = [
            member
            for member in members
            if member.birth and member.birth.confidence == "snapshot_inferred"
        ]
        exact = [
            member
            for member in members
            if member.release and member.release.confidence == "event_exact"
        ]
        inferred = [
            member
            for member in members
            if member.release and member.release.confidence == "snapshot_inferred"
        ]
        last_index = points[-1].index
        still_active = [
            member for member in members if last_index in member.observations
        ]
        active_values = [item.active_bytes for item in point_states]
        impact = max(active_values) - min(active_values)
        born_bytes = sum(member.size_bytes for member in members if member.birth)
        if born_between is not None:
            anchor_bytes = born_bytes
        elif active_at is not None:
            anchor_bytes = next(
                item.active_bytes
                for item in point_states
                if item.point_index == active_at.index
            )
        else:
            anchor_bytes = impact
        if nonzero:
            first_seen_index = nonzero[0].point_index
            first_seen_label = nonzero[0].point_label
            last_seen_index = nonzero[-1].point_index
            last_seen_label = nonzero[-1].point_label
        else:
            member_births = [member.birth for member in members if member.birth]
            assert member_births
            first_birth = min(member_births, key=lambda item: item.after_index)
            last_boundary = max(
                (
                    member.release.after_index
                    if member.release is not None
                    else member.birth.after_index
                )
                for member in members
                if member.birth is not None
            )
            point_labels = {point.index: point.label for point in points}
            first_seen_index = first_birth.after_index
            first_seen_label = first_birth.after_label
            last_seen_index = last_boundary
            last_seen_label = point_labels.get(last_boundary, first_birth.after_label)
        pending.append(
            {
                "device": device,
                "pool_id": pool_id,
                "stack_key": stack_key,
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
                "births": births,
                "releases": releases,
                "nonzero": nonzero,
                "exact_births": exact_births,
                "inferred_births": inferred_births,
                "exact": exact,
                "inferred": inferred,
                "still_active": still_active,
                "impact": impact,
                "anchor_bytes": anchor_bytes,
                "born_bytes": born_bytes,
                "first_seen_index": first_seen_index,
                "first_seen_label": first_seen_label,
                "last_seen_index": last_seen_index,
                "last_seen_label": last_seen_label,
                "peak_live_bytes": _peak_live_bytes(members, points[0].index),
            }
        )

    pending.sort(
        key=lambda item: (
            -int(item["anchor_bytes"]),
            -max(point.active_bytes for point in item["points"]),
            str(item["device"]),
            str(item["pool_id"]),
            str(item["stack_key"]),
        )
    )
    result = []
    for index, item in enumerate(pending[:limit], start=1):
        point_states = item["points"]
        members = item["members"]
        exact_births = item["exact_births"]
        inferred_births = item["inferred_births"]
        exact = item["exact"]
        inferred = item["inferred"]
        still_active = item["still_active"]
        result.append(
            AllocationCohort(
                cohort_id=f"cohort-{index:04d}",
                device=item["device"],
                pool_id=item["pool_id"],
                stack_key=item["stack_key"],
                streams=tuple(sorted({member.stream for member in members}, key=str)),
                points=point_states,
                size_histogram=item["sizes"],
                births=item["births"],
                releases=item["releases"],
                peak_active_bytes=max(point.active_bytes for point in point_states),
                peak_live_bytes=item["peak_live_bytes"],
                peak_block_count=max(point.block_count for point in point_states),
                impact_bytes=item["impact"],
                first_seen_index=item["first_seen_index"],
                first_seen_label=item["first_seen_label"],
                last_seen_index=item["last_seen_index"],
                last_seen_label=item["last_seen_label"],
                born_bytes=item["born_bytes"],
                born_count=len(exact_births) + len(inferred_births),
                event_exact_birth_bytes=sum(
                    member.size_bytes for member in exact_births
                ),
                event_exact_birth_count=len(exact_births),
                snapshot_inferred_birth_bytes=sum(
                    member.size_bytes for member in inferred_births
                ),
                snapshot_inferred_birth_count=len(inferred_births),
                event_exact_release_bytes=sum(member.size_bytes for member in exact),
                event_exact_release_count=len(exact),
                snapshot_inferred_release_bytes=sum(
                    member.size_bytes for member in inferred
                ),
                snapshot_inferred_release_count=len(inferred),
                still_active_bytes=sum(member.size_bytes for member in still_active),
                still_active_count=len(still_active),
            )
        )
    return tuple(result)


def _peak_live_bytes(members: Sequence[_AllocationInstance], start_index: int) -> int:
    live = sum(
        member.size_bytes
        for member in members
        if member.birth is None and start_index in member.observations
    )
    peak = live
    transitions = []
    for member in members:
        if member.birth_order is not None:
            transitions.append((member.birth_order, member.size_bytes))
        if member.release_order is not None:
            transitions.append((member.release_order, -member.size_bytes))
    for _order, delta in sorted(transitions):
        live += delta
        peak = max(peak, live)
    return peak


TransitionT = TypeVar("TransitionT", bound=_CohortTransition)


def _aggregate_transitions(
    transitions: Sequence[TransitionT],
    kind: type[TransitionT],
) -> tuple[TransitionT, ...]:
    totals: dict[tuple[object, ...], list[int]] = defaultdict(lambda: [0, 0])
    examples: dict[tuple[object, ...], TransitionT] = {}
    for transition in transitions:
        key = (
            transition.before_index,
            transition.after_index,
            transition.before_label,
            transition.after_label,
            transition.stack_key,
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
                before_index=example.before_index,
                after_index=example.after_index,
                before_label=example.before_label,
                after_label=example.after_label,
                stack_key=example.stack_key,
                confidence=example.confidence,
                size_bytes=size_bytes,
                count=count,
            )
        )
    rows.sort(
        key=lambda item: (
            item.after_index,
            -item.size_bytes,
            item.confidence,
            item.stack_key,
        )
    )
    return tuple(rows)


def _aggregate_births(births: Sequence[CohortBirth]) -> tuple[CohortBirth, ...]:
    return _aggregate_transitions(births, CohortBirth)


def _aggregate_releases(
    releases: Sequence[CohortRelease],
) -> tuple[CohortRelease, ...]:
    return _aggregate_transitions(releases, CohortRelease)
