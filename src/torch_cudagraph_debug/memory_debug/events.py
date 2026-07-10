"""Allocator event-window extraction and attribution."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ._pool_identity import PoolId
from ._pool_ranges import PoolRangeIndex, build_pool_range_index
from ._stack_trace import (
    display_stack,
    normalize_stack_frames,
    stack_fingerprint,
    stack_frames_json,
    stack_frames_payload,
    stack_key,
)
from .allocator_snapshot import (
    AllocatorSnapshotData,
    AllocatorTraceEntry,
    normalize_device_trace_entries,
    normalize_raw_trace_entries,
    pool_id_label,
    raw_device_trace,
    stream_label,
)

WindowCause = Literal[
    "disabled",
    "boundary_unavailable",
    "truncated",
    "invalid_boundary_order",
]
HistoryWindowStatus = Literal[
    "complete",
    "disabled",
    "boundary_unavailable",
    "truncated",
    "invalid_boundary_order",
]
HISTORY_WINDOW_STATUSES = frozenset(HistoryWindowStatus.__args__)


@dataclass(frozen=True)
class EventWindow:
    """Events between a start boundary marker and either an explicit end
    marker or the end of the trace.

    A snapshot's own end marker is never visible in its own trace; recorder
    windows therefore run from the start marker to the trace end.
    """

    entries: tuple[AllocatorTraceEntry, ...]
    available: bool
    complete: bool
    warnings: tuple[str, ...]
    cause: WindowCause | None = None


@dataclass(frozen=True)
class _WindowBounds:
    start: int
    end: int
    complete: bool
    ordered: bool
    warnings: tuple[str, ...]
    cause: WindowCause | None = None


@dataclass(frozen=True)
class _DeviceEventEvidence:
    device_index: int
    status: HistoryWindowStatus
    warnings: tuple[str, ...]
    entries: tuple[Mapping[str, Any], ...]
    trace_index_offset: int = 0

    @property
    def event_count(self) -> int:
        return len(self.entries)

    def window(self) -> EventWindow:
        return EventWindow(
            entries=(
                normalize_raw_trace_entries(
                    self.entries,
                    self.device_index,
                    trace_index_offset=self.trace_index_offset,
                )
                if self.status == "complete"
                else ()
            ),
            available=self.status != "disabled",
            complete=self.status == "complete",
            warnings=self.warnings,
            cause=None if self.status == "complete" else self.status,
        )


@dataclass(frozen=True)
class _PointEventEvidence:
    start_index: int
    end_index: int
    devices: tuple[_DeviceEventEvidence, ...]

    def windows(self) -> tuple[EventWindow, ...]:
        return tuple(device.window() for device in self.devices)


def _extract_point_event_evidence(
    snapshot: AllocatorSnapshotData,
    *,
    devices: Sequence[int],
    previous_boundary_recorded: bool,
    current_boundary_recorded: bool,
    start_marker: str,
    end_marker: str,
    start_label: str,
    end_label: str,
    start_index: int,
    end_index: int,
) -> _PointEventEvidence:
    """Extract raw allocator evidence owned by the ending point."""

    evidence = []
    for device_index in devices:
        raw_entries = raw_device_trace(snapshot, device_index)
        status: HistoryWindowStatus
        warnings: tuple[str, ...]
        selected: tuple[Mapping[str, Any], ...] = ()
        trace_index_offset = 0
        if not raw_entries:
            status = "disabled"
            warnings = ("allocator event history is unavailable",)
        elif not previous_boundary_recorded:
            status = "boundary_unavailable"
            warnings = (
                f"event boundary for record {start_label!r} was not recorded; "
                "the interval is excluded from event analysis",
            )
        elif not current_boundary_recorded:
            status = "boundary_unavailable"
            warnings = (
                f"event boundary for record {end_label!r} was not recorded; "
                "the interval is excluded from event analysis",
            )
        else:
            start = _last_raw_marker_index(raw_entries, start_marker)
            end = _last_raw_marker_index(raw_entries, end_marker)
            if start is None:
                status = "truncated"
                warnings = (
                    f"event boundary for record {start_label!r} was not found in "
                    "the device trace; the history ring buffer has overwritten it "
                    "or history recording was enabled after the boundary, and the "
                    "interval is excluded from event analysis",
                )
            elif end is not None and end <= start:
                status = "invalid_boundary_order"
                warnings = ("allocator event boundary order is inconsistent",)
            else:
                # The ending snapshot's trace terminates at the boundary that
                # produced it: on the real torch path the end marker enters
                # history as the snapshot is taken and is not visible in the
                # snapshot's own trace. Recorder semantics therefore use the
                # trace end as this interval's endpoint when metadata setup
                # succeeded. This assumes the application kept allocator
                # history enabled throughout the interval; PyTorch exposes no
                # continuity signal here. A present end marker supplied by a
                # provider trims the window instead.
                upper = len(raw_entries) if end is None else end
                trace_index_offset = start + 1
                selected_rows = []
                for trace_index, entry in enumerate(
                    raw_entries[start + 1 : upper], start=start + 1
                ):
                    if not isinstance(entry, Mapping):
                        raise TypeError(
                            f"device_traces[{device_index}][{trace_index}] must be "
                            "a mapping"
                        )
                    selected_rows.append(dict(entry))
                selected = tuple(selected_rows)
                status = "complete"
                warnings = ()
        evidence.append(
            _DeviceEventEvidence(
                device_index=device_index,
                status=status,
                warnings=warnings,
                entries=selected,
                trace_index_offset=trace_index_offset,
            )
        )
    return _PointEventEvidence(
        start_index=start_index,
        end_index=end_index,
        devices=tuple(evidence),
    )


def _extract_snapshot_event_windows(
    snapshot: AllocatorSnapshotData,
    *,
    devices: Sequence[int],
    previous_boundary_recorded: bool,
    current_boundary_recorded: bool,
    start_marker: str,
    end_marker: str,
    start_label: str,
    end_label: str,
    start_index: int,
    end_index: int,
) -> tuple[EventWindow, ...]:
    """Classify one in-memory interval with Recorder-equivalent semantics."""

    return _extract_point_event_evidence(
        snapshot,
        devices=devices,
        previous_boundary_recorded=previous_boundary_recorded,
        current_boundary_recorded=current_boundary_recorded,
        start_marker=start_marker,
        end_marker=end_marker,
        start_label=start_label,
        end_label=end_label,
        start_index=start_index,
        end_index=end_index,
    ).windows()


@dataclass(frozen=True)
class AllocatorEventSummary:
    """Allocator events grouped by device, pool, stream, action, and stack."""

    device_index: int
    pool_id: PoolId | None
    stream: Any
    action: str
    stack_frames: tuple[Mapping[str, Any], ...]
    stack_fingerprint: str
    size_bytes: int
    count: int
    attribution_confidence: str

    @property
    def pool_label(self) -> str:
        if self.pool_id is not None:
            return pool_id_label(self.pool_id)
        if self.attribution_confidence == "not_applicable":
            return "pool[n/a]"
        return "pool[unknown]"

    @property
    def stack_key(self) -> str:
        return stack_key(self.stack_frames)

    def display_stack(self, depth: int) -> str:
        return display_stack(self.stack_frames, depth=depth)

    def to_dict(
        self, *, reference_label: str, candidate_label: str
    ) -> dict[str, object]:
        row = self.to_row(
            reference_label=reference_label,
            candidate_label=candidate_label,
        )
        row.pop("stack_frames_json")
        row["stack_frames"] = stack_frames_payload(self.stack_frames)
        return row

    def to_row(
        self, *, reference_label: str, candidate_label: str
    ) -> dict[str, object]:
        return {
            "reference_label": reference_label,
            "candidate_label": candidate_label,
            "device_index": self.device_index,
            "pool_id": self.pool_label,
            "stream_id": stream_label(self.stream),
            "action": self.action,
            "stack_key": self.stack_key,
            "stack_fingerprint": self.stack_fingerprint,
            "stack_frames_json": stack_frames_json(self.stack_frames),
            "size_bytes": self.size_bytes,
            "count": self.count,
            "attribution_confidence": self.attribution_confidence,
        }


def extract_event_window(
    entries: Sequence[AllocatorTraceEntry],
    *,
    start_marker: str | None,
    end_marker: str | None,
    start_label: str,
) -> EventWindow:
    """Extract events after the last start marker, through the last end
    marker or — with ``end_marker=None`` — the end of the trace.

    A requested end marker that is absent yields a truncated window.
    """

    if not entries:
        return _unavailable_window()
    bounds = _window_bounds(
        len(entries),
        start=_last_marker_index(entries, start_marker),
        end=_last_marker_index(entries, end_marker),
        start_label=start_label,
        end_marker_expected=bool(end_marker),
    )
    return EventWindow(
        entries=(
            tuple(entries[bounds.start + 1 : bounds.end]) if bounds.ordered else ()
        ),
        available=True,
        complete=bounds.complete,
        warnings=bounds.warnings,
        cause=bounds.cause,
    )


def extract_event_window_from_snapshot(
    snapshot: AllocatorSnapshotData,
    *,
    device_index: int,
    start_marker: str | None,
    end_marker: str | None,
    start_label: str,
) -> EventWindow:
    """Find raw marker bounds first, then normalize only the selected window."""

    raw_entries = raw_device_trace(snapshot, device_index)
    if not raw_entries:
        return _unavailable_window()
    bounds = _window_bounds(
        len(raw_entries),
        start=_last_raw_marker_index(raw_entries, start_marker),
        end=_last_raw_marker_index(raw_entries, end_marker),
        start_label=start_label,
        end_marker_expected=bool(end_marker),
    )
    return EventWindow(
        entries=(
            normalize_device_trace_entries(
                snapshot,
                device_index,
                start=bounds.start + 1,
                end=bounds.end,
            )
            if bounds.ordered
            else ()
        ),
        available=True,
        complete=bounds.complete,
        warnings=bounds.warnings,
        cause=bounds.cause,
    )


def _unavailable_window() -> EventWindow:
    return EventWindow(
        entries=(),
        available=False,
        complete=False,
        warnings=("allocator event history is unavailable",),
        cause="disabled",
    )


def _window_bounds(
    length: int,
    *,
    start: int | None,
    end: int | None,
    start_label: str,
    end_marker_expected: bool = False,
) -> _WindowBounds:
    warnings: list[str] = []
    if start is None:
        # The marker was overwritten in the bounded history ring buffer, or
        # history recording started after it. The remaining trace cannot be
        # trusted to cover the interval, so the window is excluded from event
        # analysis instead of replaying the whole ring from its truncated
        # start.
        warnings.append(
            f"event boundary for record {start_label!r} was not found in the "
            "device trace; the history ring buffer has overwritten it or "
            "history recording was enabled after the boundary, and the "
            "interval is excluded from event analysis"
        )
        return _WindowBounds(
            start=0,
            end=0,
            complete=False,
            ordered=False,
            warnings=tuple(warnings),
            cause="truncated",
        )
    if end is None:
        if end_marker_expected:
            # A requested end boundary is absent from this trace. On the
            # real torch path a snapshot's own boundary marker is never
            # visible in that snapshot's trace, so this is the expected
            # shape when the caller passes the ending snapshot's marker —
            # pass end_marker=None to read through the end of the trace.
            # It can also mean history recording stopped before the
            # boundary. Either way the window cannot claim the requested
            # coverage.
            warnings.append(
                "the requested end boundary was not found in the device "
                "trace, so the window cannot claim the requested coverage "
                "and is excluded from event analysis; if the end boundary "
                "is the snapshot that produced this trace, pass "
                "end_marker=None to read through the end of the trace"
            )
            return _WindowBounds(
                start=start,
                end=start,
                complete=False,
                ordered=False,
                warnings=tuple(warnings),
                cause="truncated",
            )
        end = length
    elif end <= start:
        # A shared index means one entry would serve as both boundaries —
        # the same-marker copy-paste case — which is as incoherent as a
        # reversed pair.
        warnings.append("allocator event boundary order is inconsistent")
        return _WindowBounds(
            start=start,
            end=end,
            complete=False,
            ordered=False,
            warnings=tuple(warnings),
            cause="invalid_boundary_order",
        )
    return _WindowBounds(
        start=start,
        end=end,
        complete=True,
        ordered=True,
        warnings=tuple(warnings),
    )


def summarize_allocator_events(
    entries: Sequence[AllocatorTraceEntry],
    *,
    reference_segments: Sequence[Mapping[str, Any]],
    candidate_segments: Sequence[Mapping[str, Any]],
) -> tuple[AllocatorEventSummary, ...]:
    """Aggregate historical events separately from active allocation stacks."""

    ranges = build_pool_range_index(candidate_segments, reference_segments)
    totals: dict[tuple[int, PoolId | None, Any, str, str, str], Counter[str]] = (
        defaultdict(Counter)
    )
    stack_frames: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for entry in entries:
        if entry.action == "snapshot":
            continue
        if entry.action == "oom":
            # OOM entries describe the device, not an allocator block, so
            # pool attribution is categorically meaningless for them.
            pool_id, confidence = None, "not_applicable"
        elif entry.pool_id is not None:
            pool_id, confidence = entry.pool_id, "reported"
        else:
            pool_id, confidence = _attribute_pool(
                entry.device_index, entry.addr, ranges
            )
        frames = normalize_stack_frames(entry.frames)
        fingerprint = stack_fingerprint(frames)
        stack_frames[fingerprint] = frames
        key = (
            entry.device_index,
            pool_id,
            entry.stream,
            entry.action,
            fingerprint,
            confidence,
        )
        totals[key]["count"] += 1
        totals[key]["size"] += abs(entry.size_bytes)

    return tuple(
        sorted(
            (
                AllocatorEventSummary(
                    device_index=device_index,
                    pool_id=pool_id,
                    stream=stream,
                    action=action,
                    stack_frames=stack_frames[fingerprint],
                    stack_fingerprint=fingerprint,
                    size_bytes=int(values["size"]),
                    count=int(values["count"]),
                    attribution_confidence=confidence,
                )
                for (
                    device_index,
                    pool_id,
                    stream,
                    action,
                    fingerprint,
                    confidence,
                ), values in totals.items()
            ),
            key=lambda item: (
                -item.size_bytes,
                -item.count,
                item.device_index,
                (pool_id_label(item.pool_id) if item.pool_id is not None else ""),
                stream_label(item.stream),
                item.action,
                item.stack_key,
                item.stack_fingerprint,
            ),
        )
    )


def _last_marker_index(
    entries: Sequence[AllocatorTraceEntry], marker: str | None
) -> int | None:
    if not marker:
        return None
    for index in range(len(entries) - 1, -1, -1):
        if entries[index].user_metadata == marker:
            return index
    return None


def _last_raw_marker_index(entries: Sequence[object], marker: str | None) -> int | None:
    if not marker:
        return None
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if isinstance(entry, Mapping) and str(entry.get("user_metadata", "")) == marker:
            return index
    return None


def _attribute_pool(
    device_index: int,
    addr: int | None,
    ranges: PoolRangeIndex,
) -> tuple[PoolId | None, str]:
    if addr is None:
        return None, "unknown"
    pool_id, ambiguous = ranges.resolve(device_index, addr)
    if pool_id is not None:
        return pool_id, "matched"
    if ambiguous:
        return None, "ambiguous"
    return None, "unknown"
