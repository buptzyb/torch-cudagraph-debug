"""Allocator event-window extraction and attribution."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._pool_ranges import PoolRangeIndex, build_pool_range_index
from .allocator_snapshot import (
    AllocatorSnapshotData,
    AllocatorTraceEntry,
    normalize_device_trace_entries,
    pool_id_label,
    raw_device_trace,
    stack_key_from_frames,
    stream_label,
)


@dataclass(frozen=True)
class EventWindow:
    """Events delimited by two recorder metadata markers."""

    entries: tuple[AllocatorTraceEntry, ...]
    available: bool
    complete: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _WindowBounds:
    start: int
    end: int
    complete: bool
    ordered: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AllocatorEventSummary:
    """Allocator events grouped by pool, stream, action, and event stack."""

    pool_id: tuple[Any, ...]
    stream: Any
    action: str
    stack_key: str
    size_bytes: int
    count: int
    attribution_confidence: str

    def to_row(
        self, *, reference_label: str, candidate_label: str
    ) -> dict[str, object]:
        return {
            "reference_label": reference_label,
            "candidate_label": candidate_label,
            "pool_id": pool_id_label(self.pool_id),
            "stream_id": stream_label(self.stream),
            "action": self.action,
            "stack_key": self.stack_key,
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
    """Extract events after a prior marker through the current snapshot point."""

    if not entries:
        return _unavailable_window()
    bounds = _window_bounds(
        len(entries),
        start=_last_marker_index(entries, start_marker),
        end=_last_marker_index(entries, end_marker),
        start_label=start_label,
    )
    return EventWindow(
        entries=(
            tuple(entries[bounds.start + 1 : bounds.end]) if bounds.ordered else ()
        ),
        available=True,
        complete=bounds.complete,
        warnings=bounds.warnings,
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
    )


def _unavailable_window() -> EventWindow:
    return EventWindow(
        entries=(),
        available=False,
        complete=False,
        warnings=("allocator event history is unavailable",),
    )


def _window_bounds(
    length: int,
    *,
    start: int | None,
    end: int | None,
    start_label: str,
) -> _WindowBounds:
    warnings: list[str] = []
    complete = True
    if start is None:
        complete = False
        warnings.append(
            f"event boundary for record {start_label!r} was not found; "
            "memory history may be disabled or its ring buffer may have "
            "overwritten the marker"
        )
        start = -1
    if end is None:
        end = length
    elif end < start:
        warnings.append("allocator event boundary order is inconsistent")
        return _WindowBounds(
            start=start,
            end=end,
            complete=False,
            ordered=False,
            warnings=tuple(warnings),
        )
    return _WindowBounds(
        start=start,
        end=end,
        complete=complete,
        ordered=True,
        warnings=tuple(warnings),
    )


def summarize_allocator_events(
    entries: Sequence[AllocatorTraceEntry],
    *,
    reference_segments: Sequence[Mapping[str, Any]],
    candidate_segments: Sequence[Mapping[str, Any]],
    stack_depth: int = 2,
    top: int | None = 20,
) -> tuple[AllocatorEventSummary, ...]:
    """Aggregate historical events separately from active allocation stacks."""

    if type(stack_depth) is not int:
        raise TypeError("stack_depth must be an integer")
    if stack_depth < 1:
        raise ValueError("stack_depth must be >= 1")
    if top is not None and type(top) is not int:
        raise TypeError("top must be an integer or None")
    if top is not None and top < 1:
        raise ValueError("top must be >= 1")
    ranges = build_pool_range_index(candidate_segments, reference_segments)
    totals: dict[tuple[tuple[Any, ...], Any, str, str, str], Counter[str]] = (
        defaultdict(Counter)
    )
    for entry in entries:
        if entry.action == "snapshot":
            continue
        if entry.pool_id is not None:
            pool_id, confidence = entry.pool_id, "reported"
        else:
            pool_id, confidence = _attribute_pool(
                entry.device_index, entry.addr, ranges
            )
        stack_key = stack_key_from_frames(entry.frames, depth=stack_depth)
        key = (pool_id, entry.stream, entry.action, stack_key, confidence)
        totals[key]["count"] += 1
        totals[key]["size"] += abs(entry.size_bytes)

    rows = tuple(
        sorted(
            (
                AllocatorEventSummary(
                    pool_id=pool_id,
                    stream=stream,
                    action=action,
                    stack_key=stack_key,
                    size_bytes=int(values["size"]),
                    count=int(values["count"]),
                    attribution_confidence=confidence,
                )
                for (
                    pool_id,
                    stream,
                    action,
                    stack_key,
                    confidence,
                ), values in totals.items()
            ),
            key=lambda item: (
                -item.size_bytes,
                -item.count,
                pool_id_label(item.pool_id),
                stream_label(item.stream),
                item.action,
                item.stack_key,
            ),
        )
    )
    return rows if top is None else rows[:top]


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
) -> tuple[tuple[Any, ...], str]:
    if addr is None:
        return ("unknown",), "unknown"
    pool_id = ranges.find(device_index, addr)
    if pool_id is not None:
        return pool_id, "matched"
    return ("unknown",), "unknown"
