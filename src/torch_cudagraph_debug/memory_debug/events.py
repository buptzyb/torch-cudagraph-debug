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
    pool_id_label,
    raw_device_trace,
    stream_label,
)


WindowCause = Literal[
    "history_disabled",
    "start_marker_missing",
    "boundary_order",
]


@dataclass(frozen=True)
class EventWindow:
    """Events delimited by two recorder metadata markers."""

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
            "pool_id": (
                pool_id_label(self.pool_id)
                if self.pool_id is not None
                else "pool[unknown]"
            ),
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
        cause="history_disabled",
    )


def _window_bounds(
    length: int,
    *,
    start: int | None,
    end: int | None,
    start_label: str,
) -> _WindowBounds:
    warnings: list[str] = []
    if start is None:
        # The marker was overwritten in the bounded history ring buffer. The
        # remaining trace cannot be trusted to cover the interval, so the
        # window is excluded from event analysis instead of replaying the
        # whole ring from its truncated start.
        warnings.append(
            f"event boundary for record {start_label!r} was not found in the "
            "device trace; the history ring buffer has overwritten it and the "
            "interval is excluded from event analysis"
        )
        return _WindowBounds(
            start=0,
            end=0,
            complete=False,
            ordered=False,
            warnings=tuple(warnings),
            cause="start_marker_missing",
        )
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
            cause="boundary_order",
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
        if entry.pool_id is not None:
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
    pool_id = ranges.find(device_index, addr)
    if pool_id is not None:
        return pool_id, "matched"
    return None, "unknown"
