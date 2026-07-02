"""Normalize and compare PyTorch CUDA allocator snapshots."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._pool_identity import (
    DEFAULT_POOL_ID,
    PoolId,
    UNKNOWN_STREAM,
    normalize_pool_id,
    normalize_stream,
    pool_id_label as pool_id_label,
    stream_label as stream_label,
)
from .comparison_models import MemoryLifecycleDelta
from .stats import MemoryStats

AllocatorSnapshotData = Sequence[Mapping[str, Any]] | Mapping[str, Any]
ACTIVE_STATES = frozenset(
    {"active_allocated", "active_awaiting_free", "active_pending_free"}
)


@dataclass(frozen=True)
class MemoryObservationKey:
    """Stable grouping key for one allocator pool on one CUDA stream."""

    pool_id: PoolId
    stream: Any


@dataclass(frozen=True)
class AllocatorTraceEntry:
    """Normalized allocator trace event from ``_snapshot()["device_traces"]``."""

    device_index: int
    trace_index: int
    action: str
    addr: int | None
    size_bytes: int
    stream: Any
    frames: tuple[Mapping[str, Any], ...]
    time_us: int | None
    user_metadata: str
    pool_id: PoolId | None = None


def normalize_snapshot(
    snapshot: AllocatorSnapshotData,
) -> tuple[Mapping[str, Any], ...]:
    """Return normalized segment dictionaries from a PyTorch snapshot shape."""

    segments = _segments_from_snapshot(snapshot)
    normalized = []
    for segment in segments:
        pool_id = normalize_pool_id(segment.get("segment_pool_id", DEFAULT_POOL_ID))
        stream = normalize_stream(segment.get("stream", UNKNOWN_STREAM))
        address = _optional_int(segment.get("address"))
        blocks = []
        block_address = address
        for block in segment.get("blocks", []) or []:
            current_address = _optional_int(block.get("address"))
            if current_address is None and block_address is not None:
                current_address = block_address
            size = _int(block.get("size"))
            blocks.append(
                {
                    "address": current_address,
                    "size": size,
                    "requested_size": _int(block.get("requested_size")),
                    "state": str(block.get("state", "unknown")),
                    "frames": tuple(block.get("frames") or ()),
                }
            )
            if block_address is not None:
                block_address += size
        normalized.append(
            {
                "address": address,
                "device": _optional_int(segment.get("device")),
                "stream": stream,
                "segment_pool_id": pool_id,
                "segment_type": str(segment.get("segment_type", "unknown")),
                "total_size": _int(segment.get("total_size")),
                "allocated_size": _int(segment.get("allocated_size")),
                "active_size": _int(segment.get("active_size")),
                "requested_size": _int(segment.get("requested_size")),
                "blocks": tuple(blocks),
                "frames": tuple(segment.get("frames") or ()),
            }
        )
    return tuple(normalized)


def normalize_trace_entries(
    snapshot: AllocatorSnapshotData,
) -> tuple[AllocatorTraceEntry, ...]:
    """Flatten ``device_traces`` from a private PyTorch snapshot dict."""

    if not isinstance(snapshot, Mapping):
        return ()
    entries: list[AllocatorTraceEntry] = []
    for device_index in trace_device_indices(snapshot):
        entries.extend(normalize_device_trace_entries(snapshot, device_index))
    return tuple(entries)


def trace_device_indices(snapshot: AllocatorSnapshotData) -> tuple[int, ...]:
    """Return devices with at least one raw allocator trace entry."""

    if not isinstance(snapshot, Mapping):
        return ()
    raw_traces = snapshot.get("device_traces", ()) or ()
    if not isinstance(raw_traces, Sequence) or isinstance(
        raw_traces, (str, bytes, bytearray)
    ):
        return ()
    return tuple(
        device_index
        for device_index, device_trace in enumerate(raw_traces)
        if isinstance(device_trace, Sequence)
        and not isinstance(device_trace, (str, bytes, bytearray))
        and bool(device_trace)
    )


def raw_device_trace(
    snapshot: AllocatorSnapshotData, device_index: int
) -> Sequence[object]:
    """Return one raw device trace without normalizing its entries."""

    if not isinstance(snapshot, Mapping) or device_index < 0:
        return ()
    raw_traces = snapshot.get("device_traces", ()) or ()
    if not isinstance(raw_traces, Sequence) or isinstance(
        raw_traces, (str, bytes, bytearray)
    ):
        return ()
    if device_index >= len(raw_traces):
        return ()
    device_trace = raw_traces[device_index]
    if not isinstance(device_trace, Sequence) or isinstance(
        device_trace, (str, bytes, bytearray)
    ):
        return ()
    return device_trace


def normalize_device_trace_entries(
    snapshot: AllocatorSnapshotData,
    device_index: int,
    *,
    start: int = 0,
    end: int | None = None,
) -> tuple[AllocatorTraceEntry, ...]:
    """Normalize a bounded slice of one device trace."""

    device_trace = raw_device_trace(snapshot, device_index)
    lower = max(start, 0)
    upper = (
        len(device_trace) if end is None else min(max(end, lower), len(device_trace))
    )
    entries = []
    for trace_index in range(lower, upper):
        raw = device_trace[trace_index]
        if not isinstance(raw, Mapping):
            continue
        entries.append(_normalize_trace_entry(raw, device_index, trace_index))
    return tuple(entries)


def _normalize_trace_entry(
    raw: Mapping[str, Any], device_index: int, trace_index: int
) -> AllocatorTraceEntry:
    addr = raw.get("addr")
    if addr is None:
        addr = raw.get("device_free")
    return AllocatorTraceEntry(
        device_index=device_index,
        trace_index=trace_index,
        action=str(raw.get("action", "unknown")),
        addr=_optional_int(addr),
        size_bytes=_int(raw.get("size")),
        stream=normalize_stream(raw.get("stream", UNKNOWN_STREAM)),
        frames=tuple(raw.get("frames") or ()),
        time_us=_optional_int(raw.get("time_us")),
        user_metadata=str(raw.get("user_metadata", "")),
        pool_id=(
            normalize_pool_id(raw.get("pool_id"))
            if raw.get("pool_id") is not None
            else None
        ),
    )


def summarize_snapshot(
    snapshot: AllocatorSnapshotData,
) -> dict[MemoryObservationKey, MemoryStats]:
    """Summarize a snapshot by ``(pool, stream)`` group."""

    return summarize_segments(normalize_snapshot(snapshot))


def summarize_segments(
    segments: Sequence[Mapping[str, Any]],
) -> dict[MemoryObservationKey, MemoryStats]:
    grouped: dict[MemoryObservationKey, list[Mapping[str, Any]]] = defaultdict(list)
    for segment in segments:
        key = MemoryObservationKey(
            normalize_pool_id(segment.get("segment_pool_id", DEFAULT_POOL_ID)),
            normalize_stream(segment.get("stream", UNKNOWN_STREAM)),
        )
        grouped[key].append(segment)
    return {key: _summarize_group(items) for key, items in grouped.items()}


def compare_observation_lifecycle(
    reference_segments: Sequence[Mapping[str, Any]],
    candidate_segments: Sequence[Mapping[str, Any]],
) -> dict[MemoryObservationKey, MemoryLifecycleDelta]:
    """Compare segment/block identities in one pass per snapshot."""

    counters: defaultdict[MemoryObservationKey, list[int]] = defaultdict(
        lambda: [0, 0, 0, 0]
    )
    reference_segment_keys = {_segment_key(segment) for segment in reference_segments}
    candidate_segment_keys = {_segment_key(segment) for segment in candidate_segments}
    for key in candidate_segment_keys - reference_segment_keys:
        counters[key[0]][0] += key[2]
    for key in reference_segment_keys - candidate_segment_keys:
        counters[key[0]][1] += key[2]

    reference_blocks = _block_map(reference_segments)
    candidate_blocks = _block_map(candidate_segments)
    for block_key, block in candidate_blocks.items():
        if str(block.get("state")) not in ACTIVE_STATES:
            continue
        reference_block = reference_blocks.get(block_key)
        reference_state = (
            str(reference_block.get("state", "missing"))
            if reference_block
            else "missing"
        )
        if reference_state not in ACTIVE_STATES:
            counters[block_key[0]][2] += _int(block.get("size"))
    for block_key, block in reference_blocks.items():
        if str(block.get("state")) not in ACTIVE_STATES:
            continue
        candidate_block = candidate_blocks.get(block_key)
        candidate_state = (
            str(candidate_block.get("state", "missing"))
            if candidate_block
            else "missing"
        )
        if candidate_state == "inactive":
            counters[block_key[0]][3] += _int(block.get("size"))

    return {
        key: MemoryLifecycleDelta(
            new_segment_bytes=values[0],
            removed_segment_bytes=values[1],
            newly_active_bytes=values[2],
            released_bytes=values[3],
        )
        for key, values in counters.items()
    }


def frame_location(frame: Mapping[str, Any]) -> str:
    filename = str(frame.get("filename", "<unknown>"))
    line = _int(frame.get("line"))
    name = str(frame.get("name", "<module>"))
    return f"{filename}:{line}:{name}"


def stack_key_from_frames(
    frames: Sequence[Mapping[str, Any]], *, depth: int = 2
) -> str:
    if depth < 1:
        raise ValueError("stack depth must be >= 1")
    if not frames:
        return "<unattributed>"
    return " <- ".join(frame_location(frame) for frame in frames[:depth])


def format_bytes(value: int) -> str:
    sign = "-" if value < 0 else ""
    amount = float(abs(value))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{sign}{int(amount)} B"
    return f"{sign}{amount:.2f} {unit}"


def format_delta_bytes(value: int) -> str:
    if value > 0:
        return "+" + format_bytes(value)
    return format_bytes(value)


def format_comparison(reference: int, candidate: int, delta: int) -> str:
    return (
        f"{format_bytes(reference)} -> {format_bytes(candidate)} "
        f"(delta {format_delta_bytes(delta)})"
    )


def _segments_from_snapshot(
    snapshot: AllocatorSnapshotData,
) -> Sequence[Mapping[str, Any]]:
    if isinstance(snapshot, Mapping):
        segments = snapshot.get("segments", [])
    else:
        segments = snapshot
    return [segment for segment in segments or [] if isinstance(segment, Mapping)]


def _summarize_group(
    segments: Sequence[Mapping[str, Any]],
) -> MemoryStats:
    reserved = allocated = active = requested = block_count = largest_inactive = 0
    for segment in segments:
        reserved += _int(segment.get("total_size"))
        allocated += _int(segment.get("allocated_size"))
        active += _int(segment.get("active_size"))
        for block in segment.get("blocks", []) or []:
            state = str(block.get("state", "unknown"))
            size = _int(block.get("size"))
            block_count += 1
            if state in ACTIVE_STATES:
                requested += _int(block.get("requested_size"))
            elif state == "inactive":
                largest_inactive = max(largest_inactive, size)
    return MemoryStats(
        reserved_bytes=reserved,
        allocated_bytes=allocated,
        active_bytes=active,
        requested_bytes=requested,
        segment_count=len(segments),
        block_count=block_count,
        largest_inactive_block_bytes=largest_inactive,
    )


def _segment_key(
    segment: Mapping[str, Any],
) -> tuple[MemoryObservationKey, int | None, int]:
    return (
        MemoryObservationKey(
            normalize_pool_id(segment.get("segment_pool_id")),
            normalize_stream(segment.get("stream")),
        ),
        _optional_int(segment.get("address")),
        _int(segment.get("total_size")),
    )


def _block_map(
    segments: Sequence[Mapping[str, Any]],
) -> dict[tuple[MemoryObservationKey, int | None, int], Mapping[str, Any]]:
    result = {}
    for segment in segments:
        key = MemoryObservationKey(
            normalize_pool_id(segment.get("segment_pool_id")),
            normalize_stream(segment.get("stream")),
        )
        for block in segment.get("blocks", []) or []:
            block_key = (
                key,
                _optional_int(block.get("address")),
                _int(block.get("size")),
            )
            result[block_key] = block
    return result


def _int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
