"""Normalize and compare PyTorch CUDA allocator snapshots."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._pool_identity import (
    DEFAULT_POOL_ID,
    MemoryObservationKey,
    PoolId,
    normalize_device_index,
    normalize_pool_id,
    normalize_stream,
)
from ._pool_identity import (
    pool_id_label as pool_id_label,
)
from ._pool_identity import (
    stream_label as stream_label,
)
from ._stack_trace import normalize_stack_frames, stack_key
from .comparison_models import MemoryLifecycleDelta
from .stats import MemoryStats

AllocatorSnapshotData = Sequence[Mapping[str, Any]] | Mapping[str, Any]
OWNER_ACTIVE_STATES = frozenset({"active_allocated"})
AWAITING_FREE_STATES = frozenset({"active_awaiting_free", "active_pending_free"})
ACTIVE_STATES = OWNER_ACTIVE_STATES | AWAITING_FREE_STATES

KNOWN_TRACE_ACTIONS = frozenset(
    {
        "alloc",
        "free_requested",
        "free_completed",
        "segment_alloc",
        "segment_free",
        "segment_map",
        "segment_unmap",
        "snapshot",
        "oom",
    }
)
ALLOCATION_LIFETIME_ACTIONS = frozenset({"alloc", "free_requested", "free_completed"})


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


_SEGMENT_DRIFT_FIELDS = (
    ("device", "device 0"),
    ("segment_pool_id", "the default pool"),
    ("requested_size", "the active size"),
)
_BLOCK_DRIFT_FIELDS = (
    ("requested_size", "the block size"),
    ("state", "'unknown'"),
)
# Structural sizes have no coherent substitute: defaulting them to zero
# breaks the allocated <= active <= reserved invariants downstream, so
# their absence is treated as corrupted input rather than schema drift.
_SEGMENT_REQUIRED_SIZE_FIELDS = ("total_size", "allocated_size", "active_size")


def normalize_snapshot(
    snapshot: AllocatorSnapshotData,
    *,
    warnings: list[str] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Return normalized segment dictionaries from a PyTorch snapshot shape.

    Absent identity and size fields keep their documented default values but
    are reported through ``warnings`` (aggregated per field) instead of being
    substituted silently; a field that is present with an invalid type or
    value still raises.
    """

    segments = _segments_from_snapshot(snapshot)
    missing: dict[tuple[str, str, str], int] = {}

    def count_missing(
        mapping: Mapping[str, Any],
        kind: str,
        fields: tuple[tuple[str, str], ...],
    ) -> None:
        if warnings is None:
            return
        for name, substitute in fields:
            if name not in mapping:
                key = (kind, name, substitute)
                missing[key] = missing.get(key, 0) + 1

    normalized = []
    for segment_index, segment in enumerate(segments):
        context = f"segment[{segment_index}]"
        for name in _SEGMENT_REQUIRED_SIZE_FIELDS:
            if name not in segment:
                raise TypeError(f"{context}.{name} is required")
        count_missing(segment, "segment", _SEGMENT_DRIFT_FIELDS)
        device = normalize_device_index(_int_field(segment, "device", context))
        pool_id = normalize_pool_id(segment.get("segment_pool_id", DEFAULT_POOL_ID))
        stream = normalize_stream(segment.get("stream", None))
        address = _optional_int_field(segment, "address", context)
        blocks = []
        block_address = address
        for block_index, block in enumerate(
            _mapping_sequence_field(segment, "blocks", context)
        ):
            block_context = f"{context}.blocks[{block_index}]"
            if "size" not in block:
                raise TypeError(f"{block_context}.size is required")
            count_missing(block, "block", _BLOCK_DRIFT_FIELDS)
            current_address = _optional_int_field(block, "address", block_context)
            if current_address is None and block_address is not None:
                current_address = block_address
            size = _int_field(block, "size", block_context)
            blocks.append(
                {
                    "address": current_address,
                    "size": size,
                    "requested_size": _int_field(
                        block, "requested_size", block_context, size
                    ),
                    "state": _string_field(block, "state", block_context, "unknown"),
                    "frames": _frames_field(block, block_context),
                }
            )
            if block_address is not None:
                block_address += size
        normalized.append(
            {
                "address": address,
                "device": device,
                "stream": stream,
                "segment_pool_id": pool_id,
                "segment_type": _string_field(
                    segment, "segment_type", context, "unknown"
                ),
                "total_size": _int_field(segment, "total_size", context),
                "allocated_size": _int_field(segment, "allocated_size", context),
                "active_size": _int_field(segment, "active_size", context),
                "requested_size": _int_field(
                    segment,
                    "requested_size",
                    context,
                    _int_field(segment, "active_size", context),
                ),
                "is_expandable": _bool_field(segment, "is_expandable", False),
                "blocks": tuple(blocks),
                "frames": _frames_field(segment, context),
            }
        )
    if warnings is not None:
        for (kind, name, substitute), count in sorted(missing.items()):
            warnings.append(
                f"allocator snapshot schema drift: {count} {kind}(s) missing "
                f"{name!r} (treated as {substitute})"
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
    raw_traces = snapshot.get("device_traces", ())
    if raw_traces is None:
        raise TypeError("device_traces must be a sequence when present")
    if not isinstance(raw_traces, Sequence) or isinstance(
        raw_traces, (str, bytes, bytearray)
    ):
        raise TypeError("device_traces must be a sequence")
    devices = []
    for device_index, device_trace in enumerate(raw_traces):
        if isinstance(device_trace, (str, bytes, bytearray)) or not isinstance(
            device_trace, Sequence
        ):
            raise TypeError(f"device_traces[{device_index}] must be a sequence")
        if device_trace:
            devices.append(device_index)
    return tuple(devices)


def raw_device_trace(
    snapshot: AllocatorSnapshotData, device_index: int
) -> Sequence[object]:
    """Return one raw device trace without normalizing its entries."""

    if not isinstance(snapshot, Mapping) or device_index < 0:
        return ()
    raw_traces = snapshot.get("device_traces", ())
    if raw_traces is None:
        raise TypeError("device_traces must be a sequence when present")
    if not isinstance(raw_traces, Sequence) or isinstance(
        raw_traces, (str, bytes, bytearray)
    ):
        raise TypeError("device_traces must be a sequence")
    if device_index >= len(raw_traces):
        return ()
    device_trace = raw_traces[device_index]
    if not isinstance(device_trace, Sequence) or isinstance(
        device_trace, (str, bytes, bytearray)
    ):
        raise TypeError(f"device_traces[{device_index}] must be a sequence")
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
            raise TypeError(
                f"device_traces[{device_index}][{trace_index}] must be a mapping"
            )
        entries.append(_normalize_trace_entry(raw, device_index, trace_index))
    return tuple(entries)


def normalize_raw_trace_entries(
    entries: Sequence[object],
    device_index: int,
    *,
    trace_index_offset: int = 0,
) -> tuple[AllocatorTraceEntry, ...]:
    """Normalize a preserved raw trace slice for one device."""

    normalized = []
    for offset, raw in enumerate(entries):
        trace_index = trace_index_offset + offset
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"event entries for device {device_index} at index {offset} "
                "must be a mapping"
            )
        normalized.append(_normalize_trace_entry(raw, device_index, trace_index))
    return tuple(normalized)


def _normalize_trace_entry(
    raw: Mapping[str, Any], device_index: int, trace_index: int
) -> AllocatorTraceEntry:
    addr = raw.get("addr")
    if addr is None:
        addr = raw.get("device_free")
    return AllocatorTraceEntry(
        device_index=device_index,
        trace_index=trace_index,
        action=_string_field(raw, "action", f"trace[{trace_index}]", "unknown"),
        addr=_optional_int_value(addr, f"trace[{trace_index}].addr"),
        size_bytes=_int_field(raw, "size", f"trace[{trace_index}]"),
        stream=normalize_stream(raw.get("stream", None)),
        frames=_frames_field(raw, f"trace[{trace_index}]"),
        time_us=_optional_int_field(raw, "time_us", f"trace[{trace_index}]"),
        user_metadata=_string_field(raw, "user_metadata", f"trace[{trace_index}]", ""),
        pool_id=(
            normalize_pool_id(raw.get("pool_id"))
            if raw.get("pool_id") is not None
            else None
        ),
    )


def summarize_snapshot(
    snapshot: AllocatorSnapshotData,
) -> dict[MemoryObservationKey, MemoryStats]:
    """Summarize a snapshot by ``(device, pool, stream)`` group."""

    return summarize_segments(normalize_snapshot(snapshot))


def summarize_segments(
    segments: Sequence[Mapping[str, Any]],
) -> dict[MemoryObservationKey, MemoryStats]:
    grouped: dict[MemoryObservationKey, list[Mapping[str, Any]]] = defaultdict(list)
    for segment in segments:
        key = MemoryObservationKey(
            normalize_device_index(_int(segment.get("device"))),
            normalize_pool_id(segment.get("segment_pool_id", DEFAULT_POOL_ID)),
            normalize_stream(segment.get("stream", None)),
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
    # Identity keys form multisets: entries sharing one key (notably
    # address-less segments or blocks) compare by multiplicity, so matched
    # pairs cancel and every excess instance counts once. A block whose
    # (address, size) key left the active multiset (turned inactive in
    # place, coalesced, re-split, or in a freed segment) stopped being
    # active; the newly-active side is the exact mirror.
    reference_segment_keys = Counter(
        _segment_key(segment) for segment in reference_segments
    )
    candidate_segment_keys = Counter(
        _segment_key(segment) for segment in candidate_segments
    )
    for key, count in (candidate_segment_keys - reference_segment_keys).items():
        counters[key[0]][0] += key[2] * count
    for key, count in (reference_segment_keys - candidate_segment_keys).items():
        counters[key[0]][1] += key[2] * count

    reference_active = _active_block_keys(reference_segments)
    candidate_active = _active_block_keys(candidate_segments)
    for key, count in (candidate_active - reference_active).items():
        counters[key[0]][2] += key[2] * count
    for key, count in (reference_active - candidate_active).items():
        counters[key[0]][3] += key[2] * count

    return {
        key: MemoryLifecycleDelta(
            new_segment_bytes=values[0],
            removed_segment_bytes=values[1],
            newly_active_bytes=values[2],
            became_inactive_bytes=values[3],
        )
        for key, values in counters.items()
    }


def stack_key_from_frames(frames: Sequence[Mapping[str, Any]]) -> str:
    """Return the complete normalized allocator stack key."""

    return stack_key(normalize_stack_frames(frames))


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
        segments = snapshot.get("segments", ())
    else:
        segments = snapshot
    if isinstance(segments, (str, bytes, bytearray)) or not isinstance(
        segments, Sequence
    ):
        raise TypeError("allocator snapshot segments must be a sequence")
    result = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise TypeError(f"allocator snapshot segment[{index}] must be a mapping")
        result.append(segment)
    return result


def _summarize_group(
    segments: Sequence[Mapping[str, Any]],
) -> MemoryStats:
    reserved = allocated = active = requested = block_count = 0
    inactive_block_count = largest_inactive = 0
    expandable_segment_count = expandable_reserved = 0
    for segment in segments:
        segment_reserved = _int(segment.get("total_size"))
        reserved += segment_reserved
        allocated += _int(segment.get("allocated_size"))
        active += _int(segment.get("active_size"))
        if bool(segment.get("is_expandable", False)):
            expandable_segment_count += 1
            expandable_reserved += segment_reserved
        for block in segment.get("blocks", []) or []:
            state = str(block.get("state", "unknown"))
            size = _int(block.get("size"))
            block_count += 1
            if state in ACTIVE_STATES:
                requested += _int(block.get("requested_size"))
            elif state == "inactive":
                inactive_block_count += 1
                largest_inactive = max(largest_inactive, size)
    return MemoryStats(
        reserved_bytes=reserved,
        allocated_bytes=allocated,
        active_bytes=active,
        requested_bytes=requested,
        segment_count=len(segments),
        block_count=block_count,
        inactive_block_count=inactive_block_count,
        largest_inactive_block_bytes=largest_inactive,
        expandable_segment_count=expandable_segment_count,
        expandable_reserved_bytes=expandable_reserved,
    )


def _segment_key(
    segment: Mapping[str, Any],
) -> tuple[MemoryObservationKey, int | None, int]:
    return (
        MemoryObservationKey(
            normalize_device_index(_int(segment.get("device"))),
            normalize_pool_id(segment.get("segment_pool_id")),
            normalize_stream(segment.get("stream")),
        ),
        _optional_int(segment.get("address")),
        _int(segment.get("total_size")),
    )


def _active_block_keys(
    segments: Sequence[Mapping[str, Any]],
) -> Counter[tuple[MemoryObservationKey, int | None, int]]:
    result: Counter[tuple[MemoryObservationKey, int | None, int]] = Counter()
    for segment in segments:
        key = MemoryObservationKey(
            normalize_device_index(_int(segment.get("device"))),
            normalize_pool_id(segment.get("segment_pool_id")),
            normalize_stream(segment.get("stream")),
        )
        for block in segment.get("blocks", []) or []:
            if str(block.get("state")) not in ACTIVE_STATES:
                continue
            result[
                (
                    key,
                    _optional_int(block.get("address")),
                    _int(block.get("size")),
                )
            ] += 1
    return result


def _int(value: Any) -> int:
    if value is None:
        return 0
    if type(value) is not int:
        raise TypeError("allocator value must be an integer")
    if value < 0:
        raise ValueError("allocator value must be non-negative")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("allocator value must be an integer or None")
    if value < 0:
        raise ValueError("allocator value must be non-negative")
    return value


def _int_field(
    value: Mapping[str, Any], name: str, context: str, default: int = 0
) -> int:
    if name not in value:
        return default
    raw = value[name]
    if type(raw) is not int:
        raise TypeError(f"{context}.{name} must be an integer")
    if raw < 0:
        raise ValueError(f"{context}.{name} must be non-negative")
    return raw


def _optional_int_field(
    value: Mapping[str, Any], name: str, context: str
) -> int | None:
    if name not in value or value[name] is None:
        return None
    return _optional_int_value(value[name], f"{context}.{name}")


def _optional_int_value(value: Any, context: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer or None")
    if value < 0:
        raise ValueError(f"{context} must be non-negative")
    return value


def _bool_field(value: Mapping[str, Any], name: str, default: bool) -> bool:
    if name not in value:
        return default
    raw = value[name]
    if type(raw) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return raw


def _string_field(
    value: Mapping[str, Any], name: str, context: str, default: str
) -> str:
    if name not in value:
        return default
    raw = value[name]
    if not isinstance(raw, str):
        raise TypeError(f"{context}.{name} must be a string")
    return raw


def _mapping_sequence_field(
    value: Mapping[str, Any], name: str, context: str
) -> tuple[Mapping[str, Any], ...]:
    if name not in value:
        return ()
    raw = value[name]
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise TypeError(f"{context}.{name} must be a sequence")
    result = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise TypeError(f"{context}.{name}[{index}] must be a mapping")
        result.append(item)
    return tuple(result)


def _frames_field(
    value: Mapping[str, Any], context: str
) -> tuple[Mapping[str, Any], ...]:
    return _mapping_sequence_field(value, "frames", context)
