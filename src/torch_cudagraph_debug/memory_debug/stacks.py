"""Attribute active allocator blocks to their allocation call stacks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._pool_identity import PoolId
from .allocator_snapshot import (
    ACTIVE_STATES,
    AllocatorSnapshotData,
    format_bytes,
    format_delta_bytes,
    normalize_pool_id,
    normalize_snapshot,
    normalize_stream,
    pool_id_label,
    stack_key_from_frames,
    stream_label,
)


@dataclass(frozen=True)
class AllocationStackCoverage:
    """Bytes with and without allocation stacks at one snapshot point."""

    active_bytes: int
    attributed_bytes: int
    unattributed_bytes: int

    @property
    def ratio(self) -> float:
        if self.active_bytes == 0:
            return 1.0
        return self.attributed_bytes / self.active_bytes

    def to_row(self) -> dict[str, object]:
        return {
            "active_bytes": self.active_bytes,
            "attributed_bytes": self.attributed_bytes,
            "unattributed_bytes": self.unattributed_bytes,
            "coverage_ratio": self.ratio,
        }


@dataclass(frozen=True)
class AllocationStackSummary:
    """Active bytes grouped by pool, optional stream, and allocation stack."""

    pool_id: tuple[Any, ...]
    stream: Any | None
    stack_key: str
    size_bytes: int
    requested_bytes: int
    count: int

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "pool_id": pool_id_label(self.pool_id),
            "stack_key": self.stack_key,
            "size_bytes": self.size_bytes,
            "requested_bytes": self.requested_bytes,
            "count": self.count,
            "size": format_bytes(self.size_bytes),
            "requested": format_bytes(self.requested_bytes),
        }
        if self.stream is not None:
            row["stream_id"] = stream_label(self.stream)
        return row


@dataclass(frozen=True)
class AllocationStackDelta:
    """Reference/candidate delta for one allocation-stack bucket."""

    reference_pool_id: tuple[Any, ...]
    candidate_pool_id: tuple[Any, ...]
    stream: Any | None
    stack_key: str
    reference_size_bytes: int
    candidate_size_bytes: int
    delta_size_bytes: int
    reference_requested_bytes: int
    candidate_requested_bytes: int
    delta_requested_bytes: int
    reference_count: int
    candidate_count: int
    delta_count: int

    @property
    def has_changes(self) -> bool:
        return bool(
            self.delta_size_bytes or self.delta_requested_bytes or self.delta_count
        )

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "reference_pool_id": pool_id_label(self.reference_pool_id),
            "candidate_pool_id": pool_id_label(self.candidate_pool_id),
            "stack_key": self.stack_key,
            "reference_size_bytes": self.reference_size_bytes,
            "candidate_size_bytes": self.candidate_size_bytes,
            "delta_size_bytes": self.delta_size_bytes,
            "reference_requested_bytes": self.reference_requested_bytes,
            "candidate_requested_bytes": self.candidate_requested_bytes,
            "delta_requested_bytes": self.delta_requested_bytes,
            "reference_count": self.reference_count,
            "candidate_count": self.candidate_count,
            "delta_count": self.delta_count,
            "reference_size": format_bytes(self.reference_size_bytes),
            "candidate_size": format_bytes(self.candidate_size_bytes),
            "delta_size": format_delta_bytes(self.delta_size_bytes),
        }
        if self.stream is not None:
            row["stream_id"] = stream_label(self.stream)
        return row


@dataclass(frozen=True)
class _AllocationStackIndex:
    coverage: AllocationStackCoverage
    by_pool: Mapping[tuple[PoolId, str], AllocationStackSummary]
    by_pool_stream: Mapping[tuple[PoolId, Any, str], AllocationStackSummary]


def _build_allocation_stack_index(
    segments: Sequence[Mapping[str, Any]],
    *,
    stack_depth: int,
) -> _AllocationStackIndex:
    _validate_stack_depth(stack_depth)
    aggregate: dict[tuple[PoolId, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    detailed: dict[tuple[PoolId, Any, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    active = attributed = 0
    for segment in segments:
        pool_id = normalize_pool_id(segment.get("segment_pool_id"))
        stream = normalize_stream(segment.get("stream"))
        for block in segment.get("blocks", ()):
            if str(block.get("state")) not in ACTIVE_STATES:
                continue
            size = int(block.get("size", 0) or 0)
            requested = int(block.get("requested_size", 0) or 0)
            frames = block.get("frames") or ()
            active += size
            if frames:
                attributed += size
            stack_key = stack_key_from_frames(frames, depth=stack_depth)
            for bucket in (
                aggregate[(pool_id, stack_key)],
                detailed[(pool_id, stream, stack_key)],
            ):
                bucket[0] += size
                bucket[1] += requested
                bucket[2] += 1
    return _AllocationStackIndex(
        coverage=AllocationStackCoverage(
            active_bytes=active,
            attributed_bytes=attributed,
            unattributed_bytes=max(active - attributed, 0),
        ),
        by_pool={
            key: AllocationStackSummary(
                pool_id=key[0],
                stream=None,
                stack_key=key[1],
                size_bytes=value[0],
                requested_bytes=value[1],
                count=value[2],
            )
            for key, value in aggregate.items()
        },
        by_pool_stream={
            key: AllocationStackSummary(
                pool_id=key[0],
                stream=key[1],
                stack_key=key[2],
                size_bytes=value[0],
                requested_bytes=value[1],
                count=value[2],
            )
            for key, value in detailed.items()
        },
    )


def _compare_stack_indexes(
    reference: _AllocationStackIndex,
    candidate: _AllocationStackIndex,
    *,
    by_stream: bool,
    top: int | None,
    include_unchanged: bool = False,
) -> tuple[AllocationStackDelta, ...]:
    reference_rows = (
        reference.by_pool_stream.values() if by_stream else reference.by_pool.values()
    )
    candidate_rows = (
        candidate.by_pool_stream.values() if by_stream else candidate.by_pool.values()
    )
    return _compare_stack_rows(
        reference_rows,
        candidate_rows,
        top=top,
        include_unchanged=include_unchanged,
    )


def _compare_mapped_stack_indexes(
    reference: _AllocationStackIndex,
    candidate: _AllocationStackIndex,
    mapping: Mapping[PoolId, tuple[PoolId, str]],
    *,
    top: int | None,
) -> tuple[AllocationStackDelta, ...]:
    deltas: list[AllocationStackDelta] = []
    for reference_pool, (candidate_pool, _match) in mapping.items():
        reference_rows = {
            stack_key: row
            for (pool_id, stack_key), row in reference.by_pool.items()
            if pool_id == reference_pool
        }
        candidate_rows = {
            stack_key: row
            for (pool_id, stack_key), row in candidate.by_pool.items()
            if pool_id == candidate_pool
        }
        for stack_key in set(reference_rows) | set(candidate_rows):
            reference_row = reference_rows.get(stack_key)
            candidate_row = candidate_rows.get(stack_key)
            delta = _stack_delta(
                reference_pool_id=reference_pool,
                candidate_pool_id=candidate_pool,
                stream=None,
                stack_key=stack_key,
                reference=reference_row,
                candidate=candidate_row,
            )
            if delta.has_changes:
                deltas.append(delta)
    rows = tuple(sorted(deltas, key=_stack_delta_sort_key))
    return rows if top is None else rows[:top]


def allocation_stack_coverage(
    snapshot: AllocatorSnapshotData,
    *,
    pool_id: Sequence[Any] | Any | None = None,
) -> AllocationStackCoverage:
    """Measure active bytes whose allocation stack is available."""

    pool_filter = normalize_pool_id(pool_id) if pool_id is not None else None
    active = attributed = 0
    for segment in normalize_snapshot(snapshot):
        segment_pool = normalize_pool_id(segment.get("segment_pool_id"))
        if pool_filter is not None and segment_pool != pool_filter:
            continue
        for block in segment.get("blocks", ()):
            if str(block.get("state")) not in ACTIVE_STATES:
                continue
            size = int(block.get("size", 0) or 0)
            active += size
            if block.get("frames"):
                attributed += size
    return AllocationStackCoverage(
        active_bytes=active,
        attributed_bytes=attributed,
        unattributed_bytes=max(active - attributed, 0),
    )


def summarize_allocation_stacks(
    snapshot: AllocatorSnapshotData,
    *,
    pool_id: Sequence[Any] | Any | None = None,
    stream: Any | None = None,
    stack_depth: int = 2,
    by_stream: bool = False,
    top: int | None = None,
) -> tuple[AllocationStackSummary, ...]:
    """Group active blocks by allocation stack, aggregating streams by default."""

    _validate_stack_depth(stack_depth)
    _validate_top(top)
    pool_filter = normalize_pool_id(pool_id) if pool_id is not None else None
    stream_filter = normalize_stream(stream) if stream is not None else None
    totals: dict[tuple[tuple[Any, ...], Any | None, str], list[int]] = defaultdict(
        lambda: [0, 0, 0]
    )
    for segment in normalize_snapshot(snapshot):
        segment_pool = normalize_pool_id(segment.get("segment_pool_id"))
        segment_stream = normalize_stream(segment.get("stream"))
        if pool_filter is not None and segment_pool != pool_filter:
            continue
        if stream_filter is not None and segment_stream != stream_filter:
            continue
        output_stream = segment_stream if by_stream else None
        for block in segment.get("blocks", ()):
            if str(block.get("state")) not in ACTIVE_STATES:
                continue
            stack_key = stack_key_from_frames(
                block.get("frames") or (), depth=stack_depth
            )
            bucket = totals[(segment_pool, output_stream, stack_key)]
            bucket[0] += int(block.get("size", 0) or 0)
            bucket[1] += int(block.get("requested_size", 0) or 0)
            bucket[2] += 1
    rows = tuple(
        sorted(
            (
                AllocationStackSummary(
                    pool_id=key[0],
                    stream=key[1],
                    stack_key=key[2],
                    size_bytes=value[0],
                    requested_bytes=value[1],
                    count=value[2],
                )
                for key, value in totals.items()
            ),
            key=lambda item: (
                -item.size_bytes,
                -item.requested_bytes,
                -item.count,
                pool_id_label(item.pool_id),
                "" if item.stream is None else stream_label(item.stream),
                item.stack_key,
            ),
        )
    )
    return rows if top is None else rows[:top]


def compare_allocation_stacks(
    reference: AllocatorSnapshotData,
    candidate: AllocatorSnapshotData,
    *,
    pool_id: Sequence[Any] | Any | None = None,
    stream: Any | None = None,
    stack_depth: int = 2,
    by_stream: bool = False,
    top: int | None = None,
    include_unchanged: bool = False,
) -> tuple[AllocationStackDelta, ...]:
    """Compare active allocation-stack buckets between two snapshots."""

    _validate_stack_depth(stack_depth)
    _validate_top(top)
    if type(by_stream) is not bool or type(include_unchanged) is not bool:
        raise TypeError("by_stream and include_unchanged must be booleans")
    reference_rows = summarize_allocation_stacks(
        reference,
        pool_id=pool_id,
        stream=stream,
        stack_depth=stack_depth,
        by_stream=by_stream,
    )
    candidate_rows = summarize_allocation_stacks(
        candidate,
        pool_id=pool_id,
        stream=stream,
        stack_depth=stack_depth,
        by_stream=by_stream,
    )
    return _compare_stack_rows(
        reference_rows,
        candidate_rows,
        top=top,
        include_unchanged=include_unchanged,
    )


def _compare_stack_rows(
    reference_rows: Iterable[AllocationStackSummary],
    candidate_rows: Iterable[AllocationStackSummary],
    *,
    top: int | None,
    include_unchanged: bool,
) -> tuple[AllocationStackDelta, ...]:
    reference_map = {
        (row.pool_id, row.stream, row.stack_key): row for row in reference_rows
    }
    candidate_map = {
        (row.pool_id, row.stream, row.stack_key): row for row in candidate_rows
    }
    deltas = []
    for key in set(reference_map) | set(candidate_map):
        delta = _stack_delta(
            reference_pool_id=key[0],
            candidate_pool_id=key[0],
            stream=key[1],
            stack_key=key[2],
            reference=reference_map.get(key),
            candidate=candidate_map.get(key),
        )
        if include_unchanged or delta.has_changes:
            deltas.append(delta)
    rows = tuple(sorted(deltas, key=_stack_delta_sort_key))
    return rows if top is None else rows[:top]


def _stack_delta(
    *,
    reference_pool_id: PoolId,
    candidate_pool_id: PoolId,
    stream: Any | None,
    stack_key: str,
    reference: AllocationStackSummary | None,
    candidate: AllocationStackSummary | None,
) -> AllocationStackDelta:
    reference_size = reference.size_bytes if reference else 0
    candidate_size = candidate.size_bytes if candidate else 0
    reference_requested = reference.requested_bytes if reference else 0
    candidate_requested = candidate.requested_bytes if candidate else 0
    reference_count = reference.count if reference else 0
    candidate_count = candidate.count if candidate else 0
    return AllocationStackDelta(
        reference_pool_id=reference_pool_id,
        candidate_pool_id=candidate_pool_id,
        stream=stream,
        stack_key=stack_key,
        reference_size_bytes=reference_size,
        candidate_size_bytes=candidate_size,
        delta_size_bytes=candidate_size - reference_size,
        reference_requested_bytes=reference_requested,
        candidate_requested_bytes=candidate_requested,
        delta_requested_bytes=candidate_requested - reference_requested,
        reference_count=reference_count,
        candidate_count=candidate_count,
        delta_count=candidate_count - reference_count,
    )


def _stack_delta_sort_key(
    item: AllocationStackDelta,
) -> tuple[int, int, int, str, str, str]:
    return (
        -abs(item.delta_size_bytes),
        -abs(item.delta_requested_bytes),
        -abs(item.delta_count),
        pool_id_label(item.candidate_pool_id),
        "" if item.stream is None else stream_label(item.stream),
        item.stack_key,
    )


def _validate_stack_depth(value: int) -> None:
    if type(value) is not int:
        raise TypeError("stack_depth must be an integer")
    if value < 1:
        raise ValueError("stack_depth must be >= 1")


def _validate_top(value: int | None) -> None:
    if value is None:
        return
    if type(value) is not int:
        raise TypeError("top must be an integer or None")
    if value < 1:
        raise ValueError("top must be >= 1")
