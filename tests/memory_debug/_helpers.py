from __future__ import annotations

from pathlib import Path
from typing import Any

from torch_cudagraph_debug.memory_debug import MemoryRecorder, MemoryRun


def segment(
    *,
    active: int,
    total: int | None = None,
    pool: tuple[int, int] = (0, 0),
    stream: int = 0,
    address: int = 1000,
    frame: str | None = "model.py",
    requested: int | None = None,
    device: int = 0,
) -> dict[str, Any]:
    total = active if total is None else total
    requested = active if requested is None else requested
    blocks: list[dict[str, Any]] = []
    if active:
        blocks.append(
            {
                "address": address,
                "size": active,
                "requested_size": requested,
                "state": "active_allocated",
                "frames": (
                    [{"filename": frame, "line": 10, "name": "forward"}]
                    if frame is not None
                    else []
                ),
            }
        )
    if total > active:
        blocks.append(
            {
                "address": address + active,
                "size": total - active,
                "requested_size": 0,
                "state": "inactive",
                "frames": [],
            }
        )
    return {
        "address": address,
        "device": device,
        "stream": stream,
        "segment_pool_id": list(pool),
        "segment_type": "large",
        "total_size": total,
        "allocated_size": active,
        "active_size": active,
        "requested_size": requested,
        "blocks": blocks,
    }


def snapshot(
    *segments: dict[str, Any],
    traces: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    return {
        "segments": list(segments),
        "device_traces": traces or [],
        "external_annotations": [],
        "allocator_settings": {},
    }


def event(
    action: str,
    *,
    marker: str = "",
    address: int = 1000,
    size: int = 0,
    stream: int = 0,
    pool: tuple[int, int] | None = None,
    frame: str | None = "alloc.py",
    name: str = "allocate",
    time_us: int = 1,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "action": action,
        "addr": address,
        "size": size,
        "stream": stream,
        "time_us": time_us,
        "user_metadata": marker,
        "frames": (
            [{"filename": frame, "line": 7, "name": name}]
            if action != "snapshot" and frame is not None
            else []
        ),
    }
    if pool is not None:
        value["pool_id"] = list(pool)
    return value


def make_run(
    snapshots: list[dict[str, Any]],
    *,
    name: str = "run",
    bundle_dir: Path | None = None,
    labels: tuple[str, ...] | None = None,
    rank: int | None = None,
    group_id: str | None = None,
    world_size: int | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> MemoryRun:
    pending = list(snapshots)
    recorder = MemoryRecorder._from_snapshot_provider(
        lambda marker: pending.pop(0),
        name=name,
        bundle_dir=bundle_dir,
        rank=rank,
        group_id=group_id,
        world_size=world_size,
        run_metadata=run_metadata,
    )
    point_labels = labels or tuple(f"point_{index}" for index in range(len(pending)))
    for label in point_labels:
        recorder.record_point(label)
    return recorder.finish()


def make_history_run(
    steps: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    *,
    labels: tuple[str, ...],
    name: str = "run",
    bundle_dir: Path | None = None,
) -> MemoryRun:
    """Build a run whose snapshots carry a cumulative, marker-complete trace.

    Each step is ``(segments, events_since_previous_point)``. Marker snapshot
    events are appended automatically so every interval window is complete.
    """

    markers: list[str] = []
    trace: list[dict[str, Any]] = []

    def provider(marker: str) -> dict[str, Any]:
        index = len(markers)
        markers.append(marker)
        segments, events_between = steps[index]
        trace.extend(events_between)
        trace.append(event("snapshot", marker=marker))
        return snapshot(*segments, traces=[list(trace)])

    recorder = MemoryRecorder._from_snapshot_provider(
        provider, name=name, bundle_dir=bundle_dir
    )
    for label in labels:
        recorder.record_point(label)
    return recorder.finish()
