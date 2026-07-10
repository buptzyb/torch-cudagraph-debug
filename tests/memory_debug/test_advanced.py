from __future__ import annotations

import json

import pytest

from torch_cudagraph_debug.memory_debug import MemoryPoolKey
from torch_cudagraph_debug.memory_debug._pool_ranges import build_pool_range_index
from torch_cudagraph_debug.memory_debug.advanced import (
    compare_allocation_stacks,
    extract_event_window,
    mutable_snapshot,
    normalize_snapshot,
    normalize_trace_entries,
    pool_id_label,
    stack_key_from_frames,
    summarize_allocation_stacks,
    summarize_allocator_events,
    summarize_segments,
    summarize_snapshot,
)

from ._helpers import event, make_run, segment, snapshot


def test_snapshot_summary_groups_every_observed_pool_and_stream() -> None:
    value = snapshot(
        segment(active=10, pool=(0, 0), stream=0),
        segment(active=20, pool=(0, 3), stream=7, address=3000),
        segment(active=30, pool=(0, 3), stream=9, address=6000),
    )

    rows = summarize_snapshot(value)

    assert len(rows) == 3
    assert sorted(item.active_bytes for item in rows.values()) == [10, 20, 30]
    assert {(pool_id_label(item.pool_id), item.stream) for item in rows} == {
        ("pool[0,0]", 0),
        ("pool[0,3]", 7),
        ("pool[0,3]", 9),
    }


def test_allocation_stacks_aggregate_streams_by_default() -> None:
    before = snapshot(
        segment(
            active=10,
            pool=(0, 3),
            stream=7,
            address=3000,
            frame="same.py",
        ),
        segment(
            active=20,
            pool=(0, 3),
            stream=9,
            address=6000,
            frame="same.py",
        ),
    )
    after = snapshot(
        segment(
            active=40,
            pool=(0, 3),
            stream=11,
            address=9000,
            frame="same.py",
        )
    )

    summary = summarize_allocation_stacks(before, pool=MemoryPoolKey(0, (0, 3)))
    deltas = compare_allocation_stacks(before, after)

    assert len(summary) == 1
    assert summary[0].size_bytes == 30
    assert summary[0].stream is None
    assert len(deltas) == 1
    assert deltas[0].delta_size_bytes == 10


def test_extract_event_window_distinguishes_missing_end_marker() -> None:
    value = snapshot(
        segment(active=64),
        traces=[
            [
                event("snapshot", marker="start"),
                event("alloc", address=9000, size=64),
            ]
        ],
    )
    entries = normalize_trace_entries(value)

    # No end marker requested: the window intentionally stays open through
    # the end of the trace.
    open_window = extract_event_window(
        entries,
        start_marker="start",
        end_marker=None,
        start_label="start",
    )
    assert open_window.complete is True
    assert len(open_window.entries) == 1

    # An end marker was requested but never found: the window must not claim
    # complete coverage.
    missing_end = extract_event_window(
        entries,
        start_marker="start",
        end_marker="end",
        start_label="start",
    )
    assert missing_end.complete is False
    assert missing_end.cause == "truncated"
    # The warning must not assert a specific cause (the natural torch shape
    # never contains the ending snapshot's own marker) and must point at
    # the end_marker=None escape hatch.
    assert any("end_marker=None" in warning for warning in missing_end.warnings)


def test_summarize_segments_requires_structural_sizes() -> None:
    # Defaulting the structural sizes to zero would fabricate a valid-looking
    # all-zero group from corrupted input.
    with pytest.raises(TypeError, match=r"segment\[0\].total_size is required"):
        summarize_segments([{"device": 0, "blocks": []}])
    with pytest.raises(TypeError, match=r"segment\[1\].allocated_size is required"):
        summarize_segments(
            [
                segment(active=10),
                {"device": 0, "total_size": 100, "active_size": 100, "blocks": []},
            ]
        )


def test_extract_event_window_rejects_shared_boundary_marker() -> None:
    value = snapshot(
        segment(active=64),
        traces=[
            [
                event("snapshot", marker="boundary"),
                event("alloc", address=9000, size=64),
            ]
        ],
    )
    entries = normalize_trace_entries(value)

    # The same marker for both bounds resolves to one entry serving as start
    # and end; that must not be classified as a complete empty window.
    window = extract_event_window(
        entries,
        start_marker="boundary",
        end_marker="boundary",
        start_label="boundary",
    )
    assert window.complete is False
    assert window.cause == "invalid_boundary_order"
    assert window.entries == ()


def test_pool_range_prefers_exact_device_over_device_fallback() -> None:
    index = build_pool_range_index(
        (
            {
                "address": 1000,
                "total_size": 100,
                "device": None,
                "segment_pool_id": (9, 9),
            },
            {
                "address": 1000,
                "total_size": 100,
                "device": 0,
                "segment_pool_id": (1, 1),
            },
        )
    )

    assert index.find(0, 1050) == (1, 1)
    assert index.find(1, 1050) == (9, 9)


@pytest.mark.parametrize("name", ("stack_depth", "top"))
def test_advanced_stack_helpers_reject_removed_lossy_options(name: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        summarize_allocation_stacks(snapshot(), **{name: 1})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field,value,error",
    (
        ("total_size", "10", TypeError),
        ("active_size", True, TypeError),
        ("allocated_size", -1, ValueError),
        ("blocks", "not-a-sequence", TypeError),
    ),
)
def test_snapshot_normalization_rejects_invalid_present_fields(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    malformed = segment(active=10)
    malformed[field] = value

    with pytest.raises(error):
        summarize_snapshot(snapshot(malformed))


def test_block_address_inference_resynchronizes_after_explicit_address() -> None:
    value = segment(active=200, address=1000)
    value.pop("address")
    value["blocks"] = [
        {
            "address": 1100,
            "size": 100,
            "requested_size": 100,
            "state": "active_allocated",
            "frames": [],
        },
        {
            "size": 100,
            "requested_size": 100,
            "state": "active_allocated",
            "frames": [],
        },
    ]

    normalized = normalize_snapshot(snapshot(value))
    assert [block["address"] for block in normalized[0]["blocks"]] == [1100, 1200]


def test_stack_normalization_rejects_invalid_frame_fields() -> None:
    malformed = segment(active=10)
    malformed["blocks"][0]["frames"][0]["filename"] = 123

    with pytest.raises(TypeError, match="frame.filename"):
        summarize_allocation_stacks(snapshot(malformed))


def test_stack_key_and_advanced_helpers_always_use_complete_stacks() -> None:
    shared = {"filename": "shared.py", "line": 1, "name": "allocate"}
    left = {"filename": "left.py", "line": 2, "name": "forward"}
    right = {"filename": "right.py", "line": 3, "name": "forward"}
    first = segment(active=10, address=1000)
    second = segment(active=20, address=2000)
    first["blocks"][0]["frames"] = [shared, left]
    second["blocks"][0]["frames"] = [shared, right]

    rows = summarize_allocation_stacks(snapshot(first, second))

    assert len(rows) == 2
    assert {row.stack_key for row in rows} == {
        stack_key_from_frames([shared, left]),
        stack_key_from_frames([shared, right]),
    }

    raw = snapshot(
        traces=[
            [
                {
                    "action": "alloc",
                    "addr": 1000,
                    "size": 10,
                    "stream": 0,
                    "frames": [shared, left],
                    "pool_id": [0, 0],
                },
                {
                    "action": "alloc",
                    "addr": 2000,
                    "size": 20,
                    "stream": 0,
                    "frames": [shared, right],
                    "pool_id": [0, 0],
                },
            ]
        ]
    )
    event_rows = summarize_allocator_events(
        normalize_trace_entries(raw),
        reference_segments=(),
        candidate_segments=(),
    )

    assert len(event_rows) == 2
    assert {row.stack_key for row in event_rows} == {
        stack_key_from_frames([shared, left]),
        stack_key_from_frames([shared, right]),
    }


def test_trace_normalization_rejects_invalid_device_trace_container() -> None:
    malformed = snapshot()
    malformed["device_traces"] = ["not-a-trace"]

    with pytest.raises(TypeError, match=r"device_traces\[0\]"):
        normalize_trace_entries(malformed)


def test_oom_device_free_is_not_an_address_and_never_pool_attributes() -> None:
    # Real torch OOM entries carry the free-byte count under ``device_free``
    # and have no ``addr``; the byte count must not leak into address-based
    # pool attribution even when it numerically falls inside a segment range.
    value = snapshot(segment(active=8192, address=4096, pool=(3, 7)))
    value["device_traces"] = [
        [
            {
                "action": "oom",
                "device_free": 5000,
                "size": 1 << 30,
                "stream": 0,
                "time_us": 1,
                "user_metadata": "",
                "frames": [],
            }
        ]
    ]

    (entry,) = normalize_trace_entries(value)
    assert entry.addr is None
    assert entry.device_free_bytes == 5000

    (row,) = summarize_allocator_events(
        [entry],
        reference_segments=normalize_snapshot(value),
        candidate_segments=normalize_snapshot(value),
    )
    assert row.action == "oom"
    assert row.pool_id is None
    assert row.attribution_confidence == "not_applicable"


def test_full_stack_identity_includes_fx_frame_metadata() -> None:
    first = segment(active=10, address=1000)
    second = segment(active=10, address=2000)
    first["blocks"][0]["frames"] = [
        {
            "filename": "model.py",
            "line": 10,
            "name": "forward",
            "fx_node_op": "call_module",
            "fx_node_name": "left",
            "fx_original_trace": "model.left",
        }
    ]
    second["blocks"][0]["frames"] = [
        {
            "filename": "model.py",
            "line": 10,
            "name": "forward",
            "fx_node_op": "call_module",
            "fx_node_name": "right",
            "fx_original_trace": "model.right",
        }
    ]

    rows = summarize_allocation_stacks(snapshot(first, second))

    assert len(rows) == 2
    assert len({row.stack_fingerprint for row in rows}) == 2
    assert {row.stack_key for row in rows} == {"model.py:10:forward"}
    assert {row.stack_frames[0]["fx_node_name"] for row in rows} == {
        "left",
        "right",
    }
    assert {row.to_dict()["stack_frames"][0]["fx_original_trace"] for row in rows} == {
        "model.left",
        "model.right",
    }
    assert {
        json.loads(row.to_row()["stack_frames_json"])[0]["fx_node_name"] for row in rows
    } == {"left", "right"}
    assert {row.display_stack(1) for row in rows} == {
        'model.py:10:forward [fx_node_op="call_module", '
        'fx_node_name="left", fx_original_trace="model.left"]',
        'model.py:10:forward [fx_node_op="call_module", '
        'fx_node_name="right", fx_original_trace="model.right"]',
    }
    assert [row.stack_fingerprint for row in rows] == sorted(
        row.stack_fingerprint for row in rows
    )

    raw = snapshot(
        traces=[
            [
                {
                    "action": "alloc",
                    "addr": 1000,
                    "size": 10,
                    "stream": 0,
                    "frames": list(first["blocks"][0]["frames"]),
                    "pool_id": [0, 0],
                },
                {
                    "action": "alloc",
                    "addr": 2000,
                    "size": 10,
                    "stream": 0,
                    "frames": list(second["blocks"][0]["frames"]),
                    "pool_id": [0, 0],
                },
            ]
        ]
    )
    event_rows = summarize_allocator_events(
        normalize_trace_entries(raw),
        reference_segments=(),
        candidate_segments=(),
    )

    assert {row.stack_frames[0]["fx_node_name"] for row in event_rows} == {
        "left",
        "right",
    }
    assert {
        json.loads(
            row.to_row(reference_label="before", candidate_label="after")[
                "stack_frames_json"
            ]
        )[0]["fx_node_name"]
        for row in event_rows
    } == {"left", "right"}


def test_mutable_snapshot_returns_independent_editable_copy() -> None:
    run = make_run(
        [snapshot(segment(active=10))],
        labels=("point",),
    )
    point = run["point"]

    mutable = mutable_snapshot(point)
    mutable["segments"][0]["total_size"] = 99

    assert point.allocator_state()["segments"][0]["total_size"] == 10


def test_summary_reports_expandable_segments_and_inactive_blocks() -> None:
    expandable = segment(active=40, total=128, address=1000)
    expandable["is_expandable"] = True
    regular = segment(active=16, total=32, address=2000)

    rows = summarize_snapshot(snapshot(expandable, regular))
    stats = rows[next(iter(rows))]

    assert stats.segment_count == 2
    assert stats.block_count == 4
    assert stats.inactive_block_count == 2
    assert stats.expandable_segment_count == 1
    assert stats.expandable_reserved_bytes == 128
    assert stats.largest_inactive_block_bytes == 88


def test_summary_separates_awaiting_free_from_inactive_memory() -> None:
    value = segment(active=64, total=96)
    value["allocated_size"] = 40
    value["active_size"] = 64
    value["requested_size"] = 52
    value["blocks"] = [
        {
            "address": 1000,
            "size": 40,
            "requested_size": 32,
            "state": "active_allocated",
            "frames": [],
        },
        {
            "address": 1040,
            "size": 24,
            "requested_size": 20,
            "state": "active_awaiting_free",
            "frames": [],
        },
        {
            "address": 1064,
            "size": 32,
            "requested_size": 0,
            "state": "inactive",
            "frames": [],
        },
    ]

    stats = next(iter(summarize_snapshot(snapshot(value)).values()))

    assert stats.allocated_bytes == 40
    assert stats.active_bytes == 64
    assert stats.awaiting_free_bytes == 24
    assert stats.inactive_bytes == 32
    assert stats.requested_bytes == 52
    assert stats.internal_fragmentation_bytes == 12
    assert stats.to_dict()["awaiting_free_bytes"] == 24


def test_expandable_inactive_bytes_separate_expandable_capacity() -> None:
    from torch_cudagraph_debug.memory_debug import MemoryStats

    expandable = segment(active=50, total=200, address=1000)
    expandable["is_expandable"] = True
    native = segment(active=50, total=100, address=4000)

    rows = summarize_snapshot(snapshot(expandable, native))
    stats = MemoryStats.combine(rows.values())

    # Inactive capacity is partitioned by expandable versus native ownership.
    assert stats.inactive_bytes == 200
    assert stats.expandable_inactive_bytes == 150
    assert stats.expandable_reserved_bytes == 200
    assert stats.inactive_bytes - stats.expandable_inactive_bytes == 50


def test_expandable_inactive_bytes_invariants() -> None:
    from torch_cudagraph_debug.memory_debug import MemoryStats

    with pytest.raises(ValueError, match="expandable_reserved_bytes"):
        MemoryStats(
            reserved_bytes=100,
            expandable_reserved_bytes=10,
            expandable_inactive_bytes=20,
        )
    with pytest.raises(ValueError, match="inactive bytes"):
        MemoryStats(
            reserved_bytes=100,
            active_bytes=90,
            allocated_bytes=90,
            requested_bytes=90,
            expandable_reserved_bytes=100,
            expandable_inactive_bytes=20,
        )
