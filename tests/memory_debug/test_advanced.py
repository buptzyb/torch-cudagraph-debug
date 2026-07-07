from __future__ import annotations

import json

import pytest
from torch_cudagraph_debug.memory_debug._pool_ranges import build_pool_range_index
from torch_cudagraph_debug.memory_debug.advanced import (
    compare_allocation_stacks,
    normalize_trace_entries,
    pool_id_label,
    stack_key_from_frames,
    summarize_allocation_stacks,
    summarize_allocator_events,
    summarize_snapshot,
)

from ._helpers import segment, snapshot


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

    summary = summarize_allocation_stacks(before, pool_id=(0, 3))
    deltas = compare_allocation_stacks(before, after)

    assert len(summary) == 1
    assert summary[0].size_bytes == 30
    assert summary[0].stream is None
    assert len(deltas) == 1
    assert deltas[0].delta_size_bytes == 10


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
