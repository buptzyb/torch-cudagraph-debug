from __future__ import annotations

import pytest
from torch_cudagraph_debug.memory_debug._pool_ranges import build_pool_range_index
from torch_cudagraph_debug.memory_debug.advanced import (
    compare_allocation_stacks,
    normalize_trace_entries,
    pool_id_label,
    stack_key_from_frames,
    summarize_allocation_stacks,
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


@pytest.mark.parametrize(
    "kwargs",
    ({"stack_depth": True}, {"top": 0}, {"top": 1.5}),
)
def test_advanced_stack_options_reject_lossy_types(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        summarize_allocation_stacks(snapshot(), **kwargs)  # type: ignore[arg-type]


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


def test_stack_key_rejects_boolean_depth() -> None:
    frames = [{"filename": "model.py", "line": 1, "name": "forward"}]

    with pytest.raises(TypeError, match="stack depth"):
        stack_key_from_frames(frames, depth=True)


def test_trace_normalization_rejects_invalid_device_trace_container() -> None:
    malformed = snapshot()
    malformed["device_traces"] = ["not-a-trace"]

    with pytest.raises(TypeError, match=r"device_traces\[0\]"):
        normalize_trace_entries(malformed)
