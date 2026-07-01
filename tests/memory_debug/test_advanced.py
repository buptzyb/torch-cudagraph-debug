from __future__ import annotations

from torch_cudagraph_debug.memory_debug.advanced import (
    compare_allocation_stacks,
    pool_id_label,
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
