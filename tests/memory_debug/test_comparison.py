from __future__ import annotations

import pytest

import torch_cudagraph_debug.memory_debug.summary as summary_module
from torch_cudagraph_debug.memory_debug import (
    AttributionOptions,
    MemoryDebugError,
    MemoryHistoryError,
    MemoryOwnershipError,
    MemoryPoint,
    MemoryRecorder,
    compare_phases,
    compare_points,
)

from ._helpers import event, make_run, segment, snapshot


def test_same_run_keeps_pool_totals_and_stream_deltas_separate() -> None:
    run = make_run(
        [
            snapshot(
                segment(active=10, stream=0, address=1000),
                segment(active=20, stream=7, address=2000),
            ),
            snapshot(
                segment(active=20, stream=0, address=1000),
                segment(active=10, stream=7, address=2000),
            ),
        ],
        labels=("before", "after"),
    )

    comparison = run.compare("before", "after")

    assert comparison.lifecycle_available is True
    assert comparison.pools[0].delta.active_bytes == 0
    by_stream = {
        item.before_stream: item.delta.active_bytes for item in comparison.pool_streams
    }
    assert by_stream == {0: 10, 7: -10}
    assert comparison.pools[0].before.active_bytes == 30
    assert comparison.pools[0].after.active_bytes == 30


def test_cross_run_matches_only_default_pool_without_mapping() -> None:
    before = make_run(
        [
            snapshot(
                segment(active=10),
                segment(
                    active=20,
                    pool=(0, 1),
                    stream=7,
                    address=9000,
                ),
            )
        ],
        name="before",
        labels=("point",),
    )
    after = make_run(
        [
            snapshot(
                segment(active=15),
                segment(
                    active=30,
                    pool=(0, 1),
                    stream=7,
                    address=12000,
                ),
            )
        ],
        name="after",
        labels=("point",),
    )

    comparison = compare_points(before["point"], after["point"])

    assert [item.match for item in comparison.pools].count("default") == 1
    private = [
        item
        for item in comparison.pools
        if item.before_pool_id == (0, 1) or item.after_pool_id == (0, 1)
    ]
    assert {item.match for item in private} == {"before_only", "after_only"}
    assert all(
        item.match in {"before_only", "after_only"} for item in comparison.pool_streams
    )
    assert comparison.lifecycle_available is False
    assert any("private pools are unmatched" in item for item in comparison.warnings)


def test_cross_run_totals_include_unmatched_private_pools() -> None:
    before = make_run(
        [
            snapshot(
                segment(active=10),
                segment(active=20, pool=(0, 1), address=2000),
            )
        ],
        labels=("point",),
    )
    after = make_run(
        [
            snapshot(
                segment(active=15),
                segment(active=40, pool=(0, 9), address=3000),
            )
        ],
        labels=("point",),
    )

    comparison = compare_points(before["point"], after["point"])
    totals = {item.scope: item for item in comparison.totals}

    assert totals["all"].before.active_bytes == 30
    assert totals["all"].after.active_bytes == 55
    assert totals["all"].delta.active_bytes == 25
    assert totals["default"].delta.active_bytes == 5
    assert totals["private"].delta.active_bytes == 20


def test_cross_run_private_pool_mapping_is_explicit_and_one_to_one() -> None:
    before = make_run(
        [
            snapshot(
                segment(active=10, pool=(0, 1)),
                segment(active=20, pool=(0, 2), address=3000),
            )
        ],
        labels=("point",),
    )
    after = make_run(
        [
            snapshot(
                segment(active=15, pool=(0, 8)),
                segment(active=25, pool=(0, 9), address=4000),
            )
        ],
        labels=("point",),
    )

    comparison = compare_points(
        before["point"],
        after["point"],
        pool_mapping={(0, 1): (0, 8), (0, 2): (0, 9)},
    )
    assert all(item.match == "mapped" for item in comparison.pools)
    assert [item.delta.active_bytes for item in comparison.pools] == [5, 5]

    with pytest.raises(ValueError, match="target.*more than once"):
        compare_points(
            before["point"],
            after["point"],
            pool_mapping={(0, 1): (0, 8), (0, 2): (0, 8)},
        )
    with pytest.raises(ValueError, match="does not exist"):
        compare_points(
            before["point"],
            after["point"],
            pool_mapping={(0, 7): (0, 8)},
        )


def test_compare_points_rejects_same_run_and_events() -> None:
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        labels=("before", "after"),
    )
    other = make_run(
        [snapshot(segment(active=30))],
        labels=("point",),
    )

    with pytest.raises(MemoryOwnershipError, match="independent runs"):
        compare_points(run["before"], run["after"])
    with pytest.raises(MemoryDebugError, match="events cannot"):
        compare_points(
            run["before"],
            other["point"],
            attribution=AttributionOptions(events=True),
        )


def test_missing_stack_history_warns_or_errors() -> None:
    run = make_run(
        [
            snapshot(segment(active=10, frame=None)),
            snapshot(segment(active=20, frame=None)),
        ],
        labels=("before", "after"),
    )

    warning = run.compare(
        "before",
        "after",
        attribution=AttributionOptions(stacks=True),
    )
    assert any("coverage is incomplete" in item for item in warning.warnings)
    assert warning.before_stack_coverage is not None
    assert warning.before_stack_coverage.ratio == 0.0

    with pytest.raises(MemoryHistoryError, match="coverage is incomplete"):
        run.compare(
            "before",
            "after",
            attribution=AttributionOptions(
                stacks=True,
                on_missing="error",
            ),
        )


def test_event_history_uses_point_boundary_marker() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        traces = (
            [[event("snapshot", marker=marker)]]
            if len(markers) == 1
            else [
                [
                    event("snapshot", marker=markers[0]),
                    event(
                        "alloc",
                        address=9000,
                        size=64,
                        stream=7,
                        pool=(0, 5),
                    ),
                ]
            ]
        )
        return snapshot(
            segment(
                active=64,
                pool=(0, 5),
                stream=7,
                address=9000,
            ),
            traces=traces,
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.mark("before")
    recorder.mark("after")
    run = recorder.finish()
    comparison = run.compare(
        "before",
        "after",
        attribution=AttributionOptions(events=True, on_missing="error"),
    )

    assert comparison.events_available is True
    assert comparison.events_complete is True
    assert len(comparison.allocator_events) == 1
    assert comparison.allocator_events[0].pool_id == (0, 5)
    assert comparison.allocator_events[0].action == "alloc"


def test_event_history_normalizes_only_the_marker_window(monkeypatch) -> None:
    markers: list[str] = []
    normalized = 0
    original = summary_module._normalize_trace_entry

    def counting_normalizer(*args, **kwargs):
        nonlocal normalized
        normalized += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(summary_module, "_normalize_trace_entry", counting_normalizer)

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            traces = [[event("snapshot", marker=marker)]]
        else:
            traces = [
                [
                    *(
                        event("alloc", address=1000 + index, size=1)
                        for index in range(50)
                    ),
                    event("snapshot", marker=markers[0]),
                    event("alloc", address=9000, size=64, pool=(0, 5)),
                ]
            ]
        return snapshot(
            segment(active=64, pool=(0, 5), address=9000),
            traces=traces,
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.mark("before")
    recorder.mark("after")
    comparison = recorder.finish().compare(
        "before",
        "after",
        attribution=AttributionOptions(events=True, on_missing="error"),
    )

    assert len(comparison.allocator_events) == 1
    assert normalized == 1


def test_timeline_reports_absolute_state_and_zero_deltas() -> None:
    run = make_run(
        [
            snapshot(segment(active=10, total=20)),
            snapshot(segment(active=10, total=20)),
            snapshot(),
        ],
        labels=("start", "steady", "released"),
    )

    timeline = run.timeline()
    steady = [item for item in timeline.pools if item.point_label == "steady"][0]
    released = [item for item in timeline.pools if item.point_label == "released"][0]
    assert steady.stats.reserved_bytes == 20
    assert steady.delta.reserved_bytes == 0
    assert released.stats.reserved_bytes == 0
    assert released.delta.reserved_bytes == -20
    assert "steady" in timeline.to_text(include_unchanged=True)
    compact = timeline.to_text(include_unchanged=False)
    assert "steady" in compact
    assert "allocated=10 B (delta +0 B)" not in compact


def test_phase_comparison_preserves_four_point_identity() -> None:
    baseline = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=30))],
        name="baseline",
        labels=("start", "end"),
    )
    candidate = make_run(
        [snapshot(segment(active=20)), snapshot(segment(active=35))],
        name="candidate",
        labels=("start", "end"),
    )

    phase = compare_phases(
        baseline.between("start", "end"),
        candidate.between("start", "end"),
        attribution=AttributionOptions(stacks=True),
    )
    row = next(item for item in phase.decomposition if item["metric"] == "active_bytes")
    assert row["start_delta_bytes"] == 10
    assert row["baseline_growth_bytes"] == 20
    assert row["candidate_growth_bytes"] == 15
    assert row["end_delta_bytes"] == 5
    assert row["identity_holds"] is True


def test_phase_total_identity_includes_unmatched_private_pools() -> None:
    baseline = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=30))],
        name="baseline",
        labels=("start", "end"),
    )
    candidate = make_run(
        [
            snapshot(
                segment(active=20),
                segment(active=10, pool=(0, 7), address=2000),
            ),
            snapshot(
                segment(active=35),
                segment(active=50, pool=(0, 7), address=2000),
            ),
        ],
        name="candidate",
        labels=("start", "end"),
    )

    phase = compare_phases(
        baseline.between("start", "end"),
        candidate.between("start", "end"),
    )
    row = next(
        item
        for item in phase.total_decomposition
        if item["scope"] == "all" and item["metric"] == "active_bytes"
    )

    assert row == {
        "scope": "all",
        "metric": "active_bytes",
        "start_delta_bytes": 20,
        "baseline_growth_bytes": 20,
        "candidate_growth_bytes": 55,
        "growth_delta_bytes": 35,
        "end_delta_bytes": 55,
        "identity_holds": True,
    }


def test_event_history_uses_trace_for_segment_device() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        traces = (
            [[], []]
            if len(markers) == 1
            else [
                [
                    event("snapshot", marker=markers[0]),
                    event("alloc", address=9000, size=64),
                ],
                [event("snapshot", marker=markers[0])],
            ]
        )
        active_segment = segment(active=64, pool=(0, 5), address=9000)
        active_segment["device"] = 0
        other_segment = segment(active=64, pool=(0, 6), address=9000)
        other_segment["device"] = 1
        return snapshot(other_segment, active_segment, traces=traces)

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.mark("before")
    recorder.mark("after")
    comparison = recorder.finish().compare(
        "before",
        "after",
        attribution=AttributionOptions(events=True, on_missing="error"),
    )

    assert comparison.events_available is True
    assert comparison.events_complete is True
    assert len(comparison.allocator_events) == 1
    assert comparison.allocator_events[0].pool_id == (0, 5)
    assert comparison.allocator_events[0].size_bytes == 64


def test_event_history_detects_overwritten_marker_on_segment_device() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        traces = (
            [[], []]
            if len(markers) == 1
            else [
                [event("alloc", address=9000, size=64, pool=(0, 5))],
                [event("snapshot", marker=markers[0])],
            ]
        )
        active_segment = segment(active=64, pool=(0, 5), address=9000)
        active_segment["device"] = 0
        return snapshot(active_segment, traces=traces)

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.mark("before")
    recorder.mark("after")
    comparison = recorder.finish().compare(
        "before",
        "after",
        attribution=AttributionOptions(events=True),
    )

    assert comparison.events_available is True
    assert comparison.events_complete is False
    assert len(comparison.allocator_events) == 1
    assert any("device 0" in warning for warning in comparison.warnings)


def test_manifest_only_comparisons_do_not_load_raw_snapshots(monkeypatch) -> None:
    left = make_run([snapshot(segment(active=10))], name="left")
    right = make_run([snapshot(segment(active=20))], name="right")
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        labels=("before", "after"),
    )
    original = MemoryPoint.raw_snapshot
    calls: list[tuple[str, int]] = []

    def tracked(point: MemoryPoint):
        calls.append((point.run_id, point.index))
        return original(point)

    monkeypatch.setattr(MemoryPoint, "raw_snapshot", tracked)
    compare_points(left.points[0], right.points[0])
    run.timeline()
    assert calls == []


def test_attributed_timeline_and_phase_load_each_point_once(monkeypatch) -> None:
    timeline_run = make_run(
        [
            snapshot(segment(active=10)),
            snapshot(segment(active=20)),
            snapshot(segment(active=30)),
        ],
        labels=("a", "b", "c"),
    )
    baseline = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        name="baseline",
        labels=("start", "end"),
    )
    candidate = make_run(
        [snapshot(segment(active=12)), snapshot(segment(active=25))],
        name="candidate",
        labels=("start", "end"),
    )
    original = MemoryPoint.raw_snapshot
    calls: list[tuple[str, int]] = []

    def tracked(point: MemoryPoint):
        calls.append((point.run_id, point.index))
        return original(point)

    monkeypatch.setattr(MemoryPoint, "raw_snapshot", tracked)
    timeline_run.timeline(attribution=AttributionOptions(stacks=True))
    assert calls == [(timeline_run.run_id, index) for index in range(3)]

    calls.clear()
    compare_phases(
        baseline.between("start", "end"),
        candidate.between("start", "end"),
        attribution=AttributionOptions(stacks=True),
    )
    assert calls == [
        (baseline.run_id, 0),
        (baseline.run_id, 1),
        (candidate.run_id, 0),
        (candidate.run_id, 1),
    ]
