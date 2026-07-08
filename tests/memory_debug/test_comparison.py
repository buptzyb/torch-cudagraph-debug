from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import torch_cudagraph_debug.memory_debug.allocator_snapshot as allocator_snapshot_module
import torch_cudagraph_debug.memory_debug.recording as recording_module
from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryDebugError,
    MemoryHistoryBoundaryError,
    MemoryHistoryDisabledError,
    MemoryHistoryTruncatedError,
    MemoryOwnershipError,
    MemoryPoint,
    MemoryPoolKey,
    MemoryReconciliationError,
    MemoryRecorder,
    MemoryRun,
    compare_phases,
    compare_points,
)

from ._helpers import event, make_history_run, make_run, segment, snapshot


@pytest.mark.parametrize("analysis", ["compare", "timeline"])
def test_events_and_lifetimes_share_loaded_event_evidence(
    analysis: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "shared-event-evidence.tcgd-memory"
    make_history_run(
        [
            ([segment(active=0, total=64)], []),
            (
                [segment(active=64, total=64)],
                [event("alloc", address=1000, size=64)],
            ),
        ],
        labels=("before", "after"),
        bundle_dir=bundle,
    )
    run = MemoryRun.load(bundle, cache_snapshots=False)
    original_read = recording_module._read_gzip_json
    read_names: list[str] = []

    def tracked_read(
        path: Path, *, context: str, expected_sha256: str | None = None
    ) -> object:
        read_names.append(path.name)
        return original_read(
            path,
            context=context,
            expected_sha256=expected_sha256,
        )

    monkeypatch.setattr(recording_module, "_read_gzip_json", tracked_read)
    options = MemoryAttributionOptions(events=True, lifetimes=True)

    if analysis == "compare":
        run.compare("before", "after", attribution=options)
    else:
        run.timeline(attribution=options)

    assert read_names.count("0000-0001.json.gz") == 1


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
    assert comparison.lifecycle_confidence == "exact"
    assert comparison.pool_comparisons[0].delta.active_bytes == 0
    by_stream = {
        item.reference_key.stream
        if item.reference_key is not None
        else None: item.delta.active_bytes
        for item in comparison.observation_comparisons
    }
    assert by_stream == {0: 10, 7: -10}
    assert comparison.pool_comparisons[0].reference.active_bytes == 30
    assert comparison.pool_comparisons[0].candidate.active_bytes == 30


def test_same_size_blocks_keep_distinct_address_lifecycle_identity() -> None:
    before = segment(active=20, total=30, address=1000)
    before["blocks"] = [
        {
            "address": 1000,
            "size": 10,
            "requested_size": 10,
            "state": "active_allocated",
            "frames": [],
        },
        {
            "address": 1010,
            "size": 10,
            "requested_size": 10,
            "state": "active_allocated",
            "frames": [],
        },
        {
            "address": 1020,
            "size": 10,
            "requested_size": 0,
            "state": "inactive",
            "frames": [],
        },
    ]
    after = segment(active=20, total=30, address=1000)
    after["blocks"] = [
        {
            "address": 1000,
            "size": 10,
            "requested_size": 0,
            "state": "inactive",
            "frames": [],
        },
        {
            "address": 1010,
            "size": 10,
            "requested_size": 10,
            "state": "active_allocated",
            "frames": [],
        },
        {
            "address": 1020,
            "size": 10,
            "requested_size": 10,
            "state": "active_allocated",
            "frames": [],
        },
    ]
    run = make_run(
        [snapshot(before), snapshot(after)],
        labels=("before", "after"),
    )

    comparison = run.compare("before", "after")
    lifecycle = comparison.pool_comparisons[0].lifecycle

    assert lifecycle is not None
    assert lifecycle.newly_active_bytes == 10
    assert lifecycle.became_inactive_bytes == 10


def test_coalesced_freed_blocks_count_as_became_inactive() -> None:
    before = segment(active=14, total=14, address=1000)
    before["blocks"] = [
        {
            "address": 1000,
            "size": 10,
            "requested_size": 10,
            "state": "active_allocated",
            "frames": [],
        },
        {
            "address": 1010,
            "size": 4,
            "requested_size": 4,
            "state": "active_allocated",
            "frames": [],
        },
    ]
    # Both blocks were freed and the allocator coalesced them into one
    # inactive block whose (address, size) key matches neither original.
    after = segment(active=0, total=14, address=1000)

    run = make_run(
        [snapshot(before), snapshot(after)],
        labels=("before", "after"),
    )
    lifecycle = run.compare("before", "after").pool_comparisons[0].lifecycle

    assert lifecycle is not None
    assert lifecycle.newly_active_bytes == 0
    assert lifecycle.became_inactive_bytes == 14


def test_address_less_blocks_count_with_multiplicity() -> None:
    def address_less(total: int, count: int, size: int) -> dict[str, object]:
        value = segment(active=total, total=total, address=1000)
        value["address"] = None
        value["blocks"] = [
            {
                "address": None,
                "size": size,
                "requested_size": size,
                "state": "active_allocated",
                "frames": [],
            }
            for _ in range(count)
        ]
        return value

    run = make_run(
        [
            snapshot(segment(active=0, total=200)),
            snapshot(address_less(total=200, count=2, size=100)),
        ],
        labels=("before", "after"),
    )
    comparison = run.compare("before", "after")
    lifecycle = comparison.pool_comparisons[0].lifecycle

    assert lifecycle is not None
    # Two distinct 100-byte blocks without addresses must count twice,
    # not collapse onto one identity key.
    assert lifecycle.newly_active_bytes == 200
    assert comparison.lifecycle_confidence == "approximate"
    assert any("addresses are missing" in warning for warning in comparison.warnings)


def test_cross_era_address_reuse_yields_ambiguous_event_attribution() -> None:
    run = make_history_run(
        [
            ([segment(active=100, address=1000, pool=(0, 2))], []),
            (
                [segment(active=300, address=1000)],
                [
                    event("free_requested", address=1000, size=100, time_us=2),
                    event("free_completed", address=1000, size=100, time_us=3),
                    event("alloc", address=1200, size=300, time_us=4),
                ],
            ),
        ],
        labels=("before", "after"),
    )
    comparison = run.compare(
        "before", "after", attribution=MemoryAttributionOptions(events=True)
    )

    by_action = {item.action: item for item in comparison.allocator_events}
    # Address 1000 belongs to different pools in the two snapshots; events
    # there cannot be attributed to either pool with confidence.
    assert by_action["free_requested"].pool_id is None
    assert by_action["free_requested"].attribution_confidence == "ambiguous"
    assert by_action["free_completed"].attribution_confidence == "ambiguous"
    # Address 1200 is covered only by the candidate default-pool segment.
    assert by_action["alloc"].pool_id == (0, 0)
    assert by_action["alloc"].attribution_confidence == "matched"


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

    assert [item.match for item in comparison.pool_comparisons].count("default") == 1
    private = [
        item
        for item in comparison.pool_comparisons
        if item.reference_key == MemoryPoolKey(0, (0, 1))
        or item.candidate_key == MemoryPoolKey(0, (0, 1))
    ]
    assert {item.match for item in private} == {"reference_only", "candidate_only"}
    assert all(
        item.match in {"reference_only", "candidate_only"}
        for item in comparison.observation_comparisons
    )
    assert comparison.lifecycle_available is False
    assert any("private pools are unmatched" in item for item in comparison.warnings)
    assert comparison.lifecycle_confidence == "unavailable"


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
    allocator_scope_comparisons = {
        item.scope: item for item in comparison.allocator_scope_comparisons
    }

    assert allocator_scope_comparisons["all"].reference.active_bytes == 30
    assert allocator_scope_comparisons["all"].candidate.active_bytes == 55
    assert allocator_scope_comparisons["all"].delta.active_bytes == 25
    assert allocator_scope_comparisons["default"].delta.active_bytes == 5
    assert allocator_scope_comparisons["private"].delta.active_bytes == 20


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
        pool_mapping={
            MemoryPoolKey(0, (0, 1)): MemoryPoolKey(0, (0, 8)),
            MemoryPoolKey(0, (0, 2)): MemoryPoolKey(0, (0, 9)),
        },
    )
    assert all(item.match == "mapped" for item in comparison.pool_comparisons)
    assert [item.delta.active_bytes for item in comparison.pool_comparisons] == [5, 5]

    with pytest.raises(ValueError, match="candidate.*more than once"):
        compare_points(
            before["point"],
            after["point"],
            pool_mapping={
                MemoryPoolKey(0, (0, 1)): MemoryPoolKey(0, (0, 8)),
                MemoryPoolKey(0, (0, 2)): MemoryPoolKey(0, (0, 8)),
            },
        )
    with pytest.raises(ValueError, match="does not exist"):
        compare_points(
            before["point"],
            after["point"],
            pool_mapping={MemoryPoolKey(0, (0, 7)): MemoryPoolKey(0, (0, 8))},
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
            attribution=MemoryAttributionOptions(events=True),
        )
    with pytest.raises(MemoryDebugError, match="lifetimes cannot"):
        compare_points(
            run["before"],
            other["point"],
            attribution=MemoryAttributionOptions(lifetimes=True),
        )


def test_missing_stack_history_raises_typed_error() -> None:
    run = make_run(
        [
            snapshot(segment(active=10, frame=None)),
            snapshot(segment(active=20, frame=None)),
        ],
        labels=("before", "after"),
    )

    with pytest.raises(
        MemoryHistoryDisabledError, match="stack history is unavailable"
    ):
        run.compare(
            "before",
            "after",
            attribution=MemoryAttributionOptions(stacks=True),
        )


def test_partial_stack_history_returns_exact_rows_and_coverage() -> None:
    reference = snapshot(
        segment(active=40, address=1000, frame="known.py"),
        segment(active=60, address=2000, frame=None),
    )
    candidate = snapshot(
        segment(active=50, address=1000, frame="known.py"),
        segment(active=70, address=2000, frame=None),
    )
    run = make_run([reference, candidate], labels=("before", "after"))

    comparison = run.compare(
        "before",
        "after",
        attribution=MemoryAttributionOptions(stacks=True),
    )

    assert comparison.reference_stack_coverage is not None
    assert comparison.candidate_stack_coverage is not None
    assert comparison.reference_stack_coverage.ratio == pytest.approx(0.4)
    assert comparison.reference_stack_coverage.unattributed_bytes == 60
    assert comparison.candidate_stack_coverage.unattributed_bytes == 70
    assert comparison.attribution_status.stacks.available is True
    assert comparison.attribution_status.stacks.complete is False
    assert any("coverage is incomplete" in warning for warning in comparison.warnings)
    unattributed = next(
        row for row in comparison.allocation_stack_comparisons if not row.stack_frames
    )
    assert unattributed.reference_size_bytes == 60
    assert unattributed.candidate_size_bytes == 70
    assert unattributed.delta_size_bytes == 10


def test_empty_stack_state_is_complete() -> None:
    run = make_run([snapshot(), snapshot()], labels=("before", "after"))

    comparison = run.compare(
        "before",
        "after",
        attribution=MemoryAttributionOptions(stacks=True),
    )

    assert comparison.reference_stack_coverage is not None
    assert comparison.reference_stack_coverage.ratio == 1.0
    assert comparison.attribution_status.stacks.available is True
    assert comparison.attribution_status.stacks.complete is True
    assert comparison.allocation_stack_comparisons == ()


def test_missing_device_trace_raises_typed_error() -> None:
    before = segment(active=10)
    after = segment(active=20)
    before["device"] = 0
    after["device"] = 0
    run = make_run(
        [snapshot(before), snapshot(after)],
        labels=("before", "after"),
    )

    with pytest.raises(
        MemoryHistoryDisabledError, match="event history is unavailable"
    ):
        run.compare(
            "before",
            "after",
            attribution=MemoryAttributionOptions(events=True),
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
    recorder.record_point("before")
    recorder.record_point("after")
    run = recorder.finish()
    comparison = run.compare(
        "before",
        "after",
        attribution=MemoryAttributionOptions(events=True),
    )

    assert comparison.attribution_status.events.available is True
    assert comparison.attribution_status.events.complete is True
    assert len(comparison.allocator_events) == 1
    assert comparison.allocator_events[0].pool_id == (0, 5)
    assert comparison.allocator_events[0].action == "alloc"


def test_event_history_normalizes_only_the_marker_window(monkeypatch) -> None:
    markers: list[str] = []
    normalized = 0
    original = allocator_snapshot_module._normalize_trace_entry

    def counting_normalizer(*args, **kwargs):
        nonlocal normalized
        normalized += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        allocator_snapshot_module, "_normalize_trace_entry", counting_normalizer
    )

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
    recorder.record_point("before")
    recorder.record_point("after")
    comparison = recorder.finish().compare(
        "before",
        "after",
        attribution=MemoryAttributionOptions(events=True),
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
    steady = [item for item in timeline.pool_entries if item.point_label == "steady"][0]
    released = [
        item for item in timeline.pool_entries if item.point_label == "released"
    ][0]
    assert steady.stats.reserved_bytes == 20
    assert steady.delta.reserved_bytes == 0
    assert released.stats.reserved_bytes == 0
    assert released.delta.reserved_bytes == -20
    assert "steady" in timeline.to_text(include_unchanged=True)
    compact = timeline.to_text(include_unchanged=False)
    assert "steady" not in compact
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
        attribution=MemoryAttributionOptions(stacks=True),
    )
    row = next(
        item
        for item in phase.pool_decomposition
        if item.components.metric == "active_bytes"
    )
    assert row.components.start_gap_bytes == 10
    assert row.components.baseline_change_bytes == 20
    assert row.components.candidate_change_bytes == 15
    assert row.components.end_gap_bytes == 5
    assert row.components.identity_holds is True


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
        for item in phase.allocator_scope_decomposition
        if item.scope == "all" and item.components.metric == "active_bytes"
    )
    assert row.to_dict() == {
        "scope": "all",
        "metric": "active_bytes",
        "start_gap_bytes": 20,
        "baseline_change_bytes": 20,
        "candidate_change_bytes": 55,
        "change_gap_bytes": 35,
        "end_gap_bytes": 55,
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
    recorder.record_point("before")
    recorder.record_point("after")
    comparison = recorder.finish().compare(
        "before",
        "after",
        attribution=MemoryAttributionOptions(events=True),
    )

    assert comparison.attribution_status.events.available is True
    assert comparison.attribution_status.events.complete is True
    assert len(comparison.allocator_events) == 1
    assert comparison.allocator_events[0].pool_id == (0, 5)
    assert comparison.allocator_events[0].size_bytes == 64


def test_event_only_range_loads_endpoint_states_and_all_event_chunks(
    monkeypatch,
) -> None:
    run = make_history_run(
        [
            ([segment(active=0, address=1000)], []),
            (
                [segment(active=64, address=1000)],
                [event("alloc", address=1000, size=64, pool=(0, 0))],
            ),
            (
                [segment(active=128, address=1000)],
                [event("alloc", address=2000, size=64, pool=(0, 0))],
            ),
        ],
        labels=("start", "middle", "end"),
    )
    original = MemoryPoint.allocator_state
    state_calls: list[int] = []

    def tracked(point: MemoryPoint):
        state_calls.append(point.index)
        return original(point)

    monkeypatch.setattr(MemoryPoint, "allocator_state", tracked)
    comparison = run.compare(
        "start",
        "end",
        attribution=MemoryAttributionOptions(events=True),
    )

    assert state_calls == [0, 2]
    assert sum(item.count for item in comparison.allocator_events) == 2


@pytest.mark.parametrize("missing_capture", [1, 2])
def test_unrecorded_event_endpoint_raises_typed_error(
    missing_capture: int,
) -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        trace = [
            event("snapshot", marker=item)
            for index, item in enumerate(markers, start=1)
            if index != missing_capture
        ]
        return snapshot(segment(active=64), traces=[trace])

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    capture = recorder._collector.capture
    capture_count = 0

    def without_endpoint_boundary(marker: str, **kwargs):
        nonlocal capture_count
        result = capture(marker, **kwargs)
        capture_count += 1
        return replace(result, boundary_recorded=capture_count != missing_capture)

    recorder._collector.capture = without_endpoint_boundary
    recorder.record_point("before")
    recorder.record_point("after")
    run = recorder.finish()

    with pytest.raises(MemoryHistoryBoundaryError, match="could not be recorded"):
        run.compare(
            "before",
            "after",
            attribution=MemoryAttributionOptions(events=True),
        )


def test_event_history_overwritten_marker_raises_typed_error() -> None:
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
    recorder.record_point("before")
    recorder.record_point("after")
    run = recorder.finish()

    assert run.compare("before", "after").candidate is run["after"]
    assert not any(
        "event boundary" in warning
        for point in run.points
        for warning in point.warnings
    )

    with pytest.raises(MemoryHistoryTruncatedError, match="truncated"):
        run.compare(
            "before",
            "after",
            attribution=MemoryAttributionOptions(events=True),
        )


def test_end_snapshot_without_own_marker_is_complete_evidence() -> None:
    """The ending snapshot's trace terminates at the boundary that produced it.

    On the real torch path the end marker enters history as the snapshot is
    taken and is never visible in the snapshot's own trace; the end boundary
    is attested by ``boundary_recorded``. A window that runs from the start
    marker to the end of the trace is therefore complete — classifying it
    truncated would fail every real recorder run.
    """

    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            traces = [[event("snapshot", marker=marker)]]
        else:
            traces = [
                [
                    event("snapshot", marker=markers[0]),
                    event("alloc", address=9000, size=64, pool=(0, 5)),
                ]
            ]
        active_segment = segment(active=64, pool=(0, 5), address=9000)
        return snapshot(active_segment, traces=traces)

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("before")
    recorder.record_point("after")
    run = recorder.finish()

    comparison = run.compare(
        "before",
        "after",
        attribution=MemoryAttributionOptions(events=True),
    )
    assert comparison.attribution_status.events.complete is True
    assert len(comparison.allocator_events) == 1
    assert comparison.allocator_events[0].action == "alloc"


def test_event_history_boundary_order_raises_reconciliation_error() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        traces = (
            [[event("snapshot", marker=marker)]]
            if len(markers) == 1
            else [
                [
                    event("snapshot", marker=marker),
                    event("snapshot", marker=markers[0]),
                ]
            ]
        )
        return snapshot(segment(active=64), traces=traces)

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("before")
    recorder.record_point("after")

    with pytest.raises(MemoryReconciliationError, match="boundary order"):
        recorder.finish().compare(
            "before",
            "after",
            attribution=MemoryAttributionOptions(events=True),
        )


def test_manifest_only_comparisons_do_not_load_allocator_states(monkeypatch) -> None:
    reference = make_run([snapshot(segment(active=10))], name="reference")
    candidate = make_run([snapshot(segment(active=20))], name="candidate")
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        labels=("before", "after"),
    )
    original = MemoryPoint.allocator_state
    calls: list[tuple[str, int]] = []

    def tracked(point: MemoryPoint):
        calls.append((point.run_id, point.index))
        return original(point)

    monkeypatch.setattr(MemoryPoint, "allocator_state", tracked)
    compare_points(reference.points[0], candidate.points[0])
    run.timeline()
    assert calls == []


def test_attributed_timeline_and_phase_load_each_allocator_state_once(
    monkeypatch,
) -> None:
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
    original = MemoryPoint.allocator_state
    calls: list[tuple[str, int]] = []

    def tracked(point: MemoryPoint):
        calls.append((point.run_id, point.index))
        return original(point)

    monkeypatch.setattr(MemoryPoint, "allocator_state", tracked)
    timeline_run.timeline(attribution=MemoryAttributionOptions(stacks=True))
    assert calls == [(timeline_run.run_id, index) for index in range(3)]

    calls.clear()
    compare_phases(
        baseline.between("start", "end"),
        candidate.between("start", "end"),
        attribution=MemoryAttributionOptions(stacks=True),
    )
    assert calls == [
        (baseline.run_id, 0),
        (baseline.run_id, 1),
        (candidate.run_id, 0),
        (candidate.run_id, 1),
    ]


def test_phase_comparison_rejects_ranges_from_the_same_run() -> None:
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        labels=("start", "end"),
    )

    with pytest.raises(ValueError, match="independent runs"):
        compare_phases(
            run.between("start", "end"),
            run.between("start", "end"),
        )


def test_phase_pool_mapping_accepts_pools_present_at_only_one_endpoint() -> None:
    baseline = make_run(
        [
            snapshot(segment(active=10)),
            snapshot(
                segment(active=10),
                segment(active=20, pool=(0, 1), address=2000),
            ),
        ],
        name="baseline",
        labels=("start", "end"),
    )
    candidate = make_run(
        [
            snapshot(segment(active=10)),
            snapshot(
                segment(active=10),
                segment(active=30, pool=(0, 8), address=3000),
            ),
        ],
        name="candidate",
        labels=("start", "end"),
    )

    phase = compare_phases(
        baseline.between("start", "end"),
        candidate.between("start", "end"),
        pool_mapping={
            MemoryPoolKey(0, (0, 1)): MemoryPoolKey(0, (0, 8)),
        },
    )

    active = next(
        item
        for item in phase.pool_decomposition
        if item.baseline_key == MemoryPoolKey(0, (0, 1))
        and item.components.metric == "active_bytes"
    )
    assert active.components.start_gap_bytes == 0
    assert active.components.baseline_change_bytes == 20
    assert active.components.candidate_change_bytes == 30
    assert active.components.end_gap_bytes == 10
    assert active.components.identity_holds


def test_lifetimes_attribution_requires_history() -> None:
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        labels=("before", "after"),
    )

    with pytest.raises(
        MemoryHistoryDisabledError, match="event history is unavailable"
    ):
        run.compare(
            "before",
            "after",
            attribution=MemoryAttributionOptions(lifetimes=True),
        )
