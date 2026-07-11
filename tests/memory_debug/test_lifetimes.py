from __future__ import annotations

import json
from pathlib import Path

import pytest

from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryDebugError,
    MemoryDisplayOptions,
    MemoryHistoryDisabledError,
    MemoryHistoryTruncatedError,
    MemoryLifetimeOptions,
    MemoryLifetimeSelection,
    MemoryOwnershipError,
    MemoryPoint,
    MemoryReconciliationError,
    MemoryRecorder,
    MemoryRun,
    compare_points,
)
from torch_cudagraph_debug.memory_debug.advanced import (
    ALLOCATION_LIFETIME_ACTIONS,
    KNOWN_TRACE_ACTIONS,
)

from ._helpers import event, make_history_run, make_run, segment, snapshot


def _set_device(value: dict[str, object], device: int) -> dict[str, object]:
    value["device"] = device
    return value


def test_idle_selected_device_does_not_block_event_analysis() -> None:
    """A selected device with no allocations must not fail strict evidence."""

    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(
                segment(active=0, total=64),
                traces=[[event("snapshot", marker=marker)], []],
            )
        return snapshot(
            segment(active=64, total=64),
            traces=[
                [
                    event("snapshot", marker=markers[0]),
                    event("alloc", address=1000, size=64),
                    event("snapshot", marker=marker),
                ],
                [],
            ],
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider, devices=(0, 1))
    recorder.record_point("before")
    recorder.record_point("after")
    run = recorder.finish()

    report = run.lifetimes()
    assert report.history_complete is True
    assert report.cohorts

    comparison = run.compare(
        "before", "after", attribution=MemoryAttributionOptions(events=True)
    )
    assert comparison.attribution_status.events.complete is True


def test_lifetimes_require_event_history() -> None:
    run = make_run(
        [snapshot(segment(active=64, address=1000)), snapshot()],
        labels=("anchor", "final"),
    )

    with pytest.raises(MemoryHistoryDisabledError, match="unavailable for interval"):
        run.lifetimes(MemoryLifetimeSelection.active_at("anchor"))


def test_event_lifetimes_split_same_address_reuse_generation() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(
                segment(active=64, requested=60, address=1000, frame="old_alloc.py"),
                traces=[[event("snapshot", marker=marker)]],
            )
        return snapshot(
            segment(active=64, address=1000, frame="new_alloc.py"),
            traces=[
                [
                    event("snapshot", marker=markers[0]),
                    event(
                        "free_requested",
                        address=1000,
                        size=60,
                        frame="request.py",
                        name="request_grad_free",
                        time_us=2,
                    ),
                    event(
                        "free_completed",
                        address=1000,
                        size=64,
                        frame="completed.py",
                        name="complete_free",
                        time_us=3,
                    ),
                    event(
                        "alloc",
                        address=1000,
                        size=64,
                        frame="new_alloc.py",
                        time_us=4,
                    ),
                ]
            ],
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("anchor")
    recorder.record_point("reused")
    run = recorder.finish()

    report = run.lifetimes(
        MemoryLifetimeSelection.active_at("anchor"),
        through="reused",
        options=MemoryLifetimeOptions(
            display=MemoryDisplayOptions(stack_depth=4),
        ),
    )

    assert report.history_available is True
    assert report.history_complete is True
    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert [item.active_bytes for item in cohort.points] == [64, 0]
    assert cohort.free_requested_bytes == 64
    assert cohort.free_completed_bytes == 64
    assert cohort.free_requests[0].origin == "event"
    assert "request.py" in cohort.free_requests[0].stack_key
    assert "new_alloc.py" not in cohort.stack_key

    born = run.lifetimes(
        MemoryLifetimeSelection.born_between("anchor", "reused"),
        options=MemoryLifetimeOptions(
            display=MemoryDisplayOptions(stack_depth=4),
        ),
    )
    assert len(born.cohorts) == 1
    assert [item.active_bytes for item in born.cohorts[0].points] == [0, 64]
    assert born.cohorts[0].born_bytes == 64
    assert "new_alloc.py" in born.cohorts[0].stack_key


def test_born_between_keeps_event_only_transient_allocation(tmp_path: Path) -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(traces=[[event("snapshot", marker=marker)]])
        return snapshot(
            traces=[
                [
                    event("snapshot", marker=markers[0]),
                    event(
                        "alloc",
                        address=9000,
                        size=64,
                        pool=(0, 7),
                        frame="born.py",
                        time_us=2,
                    ),
                    event(
                        "free_requested",
                        address=9000,
                        size=64,
                        pool=(0, 7),
                        frame="requested.py",
                        time_us=3,
                    ),
                    event(
                        "free_completed",
                        address=9000,
                        size=64,
                        pool=(0, 7),
                        frame="completed.py",
                        time_us=4,
                    ),
                ]
            ]
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("before")
    recorder.record_point("after")
    report = recorder.finish().lifetimes(
        MemoryLifetimeSelection.born_between("before", "after"),
        options=MemoryLifetimeOptions(
            display=MemoryDisplayOptions(stack_depth=4),
        ),
    )

    assert report.history_complete is True
    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert cohort.pool_id == (0, 7)
    assert [item.active_bytes for item in cohort.points] == [0, 0]
    assert cohort.peak_active_bytes == 0
    assert cohort.event_unreusable_peak_bytes == 64
    assert cohort.born_bytes == 64
    assert cohort.free_requested_bytes == 64
    assert "born.py" in cohort.stack_key
    assert "requested.py" in cohort.free_requests[0].stack_key

    text = report.to_text()
    assert "snapshot_peak=0 B" in text
    assert "owner_event_peak=64 B" in text
    assert "unreusable_event_peak=64 B" in text
    assert "snapshot_blocks=0" in text

    paths = report.write(tmp_path / "born")
    assert "birth_stacks" in paths
    assert "free_request_stacks" in paths


def test_event_owner_peak_ignores_pre_range_awaiting_free_blocks() -> None:
    alloc_frames = [{"filename": "alloc.py", "line": 7, "name": "allocate"}]

    def awaiting_block_segment() -> dict[str, object]:
        value = segment(active=100, address=1000, frame=None)
        value["blocks"][0]["state"] = "active_awaiting_free"
        value["blocks"][0]["frames"] = [dict(item) for item in alloc_frames]
        return value

    born = segment(active=100, address=2000, frame=None)
    born["blocks"][0]["frames"] = [dict(item) for item in alloc_frames]

    run = make_history_run(
        [
            ([awaiting_block_segment()], []),
            (
                [awaiting_block_segment(), born],
                [event("alloc", address=2000, size=100, time_us=2)],
            ),
        ],
        labels=("start", "end"),
    )
    report = run.lifetimes(MemoryLifetimeSelection.all())

    assert report.history_complete is True
    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    # The pre-range awaiting-free block never held owner-active bytes inside
    # the range, so only the in-range birth defines the owner peak.
    assert cohort.event_owner_peak_bytes == 100
    assert cohort.event_unreusable_peak_bytes == 200


def test_embedded_lifetimes_keep_event_only_transient_allocation() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(traces=[[event("snapshot", marker=marker)]])
        return snapshot(
            traces=[
                [
                    event("snapshot", marker=markers[0]),
                    event("alloc", address=9000, size=64, pool=(0, 7)),
                    event(
                        "free_requested",
                        address=9000,
                        size=64,
                        pool=(0, 7),
                        frame="requested.py",
                    ),
                    event(
                        "free_completed",
                        address=9000,
                        size=64,
                        pool=(0, 7),
                        frame="completed.py",
                    ),
                ]
            ]
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("before")
    recorder.record_point("after")
    run = recorder.finish()
    options = MemoryAttributionOptions(
        events=True,
        lifetimes=True,
    )

    comparison = run.compare("before", "after", attribution=options)
    timeline = run.timeline(attribution=options)

    assert comparison.allocation_lifetimes is not None
    assert timeline.allocation_lifetimes is not None
    for report in (
        comparison.allocation_lifetimes,
        timeline.allocation_lifetimes,
    ):
        assert report.history_complete is True
        assert len(report.cohorts) == 1
        cohort = report.cohorts[0]
        assert [item.active_bytes for item in cohort.points] == [0, 0]
        assert cohort.born_bytes == 64
        assert cohort.free_requested_bytes == 64
        assert cohort.free_completed_bytes == 64


def test_born_between_uses_trace_devices_without_endpoint_segments() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        resident = segment(active=16, address=1000)
        resident["device"] = 0
        if len(markers) == 1:
            traces = [
                [event("snapshot", marker=marker)],
                [event("snapshot", marker=marker)],
            ]
        else:
            traces = [
                [event("snapshot", marker=markers[0])],
                [
                    event("snapshot", marker=markers[0]),
                    event("alloc", address=9000, size=32, pool=(0, 8)),
                    event("free_requested", address=9000, size=32, pool=(0, 8)),
                    event("free_completed", address=9000, size=32, pool=(0, 8)),
                ],
            ]
        return snapshot(resident, traces=traces)

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("before")
    recorder.record_point("after")
    report = recorder.finish().lifetimes(
        MemoryLifetimeSelection.born_between("before", "after"),
        options=MemoryLifetimeOptions(),
    )

    assert len(report.cohorts) == 1
    assert report.cohorts[0].device == 1
    assert report.cohorts[0].born_bytes == 32
    assert report.cohorts[0].free_completed_bytes == 32


def test_truncated_history_raises_typed_error() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(segment(active=64, address=1000))
        return snapshot(
            traces=[
                [
                    event(
                        "free_requested",
                        address=1000,
                        size=64,
                        frame="request.py",
                    )
                ]
            ]
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("anchor")
    recorder.record_point("final")
    run = recorder.finish()

    with pytest.raises(
        MemoryHistoryTruncatedError, match="truncated for interval"
    ) as excinfo:
        run.lifetimes(MemoryLifetimeSelection.active_at("anchor"))
    assert "anchor -> final" in str(excinfo.value)


def test_lifetime_boundary_order_raises_reconciliation_error() -> None:
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
        return snapshot(segment(active=64, address=1000), traces=traces)

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("anchor")
    recorder.record_point("final")

    with pytest.raises(MemoryReconciliationError, match="boundary order"):
        recorder.finish().lifetimes(MemoryLifetimeSelection.active_at("anchor"))


def test_free_completion_without_request_is_a_contradiction() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(
                segment(active=64, address=1000, frame="allocation.py"),
                traces=[[event("snapshot", marker=marker)]],
            )
        return snapshot(
            traces=[
                [
                    event("snapshot", marker=markers[0]),
                    event(
                        "free_completed",
                        address=1000,
                        size=64,
                        frame="completion.py",
                    ),
                ]
            ]
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("owned")
    recorder.record_point("completed")
    run = recorder.finish()

    with pytest.raises(
        MemoryReconciliationError, match="without any free_requested"
    ) as excinfo:
        run.lifetimes(MemoryLifetimeSelection.active_at("owned"))
    assert "0x3e8" in str(excinfo.value)
    assert "please report it" in str(excinfo.value)


def test_contradictions_are_grouped_by_reason_and_device() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(
                _set_device(
                    segment(
                        active=64,
                        address=5000,
                        frame="tracked_size.py",
                    ),
                    0,
                ),
                _set_device(
                    segment(
                        active=64,
                        address=6000,
                        frame="tracked_reuse.py",
                    ),
                    0,
                ),
                traces=[
                    [event("snapshot", marker=marker)],
                    [event("snapshot", marker=marker)],
                ],
            )
        return snapshot(
            traces=[
                [event("snapshot", marker=markers[0])]
                + [
                    event("free_completed", address=1000 + index, size=64)
                    for index in range(12)
                ]
                + [event("free_completed", address=5000, size=32) for _ in range(2)]
                + [
                    event(
                        "alloc",
                        address=6000,
                        size=64,
                        frame=f"reuse_{index}.py",
                    )
                    for index in range(2)
                ],
                [event("snapshot", marker=markers[0])]
                + [
                    event("free_completed", address=2000 + index, size=64)
                    for index in range(5)
                ],
            ]
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("before")
    recorder.record_point("after")
    run = recorder.finish()

    with pytest.raises(MemoryReconciliationError) as excinfo:
        run.lifetimes()
    message = str(excinfo.value)

    device_0 = next(
        line for line in message.splitlines() if "12 free_completed events" in line
    )
    assert "device 0" in device_0
    assert device_0.count("0x") == 3
    device_1 = next(
        line for line in message.splitlines() if "5 free_completed events" in line
    )
    assert "device 1" in device_1
    assert "incompatible with the tracked allocation" in message
    assert "reused an address" in message


def test_lifetimes_keep_devices_separate_for_same_address() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        before = len(markers) == 1
        segments = []
        for device, frame in ((0, "device0.py"), (1, "device1.py")):
            item = segment(
                active=64 if before else 0,
                total=64,
                address=1000,
                frame=frame,
            )
            segments.append(_set_device(item, device))
        if before:
            traces = [
                [event("snapshot", marker=marker)],
                [event("snapshot", marker=marker)],
            ]
        else:
            traces = [
                [
                    event("snapshot", marker=markers[0]),
                    event(
                        "free_requested",
                        address=1000,
                        size=64,
                        frame="free0.py",
                    ),
                    event(
                        "free_completed",
                        address=1000,
                        size=64,
                        frame="done0.py",
                    ),
                ],
                [
                    event("snapshot", marker=markers[0]),
                    event(
                        "free_requested",
                        address=1000,
                        size=64,
                        frame="free1.py",
                    ),
                    event(
                        "free_completed",
                        address=1000,
                        size=64,
                        frame="done1.py",
                    ),
                ],
            ]
        return snapshot(*segments, traces=traces)

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("anchor")
    recorder.record_point("final")
    report = recorder.finish().lifetimes(
        MemoryLifetimeSelection.active_at("anchor"),
        options=MemoryLifetimeOptions(),
    )

    assert {item.device for item in report.cohorts} == {0, 1}
    request_stacks = {
        item.device: item.free_requests[0].stack_key for item in report.cohorts
    }
    completion_stacks = {
        item.device: item.free_completions[0].stack_key for item in report.cohorts
    }
    assert "free0.py" in request_stacks[0]
    assert "free1.py" in request_stacks[1]
    assert "done0.py" in completion_stacks[0]
    assert "done1.py" in completion_stacks[1]


def test_lifetimes_do_not_match_event_to_other_device_same_address() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        resident = _set_device(
            segment(active=64, address=1000, frame="device0.py"),
            0,
        )
        if len(markers) == 1:
            traces = [
                [event("snapshot", marker=marker)],
                [event("snapshot", marker=marker)],
            ]
        else:
            traces = [
                [event("snapshot", marker=markers[0])],
                [
                    event("snapshot", marker=markers[0]),
                    event(
                        "free_requested",
                        address=1000,
                        size=64,
                        frame="unrelated_device1.py",
                    ),
                ],
            ]
        return snapshot(resident, traces=traces)

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("anchor")
    recorder.record_point("final")
    run = recorder.finish()

    with pytest.raises(MemoryReconciliationError) as excinfo:
        run.lifetimes(MemoryLifetimeSelection.active_at("anchor"))
    message = str(excinfo.value)
    assert "free_requested" in message
    assert "device 1" in message
    assert "device 0" not in message


def test_event_only_birth_infers_pool_from_segment_range() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        reserved = segment(
            active=0,
            total=256,
            address=9000,
            pool=(0, 7),
        )
        if len(markers) == 1:
            traces = [[event("snapshot", marker=marker)]]
        else:
            traces = [
                [
                    event("snapshot", marker=markers[0]),
                    event("alloc", address=9000, size=64, frame="born.py"),
                    event(
                        "free_requested",
                        address=9000,
                        size=64,
                        frame="requested.py",
                    ),
                    event(
                        "free_completed",
                        address=9000,
                        size=64,
                        frame="completed.py",
                    ),
                ]
            ]
        return snapshot(reserved, traces=traces)

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("before")
    recorder.record_point("after")
    report = recorder.finish().lifetimes(
        MemoryLifetimeSelection.born_between("before", "after"),
        options=MemoryLifetimeOptions(),
    )

    assert len(report.cohorts) == 1
    assert report.cohorts[0].pool_id == (0, 7)
    assert report.cohorts[0].free_completed_bytes == 64


def test_free_request_and_completion_are_distinct_lifetime_transitions() -> None:
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(
                segment(active=64, address=1000, frame="lifetime.py"),
                traces=[[event("snapshot", marker=marker)]],
            )
        if len(markers) == 2:
            awaiting = segment(active=64, address=1000, frame="lifetime.py")
            awaiting["blocks"][0]["state"] = "active_awaiting_free"
            return snapshot(
                awaiting,
                traces=[
                    [
                        event("snapshot", marker=markers[0]),
                        event(
                            "free_requested",
                            address=1000,
                            size=64,
                            frame="request.py",
                        ),
                    ]
                ],
            )
        return snapshot(
            traces=[
                [
                    event("snapshot", marker=markers[1]),
                    event(
                        "free_completed",
                        address=1000,
                        size=64,
                        frame="completion.py",
                    ),
                ]
            ]
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("owned")
    recorder.record_point("awaiting")
    recorder.record_point("reusable")
    report = recorder.finish().lifetimes(
        MemoryLifetimeSelection.active_at("owned"),
        options=MemoryLifetimeOptions(),
    )

    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert [point.owner_active_bytes for point in cohort.points] == [64, 0, 0]
    assert [point.awaiting_free_bytes for point in cohort.points] == [0, 64, 0]
    assert cohort.free_requested_bytes == 64
    assert cohort.free_completed_bytes == 64
    assert cohort.awaiting_free_at_end_bytes == 0
    assert "request.py" in cohort.free_requests[0].stack_key
    assert "completion.py" in cohort.free_completions[0].stack_key
    assert cohort.size_outcomes[0].terminal_state == "free_completed"


def test_full_stack_identity_and_display_limit_do_not_drop_cohorts() -> None:
    def stacked(address: int, leaf: str) -> dict[str, object]:
        value = segment(active=64, address=address, frame="shared.py")
        value["blocks"][0]["frames"] = [
            {"filename": "shared.py", "line": 10, "name": "forward"},
            {"filename": leaf, "line": 20, "name": "allocate"},
        ]
        return value

    first = stacked(1000, "left.py")
    second = stacked(2000, "right.py")
    run = make_history_run(
        [([first, second], []), ([first, second], [])],
        labels=("start", "end"),
    )
    report = run.lifetimes(
        MemoryLifetimeSelection.active_at("start"),
        options=MemoryLifetimeOptions(
            display=MemoryDisplayOptions(stack_depth=1, limit=1)
        ),
    )

    assert len(report.cohorts) == 2
    assert len(report.cohort_rows()) == 2
    assert len(report.to_dict()["cohorts"]) == 2
    assert len({cohort.stack_fingerprint for cohort in report.cohorts}) == 2
    assert {cohort.display_stack(1) for cohort in report.cohorts} == {
        "shared.py:10:forward"
    }
    assert "showing 1 of 2" in report.to_text()
    comparison = run.compare(
        "start",
        "end",
        attribution=MemoryAttributionOptions(
            lifetimes=True,
            events=True,
            display=MemoryDisplayOptions(stack_depth=1, limit=1),
        ),
    )
    assert comparison.allocation_lifetimes is not None
    first_id, second_id = (
        cohort.cohort_id for cohort in comparison.allocation_lifetimes.cohorts
    )
    assert first_id in comparison.to_html()
    assert second_id not in comparison.to_html()
    assert "Showing 1 of 2 cohorts." in comparison.to_html()
    assert len(comparison.to_dict()["allocation_lifetimes"]["cohorts"]) == 2

    reversed_run = make_history_run(
        [([second, first], []), ([second, first], [])],
        labels=("start", "end"),
    )
    reversed_report = reversed_run.lifetimes(
        MemoryLifetimeSelection.active_at("start"),
        options=MemoryLifetimeOptions(
            display=MemoryDisplayOptions(stack_depth=1, limit=1)
        ),
    )
    assert {cohort.cohort_id for cohort in report.cohorts} == {
        cohort.cohort_id for cohort in reversed_report.cohorts
    }


def test_unattributed_allocations_use_size_in_cohort_identity() -> None:
    report = make_run(
        [
            snapshot(
                segment(active=64, address=1000, frame=None),
                segment(active=128, address=2000, frame=None),
            )
        ],
        labels=("only",),
    ).lifetimes()

    assert len(report.cohorts) == 2
    assert {cohort.size_histogram[0].size_bytes for cohort in report.cohorts} == {
        64,
        128,
    }
    assert all(cohort.stack_fingerprint == "unattributed" for cohort in report.cohorts)


def test_size_outcomes_separate_persistent_and_completed_instances() -> None:
    report = make_history_run(
        [
            (
                [
                    segment(active=64, address=1000, frame="same.py"),
                    segment(active=64, address=2000, frame="same.py"),
                ],
                [],
            ),
            (
                [segment(active=64, address=1000, frame="same.py")],
                [
                    event("free_requested", address=2000, size=64, frame="same.py"),
                    event("free_completed", address=2000, size=64, frame="same.py"),
                ],
            ),
        ],
        labels=("start", "end"),
    ).lifetimes()

    assert len(report.cohorts) == 1
    outcomes = {
        item.terminal_state: (item.count, item.total_bytes)
        for item in report.cohorts[0].size_outcomes
    }
    assert outcomes == {
        "free_completed": (1, 64),
        "owner_active": (1, 64),
    }


def test_allocator_action_taxonomy_is_complete_and_unknown_actions_warn() -> None:
    assert KNOWN_TRACE_ACTIONS == {
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
    assert ALLOCATION_LIFETIME_ACTIONS == {
        "alloc",
        "free_requested",
        "free_completed",
    }
    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(traces=[[event("snapshot", marker=marker)]])
        actions = sorted(
            KNOWN_TRACE_ACTIONS - ALLOCATION_LIFETIME_ACTIONS - {"snapshot"}
        ) + ["future_action"]
        return snapshot(
            traces=[
                [event("snapshot", marker=markers[0])]
                + [
                    event(action, address=1000 + index * 128, size=64)
                    for index, action in enumerate(actions)
                ]
            ]
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
    assert {item.action for item in comparison.allocator_events} == (
        KNOWN_TRACE_ACTIONS - ALLOCATION_LIFETIME_ACTIONS - {"snapshot"}
    ) | {"future_action"}

    lifetime_run = make_history_run(
        [
            ([], []),
            (
                [],
                [
                    event("segment_alloc", address=1000, size=64),
                    event("alloc", address=1000, size=64, frame="born.py"),
                    event("future_action", address=1000, size=64),
                    event("free_requested", address=1000, size=64),
                    event("free_completed", address=1000, size=64),
                    event("segment_free", address=1000, size=64),
                ],
            ),
        ],
        labels=("before", "after"),
    )
    report = lifetime_run.lifetimes()
    assert any("future_action" in warning for warning in report.warnings)


def test_lifetime_report_owns_all_formats(tmp_path: Path) -> None:
    run = make_history_run(
        [
            ([], []),
            (
                [segment(active=64, frame="grads.py")],
                [event("alloc", address=1000, size=64, frame="grads.py")],
            ),
            (
                [],
                [
                    event("free_requested", address=1000, size=64, frame="grads.py"),
                    event("free_completed", address=1000, size=64, frame="grads.py"),
                ],
            ),
        ],
        labels=("before", "anchor", "final"),
    )
    report = run.lifetimes(MemoryLifetimeSelection.active_at("anchor"))

    output = tmp_path / "lifetimes"
    paths = report.write(output)
    assert set(paths) == {
        "text",
        "json",
        "html",
        "cohorts",
        "cohort_points",
        "size_histograms",
        "size_outcomes",
        "free_request_stacks",
        "free_completion_stacks",
    }
    payload = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "allocation-lifetime-analysis"
    assert payload["cohorts"][0]["peak_active_bytes"] == 64
    assert "event" in (output / "free_request_stacks.csv").read_text(encoding="utf-8")
    assert "free_completed" in (output / "size_outcomes.csv").read_text(
        encoding="utf-8"
    )
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "<svg" in html
    assert 'points="80.0,' in html


def test_compare_and_timeline_can_embed_lifetime_summary(tmp_path: Path) -> None:
    run = make_history_run(
        [
            ([segment(active=64, frame="grads.py")], []),
            (
                [],
                [
                    event("free_requested", address=1000, size=64, frame="grads.py"),
                    event("free_completed", address=1000, size=64, frame="grads.py"),
                ],
            ),
        ],
        labels=("before", "after"),
    )
    options = MemoryAttributionOptions(lifetimes=True)

    comparison = run.compare("before", "after", attribution=options)
    assert comparison.allocation_lifetimes is not None
    assert "allocation cohorts" in comparison.to_text()
    comparison_paths = comparison.write(tmp_path / "comparison")
    assert "cohorts" in comparison_paths

    timeline = run.timeline(attribution=options)
    assert timeline.allocation_lifetimes is not None
    assert all(item.allocation_lifetimes is None for item in timeline.point_comparisons)
    assert "allocation cohorts" in timeline.to_text()
    timeline_paths = timeline.write(tmp_path / "timeline")
    assert "cohort_points" in timeline_paths

    assert comparison.allocator_events == ()
    assert comparison.attribution_status.events.requested is False
    assert comparison.attribution_status.lifetimes.complete is True


def test_lifetime_point_validation_and_empty_run() -> None:
    run = make_run(
        [snapshot(segment(active=64)), snapshot()],
        labels=("before", "after"),
    )
    other = make_run([snapshot(segment(active=1))], labels=("point",))

    with pytest.raises(ValueError, match="must not come before"):
        run.lifetimes(
            MemoryLifetimeSelection.active_at("after"),
            through="before",
        )
    with pytest.raises(MemoryOwnershipError):
        run.lifetimes(MemoryLifetimeSelection.active_at(other["point"]))
    with pytest.raises(ValueError, match="requires exactly one point"):
        MemoryLifetimeSelection("active_at", start="before", end="after")
    with pytest.raises(ValueError, match="born_between end"):
        run.lifetimes(
            MemoryLifetimeSelection.born_between("before", "after"),
            through="before",
        )

    with pytest.raises(MemoryDebugError, match="cannot be compared across"):
        compare_points(
            run["before"],
            other["point"],
            attribution=MemoryAttributionOptions(lifetimes=True),
        )
    empty = MemoryRecorder().finish()
    with pytest.raises(MemoryDebugError, match="at least one point"):
        empty.lifetimes()


def test_lifetime_scan_loads_each_allocator_state_once(monkeypatch) -> None:
    run = make_history_run(
        [
            ([segment(active=10)], []),
            ([segment(active=10)], []),
            ([segment(active=10)], []),
        ],
        labels=("a", "b", "c"),
    )
    original = MemoryPoint.allocator_state
    calls: list[int] = []

    def tracked(point: MemoryPoint):
        calls.append(point.index)
        return original(point)

    monkeypatch.setattr(MemoryPoint, "allocator_state", tracked)
    run.lifetimes()
    assert calls == [0, 1, 2]


def test_combined_timeline_attribution_loads_each_allocator_state_once(
    monkeypatch,
) -> None:
    run = make_history_run(
        [
            ([segment(active=10)], []),
            ([segment(active=10)], []),
            ([segment(active=10)], []),
        ],
        labels=("before", "middle", "after"),
    )
    original = MemoryPoint.allocator_state
    calls: list[int] = []

    def tracked(point: MemoryPoint):
        calls.append(point.index)
        return original(point)

    monkeypatch.setattr(MemoryPoint, "allocator_state", tracked)
    run.timeline(
        attribution=MemoryAttributionOptions(
            stacks=True,
            events=True,
            lifetimes=True,
        )
    )
    assert calls == [0, 1, 2]


def test_lifetimes_reconcile_event_births_with_rounded_blocks() -> None:
    """A surviving request must match its allocator-rounded block, not double."""

    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(traces=[[event("snapshot", marker=marker)]])
        return snapshot(
            segment(active=512, address=4096, requested=400, frame="alloc.py"),
            traces=[
                [
                    event("snapshot", marker=markers[0]),
                    event("alloc", address=4096, size=400, frame="alloc.py"),
                    event("snapshot", marker=marker),
                ]
            ],
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("before")
    recorder.record_point("after")
    report = recorder.finish().lifetimes()

    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert cohort.born_count == 1
    assert cohort.born_bytes == 512
    assert cohort.free_requests == ()
    assert cohort.free_completions == ()
    assert cohort.owner_active_at_end_count == 1
    assert cohort.owner_active_at_end_bytes == 512


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"active": 1024}, "changed allocator-rounded size"),
        ({"requested": 300}, "changed requested size"),
        ({"pool": (2, 2)}, "changed allocator pool"),
        ({"stream": 8}, "changed allocation stream"),
        ({"frame": "other.py"}, "changed allocation stack"),
    ],
)
def test_snapshot_confirmed_instance_rejects_identity_drift(
    change: dict[str, object], message: str
) -> None:
    start_args: dict[str, object] = {
        "active": 512,
        "requested": 400,
        "pool": (1, 1),
        "stream": 7,
        "frame": "first.py",
    }
    end_args = {**start_args, **change}
    run = make_history_run(
        [
            ([segment(**start_args)], []),
            ([segment(**end_args)], []),
        ],
        labels=("start", "end"),
    )

    with pytest.raises(MemoryReconciliationError, match=message):
        run.lifetimes()


def test_event_born_instance_confirms_snapshot_identity_once() -> None:
    run = make_history_run(
        [
            ([], []),
            (
                [
                    segment(
                        active=512,
                        requested=400,
                        pool=(1, 1),
                        stream=7,
                        frame="block.py",
                    )
                ],
                [
                    event(
                        "alloc",
                        size=400,
                        pool=(1, 1),
                        stream=7,
                        frame="event.py",
                    )
                ],
            ),
            (
                [
                    segment(
                        active=512,
                        requested=400,
                        pool=(1, 1),
                        stream=7,
                        frame="block.py",
                    )
                ],
                [],
            ),
        ],
        labels=("start", "confirmed", "end"),
    )

    report = run.lifetimes()

    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert cohort.pool_id == (1, 1)
    assert cohort.streams == (7,)
    assert "block.py" in cohort.stack_key
    assert "event.py" in cohort.births[0].stack_key
    assert cohort.born_bytes == 512
    assert cohort.size_histogram[0].size_bytes == 512
    assert cohort.size_histogram[0].requested_bytes == 400


@pytest.mark.parametrize(
    ("event_change", "block_change", "message"),
    [
        ({}, {"requested": 300}, "changed requested size"),
        ({"pool": (1, 1)}, {"pool": (2, 2)}, "changed allocator pool"),
        ({"stream": 7}, {"stream": 8}, "changed allocation stream"),
    ],
)
def test_event_born_instance_rejects_first_snapshot_conflict(
    event_change: dict[str, object],
    block_change: dict[str, object],
    message: str,
) -> None:
    event_args: dict[str, object] = {
        "size": 400,
        "pool": (1, 1),
        "stream": 7,
        "frame": "alloc.py",
    }
    block_args: dict[str, object] = {
        "active": 512,
        "requested": 400,
        "pool": (1, 1),
        "stream": 7,
        "frame": "alloc.py",
    }
    run = make_history_run(
        [
            ([], []),
            (
                [segment(**{**block_args, **block_change})],
                [event("alloc", **{**event_args, **event_change})],
            ),
        ],
        labels=("start", "end"),
    )

    with pytest.raises(MemoryReconciliationError, match=message):
        run.lifetimes()


def test_missing_allocation_stack_can_be_enriched_by_later_snapshot() -> None:
    run = make_history_run(
        [
            ([segment(active=512, requested=400, frame=None)], []),
            ([segment(active=512, requested=400, frame="alloc.py")], []),
        ],
        labels=("start", "end"),
    )

    report = run.lifetimes()

    assert len(report.cohorts) == 1
    assert "alloc.py" in report.cohorts[0].stack_key


def test_duplicate_free_requested_is_a_reconciliation_error() -> None:
    start = segment(active=512, requested=400, stream=7)
    end = segment(active=512, requested=400, stream=7)
    end["blocks"][0]["state"] = "active_awaiting_free"
    run = make_history_run(
        [
            ([start], []),
            (
                [end],
                [
                    event("free_requested", size=400, stream=7),
                    event("free_requested", size=400, stream=7),
                ],
            ),
        ],
        labels=("start", "end"),
    )

    with pytest.raises(MemoryReconciliationError, match="duplicate free_requested"):
        run.lifetimes()


def test_duplicate_active_address_in_snapshot_is_a_reconciliation_error() -> None:
    run = make_history_run(
        [
            (
                [
                    segment(active=512, address=1000),
                    segment(active=512, address=1000),
                ],
                [],
            ),
            ([segment(active=512, address=1000)], []),
        ],
        labels=("start", "end"),
    )

    with pytest.raises(MemoryReconciliationError, match="duplicate active block"):
        run.lifetimes()


def test_transient_in_cross_era_range_anchors_to_its_birth_era() -> None:
    """A transient born after a witnessed era change takes the new era's pool."""

    run = make_history_run(
        [
            ([segment(active=100, address=1000, pool=(0, 2), frame="old.py")], []),
            (
                [segment(active=100, address=1000, frame="new.py")],
                [
                    event("free_requested", address=1000, size=100, frame="old.py"),
                    event("free_completed", address=1000, size=100, frame="old.py"),
                    event("segment_free", address=1000, size=100),
                    event("segment_alloc", address=1000, size=100),
                    event("alloc", address=1000, size=50, frame="transient.py"),
                    event(
                        "free_requested", address=1000, size=50, frame="transient.py"
                    ),
                    event(
                        "free_completed", address=1000, size=50, frame="transient.py"
                    ),
                    event("alloc", address=1000, size=100, frame="new.py"),
                ],
            ),
        ],
        labels=("before", "after"),
    )
    report = run.lifetimes()

    # The old era ended before the transient's birth (covering segment churn),
    # so only the end snapshot can testify: the transient anchors to the new
    # era's pool instead of degrading to unknown.
    transient = next(
        cohort for cohort in report.cohorts if "transient.py" in cohort.stack_key
    )
    assert transient.pool_id == (0, 0)
    survivor = next(cohort for cohort in report.cohorts if "new.py" in cohort.stack_key)
    assert survivor.pool_id == (0, 0)


def test_transient_in_middle_era_reports_unknown_pool() -> None:
    """A transient whose era touches neither endpoint must not inherit a pool."""

    run = make_history_run(
        [
            ([segment(active=100, address=1000, pool=(0, 2), frame="old.py")], []),
            (
                [segment(active=100, address=1000, frame="new.py")],
                [
                    event("free_requested", address=1000, size=100, frame="old.py"),
                    event("free_completed", address=1000, size=100, frame="old.py"),
                    event("segment_free", address=1000, size=100),
                    event("segment_alloc", address=1000, size=100),
                    event("alloc", address=1000, size=50, frame="transient.py"),
                    event(
                        "free_requested", address=1000, size=50, frame="transient.py"
                    ),
                    event(
                        "free_completed", address=1000, size=50, frame="transient.py"
                    ),
                    event("segment_free", address=1000, size=100),
                    event("segment_alloc", address=1000, size=100),
                    event("alloc", address=1000, size=100, frame="new.py"),
                ],
            ),
        ],
        labels=("before", "after"),
    )
    report = run.lifetimes()

    transient = next(
        cohort for cohort in report.cohorts if "transient.py" in cohort.stack_key
    )
    assert transient.pool_id == ("unknown",)


def test_recycled_range_with_agreeing_endpoints_reports_unknown_pool() -> None:
    """Pool A -> pool B -> pool A recycling must not silently attribute A."""

    run = make_history_run(
        [
            ([segment(active=100, address=1000, frame="era_a.py")], []),
            (
                [segment(active=100, address=1000, frame="era_a2.py")],
                [
                    event("free_requested", address=1000, size=100, frame="era_a.py"),
                    event("free_completed", address=1000, size=100, frame="era_a.py"),
                    event("segment_free", address=1000, size=100),
                    event("segment_alloc", address=1000, size=100),
                    event("alloc", address=1040, size=16, frame="transient_b.py"),
                    event(
                        "free_requested", address=1040, size=16, frame="transient_b.py"
                    ),
                    event(
                        "free_completed", address=1040, size=16, frame="transient_b.py"
                    ),
                    event("segment_free", address=1000, size=100),
                    event("segment_alloc", address=1000, size=100),
                    event("alloc", address=1000, size=100, frame="era_a2.py"),
                ],
            ),
        ],
        labels=("start", "end"),
    )
    report = run.lifetimes()

    transient = next(
        cohort for cohort in report.cohorts if "transient_b.py" in cohort.stack_key
    )
    assert transient.pool_id == ("unknown",)


def test_pool_stamped_traces_bypass_anchoring_entirely(tmp_path: Path) -> None:
    """PyTorch 2.12+ stamps every trace entry with its pool (pytorch#177717).

    With reported pools, even the recycled A -> B -> A shape that anchoring
    must degrade to unknown attributes exactly, and the verdict survives a
    bundle round-trip.
    """

    bundle = tmp_path / "stamped.tcgd-memory"
    run = make_history_run(
        [
            ([segment(active=100, address=1000, frame="era_a.py")], []),
            (
                [segment(active=100, address=1000, frame="era_a2.py")],
                [
                    event(
                        "free_requested",
                        address=1000,
                        size=100,
                        pool=(0, 0),
                        frame="era_a.py",
                    ),
                    event(
                        "free_completed",
                        address=1000,
                        size=100,
                        pool=(0, 0),
                        frame="era_a.py",
                    ),
                    event("segment_free", address=1000, size=100, pool=(0, 0)),
                    event("segment_alloc", address=1000, size=100, pool=(0, 9)),
                    event(
                        "alloc",
                        address=1040,
                        size=16,
                        pool=(0, 9),
                        frame="transient_b.py",
                    ),
                    event(
                        "free_requested",
                        address=1040,
                        size=16,
                        pool=(0, 9),
                        frame="transient_b.py",
                    ),
                    event(
                        "free_completed",
                        address=1040,
                        size=16,
                        pool=(0, 9),
                        frame="transient_b.py",
                    ),
                    event("segment_free", address=1000, size=100, pool=(0, 9)),
                    event("segment_alloc", address=1000, size=100, pool=(0, 0)),
                    event(
                        "alloc",
                        address=1000,
                        size=100,
                        pool=(0, 0),
                        frame="era_a2.py",
                    ),
                ],
            ),
        ],
        labels=("start", "end"),
        bundle_dir=bundle,
    )

    for report in (run.lifetimes(), MemoryRun.load(bundle).lifetimes()):
        transient = next(
            cohort for cohort in report.cohorts if "transient_b.py" in cohort.stack_key
        )
        assert transient.pool_id == (0, 9)


def test_transient_with_quiet_agreeing_endpoints_takes_their_pool() -> None:
    """No covering segment churn: both endpoints testify the same pool."""

    run = make_history_run(
        [
            ([segment(active=50, total=100, address=1000, pool=(0, 5))], []),
            (
                [segment(active=50, total=100, address=1000, pool=(0, 5))],
                [
                    event("alloc", address=1060, size=16, frame="transient.py"),
                    event(
                        "free_requested", address=1060, size=16, frame="transient.py"
                    ),
                    event(
                        "free_completed", address=1060, size=16, frame="transient.py"
                    ),
                ],
            ),
        ],
        labels=("start", "end"),
    )
    report = run.lifetimes()

    transient = next(
        cohort for cohort in report.cohorts if "transient.py" in cohort.stack_key
    )
    assert transient.pool_id == (0, 5)


def test_transient_pool_index_ignores_unrelated_segment_churn() -> None:
    target = segment(
        active=0,
        total=100,
        address=1000,
        pool=(0, 5),
        frame=None,
    )
    unrelated = segment(
        active=0,
        total=100,
        address=5000,
        pool=(0, 7),
        frame=None,
    )
    run = make_history_run(
        [
            ([unrelated, target], []),
            (
                [unrelated, target],
                [
                    event("segment_free", address=5000, size=100),
                    event("segment_alloc", address=5000, size=100),
                    event("alloc", address=1040, size=16, frame="transient.py"),
                    event(
                        "free_requested",
                        address=1040,
                        size=16,
                        frame="transient.py",
                    ),
                    event(
                        "free_completed",
                        address=1040,
                        size=16,
                        frame="transient.py",
                    ),
                ],
            ),
        ],
        labels=("start", "end"),
    )

    report = run.lifetimes()

    transient = next(
        cohort for cohort in report.cohorts if "transient.py" in cohort.stack_key
    )
    assert transient.pool_id == (0, 5)


def test_quiet_endpoints_with_conflicting_pools_are_a_contradiction() -> None:
    """Pool disagreement without covering churn is self-contradictory evidence."""

    persistent = segment(active=50, total=100, address=1000)
    coverage_start = segment(active=0, total=100, address=5000, pool=(0, 3), frame=None)
    coverage_end = segment(active=0, total=100, address=5000, pool=(0, 7), frame=None)
    run = make_history_run(
        [
            ([persistent, coverage_start], []),
            (
                [persistent, coverage_end],
                [
                    event("alloc", address=5040, size=16, frame="transient.py"),
                    event(
                        "free_requested", address=5040, size=16, frame="transient.py"
                    ),
                    event(
                        "free_completed", address=5040, size=16, frame="transient.py"
                    ),
                ],
            ),
        ],
        labels=("start", "end"),
    )

    with pytest.raises(MemoryReconciliationError, match="conflicting pools"):
        run.lifetimes()


def test_unmapped_transient_address_without_explanation_is_a_contradiction() -> None:
    """A birth address invisible at both endpoints needs segment evidence."""

    run = make_history_run(
        [
            ([segment(active=64, address=1000)], []),
            (
                [segment(active=64, address=1000)],
                [
                    event("alloc", address=999000, size=32),
                    event("free_requested", address=999000, size=32),
                    event("free_completed", address=999000, size=32),
                ],
            ),
        ],
        labels=("start", "end"),
    )

    with pytest.raises(MemoryReconciliationError, match="no covering segment"):
        run.lifetimes()


def test_expandable_growth_transient_anchors_to_end_pool() -> None:
    """A page mapped in before birth anchors through the end snapshot."""

    run = make_history_run(
        [
            (
                [
                    segment(
                        active=0, total=100, address=1000, frame=None, expandable=True
                    )
                ],
                [],
            ),
            (
                [
                    segment(
                        active=0, total=100, address=1000, frame=None, expandable=True
                    ),
                    segment(
                        active=0, total=100, address=5000, frame=None, expandable=True
                    ),
                ],
                [
                    event("segment_map", address=5000, size=100),
                    event("alloc", address=5040, size=16, frame="transient.py"),
                    event(
                        "free_requested", address=5040, size=16, frame="transient.py"
                    ),
                    event(
                        "free_completed", address=5040, size=16, frame="transient.py"
                    ),
                ],
            ),
        ],
        labels=("start", "end"),
    )
    report = run.lifetimes()

    transient = next(
        cohort for cohort in report.cohorts if "transient.py" in cohort.stack_key
    )
    assert transient.pool_id == (0, 0)


def test_expandable_partial_shrink_keeps_reservation_witness_alive() -> None:
    """Unmap/remap crossings are same-owner noise while any witness byte lives."""

    ranges = [
        segment(active=0, total=100, address=1000, frame=None, expandable=True),
        segment(active=0, total=100, address=5000, frame=None, expandable=True),
    ]
    run = make_history_run(
        [
            (list(ranges), []),
            (
                list(ranges),
                [
                    # The 5000 range shrinks away and regrows: the 1000 range
                    # keeps the reservation provably alive throughout.
                    event("segment_unmap", address=5000, size=100),
                    event("segment_map", address=5000, size=100),
                    event("alloc", address=5040, size=16, frame="transient.py"),
                    event(
                        "free_requested", address=5040, size=16, frame="transient.py"
                    ),
                    event(
                        "free_completed", address=5040, size=16, frame="transient.py"
                    ),
                ],
            ),
        ],
        labels=("start", "end"),
    )
    report = run.lifetimes()

    transient = next(
        cohort for cohort in report.cohorts if "transient.py" in cohort.stack_key
    )
    assert transient.pool_id == (0, 0)


def test_expandable_full_clear_before_birth_anchors_to_end_pool() -> None:
    """A reservation whose whole footprint cleared died; only the end testifies."""

    run = make_history_run(
        [
            (
                [
                    segment(
                        active=0, total=100, address=1000, frame=None, expandable=True
                    )
                ],
                [],
            ),
            (
                [
                    segment(
                        active=0,
                        total=100,
                        address=1000,
                        pool=(0, 7),
                        frame=None,
                        expandable=True,
                    )
                ],
                [
                    event("segment_unmap", address=1000, size=100),
                    event("segment_map", address=1000, size=100),
                    event("alloc", address=1040, size=16, frame="transient.py"),
                    event(
                        "free_requested", address=1040, size=16, frame="transient.py"
                    ),
                    event(
                        "free_completed", address=1040, size=16, frame="transient.py"
                    ),
                ],
            ),
        ],
        labels=("start", "end"),
    )
    report = run.lifetimes()

    transient = next(
        cohort for cohort in report.cohorts if "transient.py" in cohort.stack_key
    )
    assert transient.pool_id == (0, 7)


def test_reservation_death_only_invalidates_anchors_across_it() -> None:
    """A death moment splits the interval: earlier births still anchor start."""

    run = make_history_run(
        [
            (
                [
                    segment(
                        active=0, total=100, address=1000, frame=None, expandable=True
                    )
                ],
                [],
            ),
            (
                [
                    segment(
                        active=0,
                        total=100,
                        address=1000,
                        pool=(0, 7),
                        frame=None,
                        expandable=True,
                    )
                ],
                [
                    event("alloc", address=1040, size=16, frame="early.py"),
                    event("free_requested", address=1040, size=16, frame="early.py"),
                    event("free_completed", address=1040, size=16, frame="early.py"),
                    event("segment_unmap", address=1000, size=100),
                    event("segment_map", address=1000, size=100),
                    event("alloc", address=1040, size=16, frame="late.py"),
                    event("free_requested", address=1040, size=16, frame="late.py"),
                    event("free_completed", address=1040, size=16, frame="late.py"),
                ],
            ),
        ],
        labels=("start", "end"),
    )
    report = run.lifetimes()

    early = next(cohort for cohort in report.cohorts if "early.py" in cohort.stack_key)
    late = next(cohort for cohort in report.cohorts if "late.py" in cohort.stack_key)
    assert early.pool_id == (0, 0)
    assert late.pool_id == (0, 7)


def test_unknown_pool_cohort_renders_on_every_report_surface() -> None:
    # A transient whose segment was itself created and freed within the
    # interval has no pool coverage at either endpoint; the resulting
    # ("unknown",) sentinel must render instead of crashing the reports.
    run = make_history_run(
        [
            ([segment(active=64, address=1000)], []),
            (
                [segment(active=64, address=1000)],
                [
                    event("segment_alloc", address=998976, size=128),
                    event("alloc", address=999000, size=32),
                    event("free_requested", address=999000, size=32),
                    event("free_completed", address=999000, size=32),
                    event("segment_free", address=998976, size=128),
                ],
            ),
        ],
        labels=("start", "end"),
    )

    report = run.lifetimes()
    transient = next(
        cohort for cohort in report.cohorts if cohort.pool_id == ("unknown",)
    )
    assert transient.pool_label == "pool[unknown]"

    text = report.to_text()
    assert "pool[unknown]" in text
    assert "pool[unknown]" in report.to_html()
    assert any(row["pool_id"] == "pool[unknown]" for row in report.cohort_rows())
    comparison_text = run.compare(
        "start", "end", attribution=MemoryAttributionOptions(lifetimes=True)
    ).to_text()
    assert "pool[unknown]" in comparison_text


def test_transient_pool_attribution_batches_interval_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch_cudagraph_debug.memory_debug.lifetimes as lifetime_module

    query_batches: list[int] = []
    run_index_builds = 0
    original_evidence = lifetime_module._segment_action_evidence
    original_run_index = lifetime_module._build_run_range_index

    def tracked_evidence(history, queries):
        query_batches.append(len(queries))
        return original_evidence(history, queries)

    def tracked_run_index(segments):
        nonlocal run_index_builds
        run_index_builds += 1
        return original_run_index(segments)

    monkeypatch.setattr(lifetime_module, "_segment_action_evidence", tracked_evidence)
    monkeypatch.setattr(lifetime_module, "_build_run_range_index", tracked_run_index)

    transient_count = 128
    events = []
    for _ in range(transient_count):
        events.extend(
            [
                event("alloc", address=1024, size=16, frame="transient.py"),
                event(
                    "free_requested",
                    address=1024,
                    size=16,
                    frame="transient.py",
                ),
                event(
                    "free_completed",
                    address=1024,
                    size=16,
                    frame="transient.py",
                ),
            ]
        )
    coverage = segment(active=0, total=4096, address=1000, frame=None)
    run = make_history_run(
        [
            ([coverage], []),
            ([coverage], events),
        ],
        labels=("start", "end"),
    )

    report = run.lifetimes()

    assert sum(cohort.born_count for cohort in report.cohorts) == transient_count
    assert query_batches == [transient_count]
    assert run_index_builds == 2


def test_refined_instances_use_one_size_basis_across_all_rows() -> None:
    """Transition rows and cohort totals must agree on the rounded size."""

    markers: list[str] = []

    def provider(marker: str):
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(traces=[[event("snapshot", marker=marker)]])
        if len(markers) == 2:
            return snapshot(
                segment(active=512, address=4096, requested=400, frame="alloc.py"),
                traces=[
                    [
                        event("snapshot", marker=markers[0]),
                        event("alloc", address=4096, size=400, frame="alloc.py"),
                        event("snapshot", marker=marker),
                    ]
                ],
            )
        return snapshot(
            traces=[
                [
                    event("snapshot", marker=markers[0]),
                    event("alloc", address=4096, size=400, frame="alloc.py"),
                    event("snapshot", marker=markers[1]),
                    event(
                        "free_requested",
                        address=4096,
                        size=400,
                        frame="free.py",
                        time_us=2,
                    ),
                    event(
                        "free_completed",
                        address=4096,
                        size=400,
                        frame="free.py",
                        time_us=3,
                    ),
                    event("snapshot", marker=marker),
                ]
            ]
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("before")
    recorder.record_point("mid")
    recorder.record_point("after")
    report = recorder.finish().lifetimes()

    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert cohort.born_bytes == 512
    assert sum(item.size_bytes for item in cohort.births) == cohort.born_bytes
    assert sum(item.size_bytes for item in cohort.free_requests) == 512
    assert cohort.free_requested_bytes == 512
    assert sum(item.size_bytes for item in cohort.free_completions) == 512
    assert cohort.free_completed_bytes == 512
