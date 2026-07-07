from __future__ import annotations

import json
from pathlib import Path

import pytest

from torch_cudagraph_debug.memory_debug import (
    MemoryAllocationLifetimeAnalysis,
    MemoryAttributionOptions,
    MemoryDebugError,
    MemoryDisplayOptions,
    MemoryHistoryError,
    MemoryLifetimeOptions,
    MemoryLifetimeSelection,
    MemoryOwnershipError,
    MemoryPoint,
    MemoryRecorder,
    compare_points,
)
from torch_cudagraph_debug.memory_debug.advanced import (
    ALLOCATION_LIFETIME_ACTIONS,
    KNOWN_TRACE_ACTIONS,
)

from ._helpers import event, make_run, segment, snapshot


def _set_device(value: dict[str, object], device: int) -> dict[str, object]:
    value["device"] = device
    return value


def test_snapshot_lifetimes_group_sizes_and_infer_free_completion() -> None:
    run = make_run(
        [
            snapshot(
                segment(active=64, address=1000, frame="grads.py"),
                segment(active=128, address=2000, frame="grads.py"),
            ),
            snapshot(
                segment(active=64, address=1000, frame="grads.py"),
                segment(active=128, address=2000, frame="grads.py"),
            ),
            snapshot(),
        ],
        labels=("anchor", "steady", "final"),
    )

    report = run.lifetimes(
        MemoryLifetimeSelection.active_at("anchor"),
        through="final",
        options=MemoryLifetimeOptions(
            events=False, display=MemoryDisplayOptions(stack_depth=4)
        ),
    )

    assert isinstance(report, MemoryAllocationLifetimeAnalysis)
    assert report.source_kind == "run"
    assert report.source_id == run.run_id
    assert report.history_requested is False
    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert json.loads(cohort.to_row()["stack_frames_json"])[0]["filename"] == (
        "grads.py"
    )
    assert [item.active_bytes for item in cohort.points] == [192, 192, 0]
    assert [item.size_bytes for item in cohort.size_histogram] == [128, 64]
    assert cohort.event_exact_free_requested_bytes == 0
    assert cohort.snapshot_inferred_free_completed_bytes == 192
    assert cohort.snapshot_inferred_free_completed_count == 2
    assert cohort.owner_active_at_end_bytes == 0
    assert cohort.free_requests[0].confidence == "snapshot_inferred"


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
            events=True,
            on_missing="error",
            display=MemoryDisplayOptions(stack_depth=4),
        ),
    )

    assert report.history_available is True
    assert report.history_complete is True
    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert [item.active_bytes for item in cohort.points] == [64, 0]
    assert cohort.event_exact_free_requested_bytes == 64
    assert cohort.snapshot_inferred_free_completed_bytes == 0
    assert cohort.free_requests[0].confidence == "event_exact"
    assert "request.py" in cohort.free_requests[0].stack_key
    assert "new_alloc.py" not in cohort.stack_key

    born = run.lifetimes(
        MemoryLifetimeSelection.born_between("anchor", "reused"),
        options=MemoryLifetimeOptions(
            events=True,
            on_missing="error",
            display=MemoryDisplayOptions(stack_depth=4),
        ),
    )
    assert len(born.cohorts) == 1
    assert [item.active_bytes for item in born.cohorts[0].points] == [0, 64]
    assert born.cohorts[0].event_exact_birth_bytes == 64
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
            events=True,
            on_missing="error",
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
    assert cohort.event_exact_birth_bytes == 64
    assert cohort.event_exact_free_requested_bytes == 64
    assert "born.py" in cohort.stack_key
    assert "requested.py" in cohort.free_requests[0].stack_key

    paths = report.write(tmp_path / "born")
    assert "birth_stacks" in paths
    assert "free_request_stacks" in paths


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
        on_missing="error",
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
        assert cohort.event_exact_birth_bytes == 64
        assert cohort.event_exact_free_requested_bytes == 64
        assert cohort.event_exact_free_completed_bytes == 64


def test_born_between_snapshot_fallback_warns_and_misses_no_survivor() -> None:
    run = make_run(
        [snapshot(), snapshot(segment(active=64, frame="born.py")), snapshot()],
        labels=("before", "born", "completed"),
    )

    report = run.lifetimes(
        MemoryLifetimeSelection.born_between("before", "born"),
        through="completed",
        options=MemoryLifetimeOptions(
            events=False, display=MemoryDisplayOptions(stack_depth=4)
        ),
    )

    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert cohort.snapshot_inferred_birth_bytes == 64
    assert cohort.snapshot_inferred_free_completed_bytes == 64
    assert [item.active_bytes for item in cohort.points] == [0, 64, 0]
    assert any(
        "transient allocations may be missing" in item for item in report.warnings
    )


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
        options=MemoryLifetimeOptions(events=True, on_missing="error"),
    )

    assert len(report.cohorts) == 1
    assert report.cohorts[0].device == 1
    assert report.cohorts[0].born_bytes == 32
    assert report.cohorts[0].event_exact_free_completed_bytes == 32


def test_incomplete_history_keeps_provable_free_request_and_warns() -> None:
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
    report = recorder.finish().lifetimes(
        MemoryLifetimeSelection.active_at("anchor"),
        options=MemoryLifetimeOptions(events=True),
    )

    assert report.history_complete is False
    assert report.cohorts[0].event_exact_free_requested_bytes == 64
    assert any("snapshot-inferred" in warning for warning in report.warnings)

    with pytest.raises(MemoryHistoryError, match="unavailable or incomplete"):
        recorder.result.lifetimes(
            MemoryLifetimeSelection.active_at("anchor"),
            options=MemoryLifetimeOptions(events=True, on_missing="error"),
        )


def test_free_completion_without_request_infers_missing_boundary() -> None:
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
    report = recorder.finish().lifetimes(
        MemoryLifetimeSelection.active_at("owned"),
        options=MemoryLifetimeOptions(events=True, on_missing="error"),
    )

    cohort = report.cohorts[0]
    assert cohort.snapshot_inferred_free_requested_bytes == 64
    assert cohort.event_exact_free_completed_bytes == 64
    assert any("inferred from snapshot" in warning for warning in report.warnings)


def test_replay_warnings_are_grouped_by_reason_and_device() -> None:
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
    report = recorder.finish().lifetimes(
        options=MemoryLifetimeOptions(events=True, on_missing="error")
    )

    unmatched = [
        warning
        for warning in report.warnings
        if "could not be matched to an active allocation" in warning
    ]
    assert len(unmatched) == 3
    device_0 = next(
        warning for warning in unmatched if "12 free_completed events" in warning
    )
    assert "device 0" in device_0
    assert device_0.count("0x") == 3
    device_1 = next(
        warning for warning in unmatched if "5 free_completed events" in warning
    )
    assert "device 1" in device_1
    assert device_1.count("0x") == 3
    size_mismatch = next(
        warning for warning in unmatched if "event sizes differed" in warning
    )
    assert "2 free_completed events" in size_mismatch
    assert size_mismatch.count("0x") == 1

    address_reuse = next(
        warning for warning in report.warnings if "reused an active address" in warning
    )
    assert "2 allocation events" in address_reuse
    assert address_reuse.count("0x") == 1


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
        options=MemoryLifetimeOptions(events=True, on_missing="error"),
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
    report = recorder.finish().lifetimes(
        MemoryLifetimeSelection.active_at("anchor"),
        options=MemoryLifetimeOptions(events=True, on_missing="error"),
    )

    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert cohort.device == 0
    assert [item.active_bytes for item in cohort.points] == [64, 64]
    assert cohort.event_exact_free_requested_bytes == 0
    assert cohort.owner_active_at_end_bytes == 64


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
        options=MemoryLifetimeOptions(events=True, on_missing="error"),
    )

    assert len(report.cohorts) == 1
    assert report.cohorts[0].pool_id == (0, 7)
    assert report.cohorts[0].event_exact_free_completed_bytes == 64


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
        options=MemoryLifetimeOptions(events=True, on_missing="error"),
    )

    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert [point.owner_active_bytes for point in cohort.points] == [64, 0, 0]
    assert [point.awaiting_free_bytes for point in cohort.points] == [0, 64, 0]
    assert cohort.event_exact_free_requested_bytes == 64
    assert cohort.event_exact_free_completed_bytes == 64
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
    run = make_run(
        [snapshot(first, second), snapshot(first, second)],
        labels=("start", "end"),
    )
    report = run.lifetimes(
        MemoryLifetimeSelection.active_at("start"),
        options=MemoryLifetimeOptions(
            events=False, display=MemoryDisplayOptions(stack_depth=1, limit=1)
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
            events=False,
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

    reversed_run = make_run(
        [snapshot(second, first), snapshot(second, first)],
        labels=("start", "end"),
    )
    reversed_report = reversed_run.lifetimes(
        MemoryLifetimeSelection.active_at("start"),
        options=MemoryLifetimeOptions(
            events=False, display=MemoryDisplayOptions(stack_depth=1, limit=1)
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
    ).lifetimes(options=MemoryLifetimeOptions(events=False))

    assert len(report.cohorts) == 2
    assert {cohort.size_histogram[0].size_bytes for cohort in report.cohorts} == {
        64,
        128,
    }
    assert all(cohort.stack_fingerprint == "unattributed" for cohort in report.cohorts)


def test_size_outcomes_separate_persistent_and_completed_instances() -> None:
    report = make_run(
        [
            snapshot(
                segment(active=64, address=1000, frame="same.py"),
                segment(active=64, address=2000, frame="same.py"),
            ),
            snapshot(segment(active=64, address=1000, frame="same.py")),
        ],
        labels=("start", "end"),
    ).lifetimes(options=MemoryLifetimeOptions(events=False))

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
        actions = sorted(KNOWN_TRACE_ACTIONS - {"snapshot"}) + ["future_action"]
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
        attribution=MemoryAttributionOptions(events=True, on_missing="error"),
    )
    assert {item.action for item in comparison.allocator_events} == (
        KNOWN_TRACE_ACTIONS - {"snapshot"}
    ) | {"future_action"}

    report = run.lifetimes(options=MemoryLifetimeOptions(events=True))
    assert any("future_action" in warning for warning in report.warnings)


def test_lifetime_report_owns_all_formats(tmp_path: Path) -> None:
    run = make_run(
        [
            snapshot(),
            snapshot(segment(active=64, frame="grads.py")),
            snapshot(),
        ],
        labels=("before", "anchor", "final"),
    )
    report = run.lifetimes(
        MemoryLifetimeSelection.active_at("anchor"),
        options=MemoryLifetimeOptions(events=False),
    )

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
    assert "snapshot_inferred" in (output / "free_request_stacks.csv").read_text(
        encoding="utf-8"
    )
    assert "snapshot_inferred" in (output / "free_completion_stacks.csv").read_text(
        encoding="utf-8"
    )
    assert "free_completed" in (output / "size_outcomes.csv").read_text(
        encoding="utf-8"
    )
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "<svg" in html
    assert 'points="80.0,' in html


def test_compare_and_timeline_can_embed_lifetime_summary(tmp_path: Path) -> None:
    run = make_run(
        [snapshot(segment(active=64, frame="grads.py")), snapshot()],
        labels=("before", "after"),
    )
    options = MemoryAttributionOptions(lifetimes=True, events=False)

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

    warning_timeline = run.timeline(
        attribution=MemoryAttributionOptions(lifetimes=True, events=True)
    )
    assert "allocator event history" in warning_timeline.to_text()
    assert "allocator event history" in warning_timeline.to_html()


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


def test_lifetime_scan_loads_each_point_once(monkeypatch) -> None:
    run = make_run(
        [
            snapshot(segment(active=10)),
            snapshot(segment(active=20)),
            snapshot(segment(active=30)),
        ],
        labels=("a", "b", "c"),
    )
    original = MemoryPoint.raw_snapshot
    calls: list[int] = []

    def tracked(point: MemoryPoint):
        calls.append(point.index)
        return original(point)

    monkeypatch.setattr(MemoryPoint, "raw_snapshot", tracked)
    run.lifetimes(options=MemoryLifetimeOptions(events=False))
    assert calls == [0, 1, 2]


def test_combined_timeline_attribution_loads_each_point_once(monkeypatch) -> None:
    run = make_run(
        [
            snapshot(segment(active=10)),
            snapshot(segment(active=20)),
            snapshot(segment(active=30)),
        ],
        labels=("before", "middle", "after"),
    )
    original = MemoryPoint.raw_snapshot
    calls: list[int] = []

    def tracked(point: MemoryPoint):
        calls.append(point.index)
        return original(point)

    monkeypatch.setattr(MemoryPoint, "raw_snapshot", tracked)
    run.timeline(
        attribution=MemoryAttributionOptions(
            stacks=True,
            events=True,
            lifetimes=True,
        )
    )
    assert calls == [0, 1, 2]
