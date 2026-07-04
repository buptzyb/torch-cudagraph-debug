from __future__ import annotations

import json
from pathlib import Path

import pytest

from torch_cudagraph_debug.memory_debug import (
    MemoryAllocationLifetimeAnalysis,
    MemoryAttributionOptions,
    MemoryLifetimeOptions,
    MemoryDebugError,
    MemoryHistoryError,
    MemoryOwnershipError,
    MemoryPoint,
    MemoryRecorder,
    compare_points,
)

from ._helpers import event, make_run, segment, snapshot


def _set_device(value: dict[str, object], device: int) -> dict[str, object]:
    value["device"] = device
    return value


def test_snapshot_lifetimes_group_sizes_and_infer_release() -> None:
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
        "anchor",
        through="final",
        options=MemoryLifetimeOptions(events=False, stack_depth=4),
    )

    assert isinstance(report, MemoryAllocationLifetimeAnalysis)
    assert report.source_kind == "run"
    assert report.source_id == run.run_id
    assert report.history_requested is False
    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert [item.active_bytes for item in cohort.points] == [192, 192, 0]
    assert [item.size_bytes for item in cohort.size_histogram] == [128, 64]
    assert cohort.event_exact_release_bytes == 0
    assert cohort.snapshot_inferred_release_bytes == 192
    assert cohort.snapshot_inferred_release_count == 2
    assert cohort.still_active_bytes == 0
    assert cohort.releases[0].confidence == "snapshot_inferred"


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
                        frame="release.py",
                        name="release_grad",
                        time_us=2,
                    ),
                    event(
                        "alloc",
                        address=1000,
                        size=64,
                        frame="new_alloc.py",
                        time_us=3,
                    ),
                ]
            ],
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("anchor")
    recorder.record_point("reused")
    run = recorder.finish()

    report = run.lifetimes(
        "anchor",
        through="reused",
        options=MemoryLifetimeOptions(
            events=True,
            on_missing="error",
            stack_depth=4,
        ),
    )

    assert report.history_available is True
    assert report.history_complete is True
    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert [item.active_bytes for item in cohort.points] == [64, 0]
    assert cohort.event_exact_release_bytes == 64
    assert cohort.snapshot_inferred_release_bytes == 0
    assert cohort.releases[0].confidence == "event_exact"
    assert "release.py" in cohort.releases[0].stack_key
    assert "new_alloc.py" not in cohort.stack_key

    born = run.lifetimes(
        born_between=("anchor", "reused"),
        options=MemoryLifetimeOptions(
            events=True,
            on_missing="error",
            stack_depth=4,
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
                        frame="released.py",
                        time_us=3,
                    ),
                ]
            ]
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("before")
    recorder.record_point("after")
    report = recorder.finish().lifetimes(
        born_between=("before", "after"),
        options=MemoryLifetimeOptions(
            events=True,
            on_missing="error",
            stack_depth=4,
        ),
    )

    assert report.history_complete is True
    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert cohort.pool_id == (0, 7)
    assert [item.active_bytes for item in cohort.points] == [0, 0]
    assert cohort.peak_active_bytes == 0
    assert cohort.peak_live_bytes == 64
    assert cohort.event_exact_birth_bytes == 64
    assert cohort.event_exact_release_bytes == 64
    assert "born.py" in cohort.stack_key
    assert "released.py" in cohort.releases[0].stack_key

    paths = report.write(tmp_path / "born")
    assert "birth_stacks" in paths
    assert "release_stacks" in paths


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
                        frame="released.py",
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
        assert cohort.event_exact_release_bytes == 64


def test_born_between_snapshot_fallback_warns_and_misses_no_survivor() -> None:
    run = make_run(
        [snapshot(), snapshot(segment(active=64, frame="born.py")), snapshot()],
        labels=("before", "born", "released"),
    )

    report = run.lifetimes(
        born_between=("before", "born"),
        through="released",
        options=MemoryLifetimeOptions(events=False, stack_depth=4),
    )

    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert cohort.snapshot_inferred_birth_bytes == 64
    assert cohort.snapshot_inferred_release_bytes == 64
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
                ],
            ]
        return snapshot(resident, traces=traces)

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("before")
    recorder.record_point("after")
    report = recorder.finish().lifetimes(
        born_between=("before", "after"),
        options=MemoryLifetimeOptions(events=True, on_missing="error"),
    )

    assert len(report.cohorts) == 1
    assert report.cohorts[0].device == 1
    assert report.cohorts[0].born_bytes == 32


def test_incomplete_history_keeps_provable_release_and_warns() -> None:
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
                        frame="release.py",
                    )
                ]
            ]
        )

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("anchor")
    recorder.record_point("final")
    report = recorder.finish().lifetimes(
        "anchor",
        options=MemoryLifetimeOptions(events=True),
    )

    assert report.history_complete is False
    assert report.cohorts[0].event_exact_release_bytes == 64
    assert any("snapshot-inferred" in warning for warning in report.warnings)

    with pytest.raises(MemoryHistoryError, match="unavailable or incomplete"):
        recorder.result.lifetimes(
            "anchor",
            options=MemoryLifetimeOptions(events=True, on_missing="error"),
        )


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
                ],
                [
                    event("snapshot", marker=markers[0]),
                    event(
                        "free_requested",
                        address=1000,
                        size=64,
                        frame="free1.py",
                    ),
                ],
            ]
        return snapshot(*segments, traces=traces)

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("anchor")
    recorder.record_point("final")
    report = recorder.finish().lifetimes(
        "anchor",
        options=MemoryLifetimeOptions(events=True, on_missing="error"),
    )

    assert {item.device for item in report.cohorts} == {0, 1}
    release_stacks = {
        item.device: item.releases[0].stack_key for item in report.cohorts
    }
    assert "free0.py" in release_stacks[0]
    assert "free1.py" in release_stacks[1]


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
        "anchor",
        options=MemoryLifetimeOptions(events=True, on_missing="error"),
    )

    assert len(report.cohorts) == 1
    cohort = report.cohorts[0]
    assert cohort.device == 0
    assert [item.active_bytes for item in cohort.points] == [64, 64]
    assert cohort.event_exact_release_bytes == 0
    assert cohort.still_active_bytes == 64


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
                        frame="released.py",
                    ),
                ]
            ]
        return snapshot(reserved, traces=traces)

    recorder = MemoryRecorder._from_snapshot_provider(provider)
    recorder.record_point("before")
    recorder.record_point("after")
    report = recorder.finish().lifetimes(
        born_between=("before", "after"),
        options=MemoryLifetimeOptions(events=True, on_missing="error"),
    )

    assert len(report.cohorts) == 1
    assert report.cohorts[0].pool_id == (0, 7)


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
        "anchor",
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
        "release_stacks",
    }
    payload = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "allocation-lifetime-analysis"
    assert payload["cohorts"][0]["peak_active_bytes"] == 64
    assert "snapshot_inferred" in (output / "release_stacks.csv").read_text(
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
    assert "top allocation cohorts" in comparison.to_text()
    comparison_paths = comparison.write(tmp_path / "comparison")
    assert "cohorts" in comparison_paths

    timeline = run.timeline(attribution=options)
    assert timeline.allocation_lifetimes is not None
    assert all(item.allocation_lifetimes is None for item in timeline.point_comparisons)
    assert "top allocation cohorts" in timeline.to_text()
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
        run.lifetimes("after", through="before")
    with pytest.raises(MemoryOwnershipError):
        run.lifetimes(other["point"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        run.lifetimes("before", born_between=("before", "after"))
    with pytest.raises(ValueError, match="born_between end"):
        run.lifetimes(
            born_between=("before", "after"),
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
