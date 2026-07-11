from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryBundleError,
    MemoryDisplayOptions,
    MemoryRunGroup,
    compare_run_group_phases,
)

from ._helpers import make_run, segment, snapshot


def _distributed_run(
    values: tuple[int, ...],
    *,
    name: str,
    rank: int,
    group_id: str,
    world_size: int = 2,
    bundle_dir: Path | None = None,
    device_memory: tuple[tuple[int, int], ...] | None = None,
):
    return make_run(
        [snapshot(segment(active=value, device=rank)) for value in values],
        name=name,
        rank=rank,
        group_id=group_id,
        world_size=world_size,
        run_metadata={"source_revision": "abc123"},
        bundle_dir=bundle_dir,
        labels=("start", "end"),
        device_memory=(
            None
            if device_memory is None
            else [{rank: sample} for sample in device_memory]
        ),
    )


def test_group_load_reports_per_rank_extrema_without_sum(tmp_path: Path) -> None:
    root = tmp_path / "baseline"
    _distributed_run(
        (10, 30),
        name="baseline",
        rank=0,
        group_id="baseline-job",
        bundle_dir=root / "rank-00000.tcgd-memory",
        device_memory=((990, 1000), (950, 1000)),
    )
    _distributed_run(
        (12, 35),
        name="baseline",
        rank=1,
        group_id="baseline-job",
        bundle_dir=root / "rank-00001.tcgd-memory",
        device_memory=((980, 1000), (940, 1000)),
    )

    group = MemoryRunGroup.load(root)
    report = group.summary()

    assert group.ranks == (0, 1)
    assert group.group_id == "baseline-job"
    assert group.world_size == 2
    assert group.missing_ranks == ()
    assert group.complete is True
    assert group.point_labels == ("start", "end")
    assert not group.warnings
    assert group[0]["end"]._state_cache == {}
    assert len(report.rank_points) == 12
    assert len(report.rank_devices) == 4
    assert report.rank_devices[-1].device_index == 1
    assert report.rank_devices[-1].cuda_allocator_residual_bytes == 25
    text = report.to_text()
    assert "not summed across ranks" in text
    assert group[0]["end"].allocator_state()["segments"]
    assert group[0]["end"]._state_cache == {}
    end_active = next(
        item
        for item in report.point_aggregates
        if item.point_label == "end"
        and item.scope == "all"
        and item.metric == "active_bytes"
    )
    assert end_active.min_value == 30
    assert end_active.min_rank == 0
    assert end_active.max_value == 35
    assert end_active.worst_rank == 1
    assert end_active.spread_value == 5
    assert "allocated_bytes" in text
    assert "reserved_bytes" in text
    assert "active_bytes" in text
    assert "requested_bytes" in text
    assert "inactive_bytes" not in text
    assert "internal_fragmentation_bytes" not in text
    assert "device-wide CUDA Runtime states are per rank/device" in text

    paths = report.write(tmp_path / "summary")
    assert set(paths) == {
        "text",
        "json",
        "html",
        "rank_points",
        "rank_devices",
        "point_aggregates",
    }
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["aggregation"] == "per_rank_extrema_no_sum"
    assert payload["kind"] == "run-group-summary"
    assert len(payload["rank_devices"]) == 4
    assert any(
        row["metric"] == "internal_fragmentation_bytes"
        for row in payload["point_aggregates"]
    )
    html = paths["html"].read_text(encoding="utf-8")
    assert "active_bytes" in html
    assert "internal_fragmentation_bytes" not in html

    cached = MemoryRunGroup.load(root, cache_snapshots=True)
    assert cached[0]["end"].allocator_state()["segments"]
    assert cached[0]["end"]._state_cache


def test_group_validation_rejects_ambiguous_rank_identity_and_points() -> None:
    rank0 = _distributed_run((10, 20), name="run", rank=0, group_id="job-a")
    duplicate_rank = _distributed_run((11, 21), name="run", rank=0, group_id="job-a")
    with pytest.raises(MemoryBundleError, match="duplicate rank"):
        MemoryRunGroup.from_runs((rank0, duplicate_rank))

    rank1_other_group = _distributed_run((11, 21), name="run", rank=1, group_id="job-b")
    with pytest.raises(MemoryBundleError, match="group IDs differ"):
        MemoryRunGroup.from_runs((rank0, rank1_other_group))

    rank1_other_points = make_run(
        [snapshot(segment(active=11)), snapshot(segment(active=21))],
        name="run",
        rank=1,
        group_id="job-a",
        world_size=2,
        labels=("start", "different"),
    )
    with pytest.raises(MemoryBundleError, match="label sequences differ"):
        MemoryRunGroup.from_runs((rank0, rank1_other_points))

    with pytest.raises(MemoryBundleError, match="has no rank"):
        MemoryRunGroup.from_runs((replace(rank0, rank=None),))


def test_group_accepts_incomplete_prefix_rank_as_crashed() -> None:
    rank0 = _distributed_run((10, 20), name="run", rank=0, group_id="job")
    crashed = replace(
        make_run(
            [snapshot(segment(active=11, device=1))],
            name="run",
            rank=1,
            group_id="job",
            world_size=2,
            run_metadata={"source_revision": "abc123"},
            labels=("start",),
        ),
        complete=False,
    )

    group = MemoryRunGroup.from_runs((rank0, crashed))
    assert group.point_labels == ("start", "end")
    assert group.complete is False
    assert any(
        "memory bundles are incomplete for ranks 1" in item for item in group.warnings
    )
    summary = group.summary()
    rank1_labels = {row.point_label for row in summary.rank_points if row.rank == 1}
    assert rank1_labels == {"start"}
    end_aggregates = [
        row for row in summary.point_aggregates if row.point_label == "end"
    ]
    assert end_aggregates and all(row.rank_count == 1 for row in end_aggregates)

    # A complete rank with fewer points is a structural difference, not a
    # crash, and remains an error.
    with pytest.raises(MemoryBundleError, match="label sequences differ"):
        MemoryRunGroup.from_runs((rank0, replace(crashed, complete=True)))


def test_group_phase_comparison_skips_rank_missing_an_incomplete_tail() -> None:
    baseline = MemoryRunGroup.from_runs(
        (
            _distributed_run((10, 20), name="baseline", rank=0, group_id="base"),
            _distributed_run((11, 21), name="baseline", rank=1, group_id="base"),
        )
    )
    candidate_rank0 = _distributed_run(
        (12, 22), name="candidate", rank=0, group_id="candidate"
    )
    candidate_rank1 = replace(
        make_run(
            [snapshot(segment(active=13, device=1))],
            name="candidate",
            rank=1,
            group_id="candidate",
            world_size=2,
            run_metadata={"source_revision": "abc123"},
            labels=("start",),
        ),
        complete=False,
    )
    candidate = MemoryRunGroup.from_runs((candidate_rank0, candidate_rank1))

    report = compare_run_group_phases(
        baseline,
        candidate,
        baseline_start="start",
        baseline_end="end",
        candidate_start="start",
        candidate_end="end",
    )

    assert tuple(report.rank_comparisons) == (0,)
    assert any(
        "rank 1 skipped" in item
        and "candidate incomplete run is missing point 'end'" in item
        for item in report.warnings
    )
    assert report.phase_aggregates
    assert all(item.rank_count == 1 for item in report.phase_aggregates)


def test_group_phase_comparison_rejects_when_no_rank_has_the_phase() -> None:
    baseline = MemoryRunGroup.from_runs(
        (_distributed_run((10, 20), name="baseline", rank=0, group_id="base"),)
    )
    candidate_run = replace(
        make_run(
            [snapshot(segment(active=12))],
            name="candidate",
            rank=0,
            group_id="candidate",
            world_size=1,
            run_metadata={"source_revision": "abc123"},
            labels=("start",),
        ),
        complete=False,
    )
    candidate = MemoryRunGroup.from_runs((candidate_run,))

    with pytest.raises(MemoryBundleError, match="no common rank contains"):
        compare_run_group_phases(
            baseline,
            candidate,
            baseline_start="start",
            baseline_end="end",
            candidate_start="start",
            candidate_end="end",
        )


def test_group_warns_for_missing_rank_and_metadata_mismatch() -> None:
    rank0 = _distributed_run((10, 20), name="run", rank=0, group_id="job", world_size=3)
    rank2 = _distributed_run((10, 20), name="run", rank=2, group_id="job", world_size=3)
    rank2 = replace(rank2, run_metadata={"source_revision": "different"})

    group = MemoryRunGroup.from_runs((rank0, rank2))

    assert any("missing ranks 1" in warning for warning in group.warnings)
    assert group.missing_ranks == (1,)
    assert group.complete is False
    assert any("metadata differs" in warning for warning in group.warnings)
    assert any(
        row.metric == "awaiting_free_bytes" for row in group.summary().point_aggregates
    )


def test_group_phase_comparison_reports_worst_rank_and_spread(tmp_path: Path) -> None:
    baseline = MemoryRunGroup.from_runs(
        (
            _distributed_run(
                (10, 30),
                name="baseline",
                rank=0,
                group_id="baseline-job",
                device_memory=((900, 1000), (880, 1000)),
            ),
            _distributed_run(
                (12, 32),
                name="baseline",
                rank=1,
                group_id="baseline-job",
                device_memory=((890, 1000), (870, 1000)),
            ),
        )
    )

    candidate_runs = []
    for rank, states in (
        (0, ((20, 10), (35, 20))),
        (1, ((22, 8), (42, 20))),
    ):
        candidate_runs.append(
            make_run(
                [
                    snapshot(
                        segment(active=default, device=rank),
                        segment(active=private, pool=(0, 7), address=2000, device=rank),
                    )
                    for default, private in states
                ],
                name="candidate",
                rank=rank,
                group_id="candidate-job",
                world_size=2,
                run_metadata={"source_revision": "abc123"},
                labels=("start", "end"),
                device_memory=(
                    [
                        {rank: sample}
                        for sample in (
                            ((850, 1000), (800, 1000))
                            if rank == 0
                            else ((840, 1000), (780, 1000))
                        )
                    ]
                ),
            )
        )
    candidate = MemoryRunGroup.from_runs(candidate_runs)

    report = compare_run_group_phases(
        baseline,
        candidate,
        baseline_start="start",
        baseline_end="end",
        candidate_start="start",
        candidate_end="end",
    )

    assert tuple(report.rank_comparisons) == (0, 1)
    assert len(report.rank_decomposition) == 54
    assert len(report.rank_device_decomposition) == 10
    active = next(
        item
        for item in report.phase_aggregates
        if item.scope == "all" and item.metric == "active_bytes"
    )
    assert active.end_gap.min_bytes == 25
    assert active.end_gap.max_bytes == 30
    assert active.end_gap.max_rank == 1
    assert active.end_gap.spread_bytes == 5
    assert active.identity_holds is True
    text = report.to_text()
    assert "not summed across ranks" in text
    assert "end gap min +25 B on rank 0, max +30 B on rank 1" in text
    assert "change gap min" in text
    assert "awaiting_free_bytes" in text
    assert "expandable_reserved_bytes" in text
    assert "device-wide CUDA Runtime equations are per rank/device" in text
    changed_device_rows = report.rank_device_decomposition_rows(include_unchanged=False)
    assert changed_device_rows
    assert all(row["metric"] != "total_bytes" for row in changed_device_rows)

    paths = report.write(tmp_path / "phase")
    assert set(paths) == {
        "text",
        "json",
        "html",
        "rank_decomposition",
        "rank_pool_decomposition",
        "rank_device_decomposition",
        "phase_aggregates",
    }
    assert paths["rank_decomposition"].is_file()
    assert report.to_dict()["kind"] == "run-group-phase-comparison"


def test_group_phase_attribution_is_display_limited_but_csv_is_complete(
    tmp_path: Path,
) -> None:
    shared = {"filename": "shared.py", "line": 1, "name": "allocate"}
    left = {"filename": "left.py", "line": 2, "name": "forward"}
    right = {"filename": "right.py", "line": 3, "name": "forward"}

    def state(left_size: int, right_size: int):
        left_segment = segment(active=left_size, address=1000)
        right_segment = segment(active=right_size, address=2000)
        left_segment["blocks"][0]["frames"] = [shared, left]
        right_segment["blocks"][0]["frames"] = [shared, right]
        return snapshot(left_segment, right_segment)

    baseline_run = make_run(
        [state(10, 20), state(15, 30)],
        name="baseline",
        rank=0,
        group_id="baseline-job",
        world_size=1,
        labels=("start", "end"),
    )
    candidate_run = make_run(
        [state(20, 40), state(30, 60)],
        name="candidate",
        rank=0,
        group_id="candidate-job",
        world_size=1,
        labels=("start", "end"),
    )
    report = compare_run_group_phases(
        MemoryRunGroup.from_runs((baseline_run,)),
        MemoryRunGroup.from_runs((candidate_run,)),
        baseline_start="start",
        baseline_end="end",
        candidate_start="start",
        candidate_end="end",
        attribution=MemoryAttributionOptions(
            stacks=True,
            display=MemoryDisplayOptions(stack_depth=1, limit=1),
        ),
    )

    assert all(
        len(comparison.allocation_stack_comparisons) == 2
        for _name, comparison in (
            ("baseline", report.rank_comparisons[0].baseline_change),
            ("candidate", report.rank_comparisons[0].candidate_change),
            ("start", report.rank_comparisons[0].start_gap),
            ("end", report.rank_comparisons[0].end_gap),
        )
    )
    text_report = report.to_text()
    assert "showing 1 of 2" in text_report
    assert report.display_stack_depth == 1
    assert report.display_limit == 1
    assert "left.py" not in text_report
    assert "right.py" not in text_report

    paths = report.write(tmp_path / "group-phase-attribution")
    assert "allocation_stack_comparisons" in paths
    csv_text = paths["allocation_stack_comparisons"].read_text(encoding="utf-8")
    assert "left.py:2:forward" in csv_text
    assert "right.py:3:forward" in csv_text
    with paths["allocation_stack_comparisons"].open(
        newline="", encoding="utf-8"
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows
    assert all(json.loads(row["stack_frames_json"]) for row in rows)
    html = paths["html"].read_text(encoding="utf-8")
    assert "Showing 6 of 12 rows." in html
    assert "shared.py:1:allocate" in html
    assert "left.py:2:forward" not in html
    assert "right.py:3:forward" not in html


def test_group_provenance_compares_multi_device_properties_not_uuid() -> None:
    rank0 = _distributed_run((10, 20), name="run", rank=0, group_id="job")
    rank1 = _distributed_run((10, 20), name="run", rank=1, group_id="job")
    base_device = {
        "index": 0,
        "name": "GPU",
        "capability": [9, 0],
        "total_memory_bytes": 100,
        "uuid": "rank-specific-0",
    }
    rank0 = replace(rank0, provenance={"devices": [base_device]})
    rank1 = replace(
        rank1,
        provenance={"devices": [{**base_device, "uuid": "rank-specific-1"}]},
    )

    matching = MemoryRunGroup.from_runs((rank0, rank1))
    assert not any("provenance differs" in warning for warning in matching.warnings)

    different = MemoryRunGroup.from_runs(
        (
            rank0,
            replace(
                rank1,
                provenance={"devices": [{**base_device, "name": "other GPU"}]},
            ),
        )
    )
    assert any("provenance differs" in warning for warning in different.warnings)


def test_group_summary_qualifies_point_warnings() -> None:
    run = _distributed_run(
        (10, 20),
        name="run",
        rank=0,
        group_id="job",
        world_size=1,
    )
    run = replace(
        run,
        points=(
            replace(run.points[0], warnings=("sample failed",)),
            run.points[1],
        ),
    )
    summary = MemoryRunGroup.from_runs((run,)).summary()
    assert summary.warnings == ("rank 0 point [0] start: sample failed",)


def test_group_phase_accepts_rank_specific_device_mappings() -> None:
    baseline_run = make_run(
        [
            snapshot(segment(active=10, device=0)),
            snapshot(segment(active=20, device=0)),
        ],
        name="run",
        rank=0,
        group_id="baseline",
        world_size=1,
        labels=("start", "end"),
        device_memory=[{0: (90, 100)}, {0: (80, 100)}],
    )
    candidate_run = make_run(
        [
            snapshot(segment(active=10, device=1)),
            snapshot(segment(active=30, device=1)),
        ],
        name="run",
        rank=0,
        group_id="candidate",
        world_size=1,
        labels=("start", "end"),
        device_memory=[{1: (70, 100)}, {1: (50, 100)}],
    )
    report = compare_run_group_phases(
        MemoryRunGroup.from_runs((baseline_run,)),
        MemoryRunGroup.from_runs((candidate_run,)),
        baseline_start="start",
        baseline_end="end",
        candidate_start="start",
        candidate_end="end",
        device_mappings={0: {0: 1}},
    )
    assert report.rank_device_decomposition
    assert {
        (
            item.decomposition.baseline_device_index,
            item.decomposition.candidate_device_index,
        )
        for item in report.rank_device_decomposition
    } == {(0, 1)}
