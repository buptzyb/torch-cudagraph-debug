from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from torch_cudagraph_debug.memory_debug import (
    MemoryBundleError,
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
):
    return make_run(
        [snapshot(segment(active=value)) for value in values],
        name=name,
        rank=rank,
        group_id=group_id,
        world_size=world_size,
        run_metadata={"source_revision": "abc123"},
        bundle_dir=bundle_dir,
        labels=("start", "end"),
    )


def test_group_load_reports_per_rank_extrema_without_sum(tmp_path: Path) -> None:
    root = tmp_path / "baseline"
    _distributed_run(
        (10, 30),
        name="baseline",
        rank=0,
        group_id="baseline-job",
        bundle_dir=root / "rank-00000.tcgd-memory",
    )
    _distributed_run(
        (12, 35),
        name="baseline",
        rank=1,
        group_id="baseline-job",
        bundle_dir=root / "rank-00001.tcgd-memory",
    )

    group = MemoryRunGroup.load(root)
    report = group.summary()

    assert group.ranks == (0, 1)
    assert group.group_id == "baseline-job"
    assert group.world_size == 2
    assert group.point_labels == ("start", "end")
    assert not group.warnings
    assert group[0]["end"]._snapshot_cache == {}
    assert group[0]["end"].raw_snapshot()["segments"]
    assert group[0]["end"]._snapshot_cache == {}
    assert len(report.rank_point_entries) == 12
    end_active = next(
        row
        for row in report.point_aggregates
        if row["point_label"] == "end"
        and row["scope"] == "all"
        and row["metric"] == "active_bytes"
    )
    assert end_active["min_value"] == 30
    assert end_active["min_rank"] == 0
    assert end_active["max_value"] == 35
    assert end_active["worst_rank"] == 1
    assert end_active["spread_value"] == 5
    assert "not summed across ranks" in report.to_text()

    paths = report.write(tmp_path / "summary")
    assert set(paths) == {
        "text",
        "json",
        "html",
        "rank_point_entries",
        "point_aggregates",
    }
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["aggregation"] == "per_rank_extrema_no_sum"
    assert payload["kind"] == "run-group-summary"

    cached = MemoryRunGroup.load(root, cache_snapshots=True)
    assert cached[0]["end"].raw_snapshot()["segments"]
    assert cached[0]["end"]._snapshot_cache


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


def test_group_warns_for_missing_rank_and_metadata_mismatch() -> None:
    rank0 = _distributed_run((10, 20), name="run", rank=0, group_id="job", world_size=3)
    rank2 = _distributed_run((10, 20), name="run", rank=2, group_id="job", world_size=3)
    rank2 = replace(rank2, run_metadata={"source_revision": "different"})

    group = MemoryRunGroup.from_runs((rank0, rank2))

    assert any("missing ranks 1" in warning for warning in group.warnings)
    assert any("metadata differs" in warning for warning in group.warnings)


def test_group_phase_comparison_reports_worst_rank_and_spread(tmp_path: Path) -> None:
    baseline = MemoryRunGroup.from_runs(
        (
            _distributed_run(
                (10, 30), name="baseline", rank=0, group_id="baseline-job"
            ),
            _distributed_run(
                (12, 32), name="baseline", rank=1, group_id="baseline-job"
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
                        segment(active=default),
                        segment(active=private, pool=(0, 7), address=2000),
                    )
                    for default, private in states
                ],
                name="candidate",
                rank=rank,
                group_id="candidate-job",
                world_size=2,
                run_metadata={"source_revision": "abc123"},
                labels=("start", "end"),
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
    assert len(report.rank_decomposition) == 30
    active = next(
        row
        for row in report.phase_aggregates
        if row["scope"] == "all" and row["metric"] == "active_bytes"
    )
    assert active["end_gap_min_bytes"] == 25
    assert active["end_gap_max_bytes"] == 30
    assert active["end_gap_max_rank"] == 1
    assert active["end_gap_spread_bytes"] == 5
    assert active["identity_holds"] is True
    assert "not summed across ranks" in report.to_text()

    paths = report.write(tmp_path / "phase")
    assert set(paths) == {
        "text",
        "json",
        "html",
        "rank_decomposition",
        "phase_aggregates",
    }
    assert paths["rank_decomposition"].is_file()
    assert report.to_dict()["kind"] == "run-group-phase-comparison"
