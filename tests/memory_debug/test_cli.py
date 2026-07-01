from __future__ import annotations

import json
from pathlib import Path

from torch_cudagraph_debug.memory_debug.cli import main

from ._helpers import make_run, segment, snapshot


def _bundles(tmp_path: Path) -> tuple[Path, Path]:
    baseline_path = tmp_path / "baseline.tcgd-memory"
    candidate_path = tmp_path / "candidate.tcgd-memory"
    make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=30))],
        name="baseline",
        bundle_dir=baseline_path,
        labels=("start", "end"),
    )
    make_run(
        [snapshot(segment(active=20)), snapshot(segment(active=35))],
        name="candidate",
        bundle_dir=candidate_path,
        labels=("start", "end"),
    )
    return baseline_path, candidate_path


def test_cli_timeline_and_compare(tmp_path: Path) -> None:
    baseline, _ = _bundles(tmp_path)
    timeline_output = tmp_path / "timeline"
    assert (
        main(
            [
                "timeline",
                str(baseline),
                "--output",
                str(timeline_output),
            ]
        )
        == 0
    )
    assert (timeline_output / "pools.csv").exists()
    assert (timeline_output / "pool_streams.csv").exists()

    comparison_output = tmp_path / "comparison"
    assert (
        main(
            [
                "compare",
                str(baseline),
                "--before",
                "start",
                "--after",
                "end",
                "--stacks",
                "--output",
                str(comparison_output),
            ]
        )
        == 0
    )
    payload = json.loads(
        (comparison_output / "report.json").read_text(encoding="utf-8")
    )
    assert payload["kind"] == "comparison"
    assert payload["pools"][0]["delta"]["active_bytes"] == 20


def test_cli_compare_runs_and_compare_phases(tmp_path: Path) -> None:
    baseline, candidate = _bundles(tmp_path)
    runs_output = tmp_path / "runs"
    assert (
        main(
            [
                "compare-runs",
                str(baseline),
                str(candidate),
                "--before",
                "end",
                "--after",
                "end",
                "--output",
                str(runs_output),
            ]
        )
        == 0
    )
    assert (runs_output / "report.txt").exists()

    phase_output = tmp_path / "phase"
    assert (
        main(
            [
                "compare-phases",
                str(baseline),
                str(candidate),
                "--baseline-start",
                "start",
                "--baseline-end",
                "end",
                "--candidate-start",
                "start",
                "--candidate-end",
                "end",
                "--output",
                str(phase_output),
            ]
        )
        == 0
    )
    assert (phase_output / "phase.csv").exists()


def test_cli_pool_mapping_syntax(tmp_path: Path) -> None:
    before_path = tmp_path / "before.tcgd-memory"
    after_path = tmp_path / "after.tcgd-memory"
    make_run(
        [snapshot(segment(active=10, pool=(0, 1)))],
        bundle_dir=before_path,
        labels=("point",),
    )
    make_run(
        [snapshot(segment(active=20, pool=(0, 8)))],
        bundle_dir=after_path,
        labels=("point",),
    )

    output = tmp_path / "mapped"
    assert (
        main(
            [
                "compare-runs",
                str(before_path),
                str(after_path),
                "--before",
                "point",
                "--after",
                "point",
                "--pool-map",
                "0,1=0,8",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert payload["pools"][0]["match"] == "mapped"


def test_cli_lifetimes_and_timeline_summary(tmp_path: Path) -> None:
    baseline, _ = _bundles(tmp_path)

    lifetime_output = tmp_path / "lifetimes"
    assert (
        main(
            [
                "lifetimes",
                str(baseline),
                "--at",
                "start",
                "--through",
                "end",
                "--no-events",
                "--output",
                str(lifetime_output),
            ]
        )
        == 0
    )
    payload = json.loads((lifetime_output / "report.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "allocation_lifetimes"
    assert (lifetime_output / "cohort_points.csv").exists()

    born_output = tmp_path / "born"
    assert (
        main(
            [
                "lifetimes",
                str(baseline),
                "--born-between",
                "start",
                "end",
                "--no-events",
                "--output",
                str(born_output),
            ]
        )
        == 0
    )
    born_payload = json.loads((born_output / "report.json").read_text(encoding="utf-8"))
    assert [item["label"] for item in born_payload["born_between"]] == [
        "start",
        "end",
    ]

    timeline_output = tmp_path / "timeline-lifetimes"
    assert (
        main(
            [
                "timeline",
                str(baseline),
                "--lifetimes",
                "--output",
                str(timeline_output),
            ]
        )
        == 0
    )
    timeline_payload = json.loads(
        (timeline_output / "report.json").read_text(encoding="utf-8")
    )
    assert timeline_payload["allocation_lifetimes"] is not None


def test_cli_group_summary_and_phase_comparison(tmp_path: Path) -> None:
    roots = {}
    for name, group_id, values in (
        ("baseline", "baseline-job", ((10, 30), (12, 32))),
        ("candidate", "candidate-job", ((20, 35), (22, 42))),
    ):
        root = tmp_path / name
        roots[name] = root
        for rank, states in enumerate(values):
            make_run(
                [snapshot(segment(active=value)) for value in states],
                name=name,
                rank=rank,
                group_id=group_id,
                world_size=2,
                bundle_dir=root / f"rank-{rank:05d}.tcgd-memory",
                labels=("start", "end"),
            )

    summary_output = tmp_path / "group-summary"
    assert (
        main(
            [
                "summarize-group",
                str(roots["baseline"]),
                "--output",
                str(summary_output),
            ]
        )
        == 0
    )
    assert (summary_output / "rank_points.csv").exists()
    assert (summary_output / "point_summary.csv").exists()

    phase_output = tmp_path / "group-phase"
    assert (
        main(
            [
                "compare-group-phases",
                str(roots["baseline"]),
                str(roots["candidate"]),
                "--baseline-start",
                "start",
                "--baseline-end",
                "end",
                "--candidate-start",
                "start",
                "--candidate-end",
                "end",
                "--output",
                str(phase_output),
            ]
        )
        == 0
    )
    assert (phase_output / "rank_phase.csv").exists()
    assert (phase_output / "phase_summary.csv").exists()
