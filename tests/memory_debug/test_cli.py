from __future__ import annotations

import json
from pathlib import Path

import pytest

from torch_cudagraph_debug.memory_debug.cli import build_parser, main

from ._helpers import event, make_history_run, make_run, segment, snapshot


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


def test_cli_summary_timeline_and_compare(tmp_path: Path, capsys) -> None:
    baseline, _ = _bundles(tmp_path)
    assert main(["summary", str(baseline)]) == 0
    assert "Memory run 'baseline'" in capsys.readouterr().out
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
    assert (timeline_output / "observations.csv").exists()

    comparison_output = tmp_path / "comparison"
    assert (
        main(
            [
                "compare-points",
                str(baseline),
                "--reference-point",
                "start",
                "--candidate-point",
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
    assert payload["kind"] == "point-comparison"
    assert payload["pool_comparisons"][0]["delta"]["active_bytes"] == 20


def test_cli_compare_runs_and_compare_phases(tmp_path: Path) -> None:
    baseline, candidate = _bundles(tmp_path)
    runs_output = tmp_path / "runs"
    assert (
        main(
            [
                "compare-points",
                str(baseline),
                str(candidate),
                "--reference-point",
                "end",
                "--candidate-point",
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
    assert (phase_output / "pool_decomposition.csv").exists()


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
                "compare-points",
                str(before_path),
                str(after_path),
                "--reference-point",
                "point",
                "--candidate-point",
                "point",
                "--pool-map",
                "0:0,1=0:0,8",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert payload["pool_comparisons"][0]["match"] == "mapped"


def test_cli_lifetimes_and_timeline_summary(tmp_path: Path, capsys) -> None:
    history_path = tmp_path / "history.tcgd-memory"
    make_history_run(
        [
            ([segment(active=10)], []),
            (
                [segment(active=10), segment(active=30, address=2000)],
                [event("alloc", address=2000, size=30, frame="grow.py")],
            ),
        ],
        labels=("start", "end"),
        bundle_dir=history_path,
    )

    lifetime_output = tmp_path / "lifetimes"
    assert (
        main(
            [
                "allocation-lifetimes",
                str(history_path),
                "--active-at",
                "start",
                "--through",
                "end",
                "--output",
                str(lifetime_output),
            ]
        )
        == 0
    )
    payload = json.loads((lifetime_output / "report.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "allocation-lifetime-analysis"
    assert (lifetime_output / "cohort_points.csv").exists()

    born_output = tmp_path / "born"
    assert (
        main(
            [
                "allocation-lifetimes",
                str(history_path),
                "--born-between",
                "start",
                "end",
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
                str(history_path),
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
    assert timeline_payload["point_comparisons"] == []

    traceless, _ = _bundles(tmp_path)
    assert (
        main(
            [
                "allocation-lifetimes",
                str(traceless),
                "--output",
                str(tmp_path / "rejected"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "event history is unavailable" in captured.err


def test_cli_group_summary_and_phase_comparison(tmp_path: Path) -> None:
    roots = {}
    for name, group_id, values in (
        ("baseline", "baseline-job", ((10, 30), (12, 32))),
        ("candidate", "candidate-job", ((20, 35), (22, 42))),
    ):
        root = tmp_path / name
        pool = (0, 1) if name == "baseline" else (0, 8)
        roots[name] = root
        for rank, states in enumerate(values):
            make_run(
                [snapshot(segment(active=value, pool=pool)) for value in states],
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
                "group-summary",
                str(roots["baseline"]),
                "--output",
                str(summary_output),
            ]
        )
        == 0
    )
    assert (summary_output / "rank_points.csv").exists()
    assert (summary_output / "point_aggregates.csv").exists()

    phase_output = tmp_path / "group-phase"
    assert (
        main(
            [
                "compare-run-group-phases",
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
                "--pool-map",
                "0@0:0,1=0:0,8",
                "--pool-map",
                "1@0:0,1=0:0,8",
                "--output",
                str(phase_output),
            ]
        )
        == 0
    )
    assert (phase_output / "rank_decomposition.csv").exists()
    assert (phase_output / "rank_pool_decomposition.csv").exists()
    assert (phase_output / "phase_aggregates.csv").exists()


def test_lifetime_cli_does_not_expose_irrelevant_common_flags() -> None:
    parser = build_parser()
    for flag in ("--only-changed", "--stacks", "--lifetimes", "--events"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "allocation-lifetimes",
                    "bundle",
                    "--output",
                    "report",
                    flag,
                ]
            )
