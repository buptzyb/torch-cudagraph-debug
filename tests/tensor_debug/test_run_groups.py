from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from torch_cudagraph_debug.tensor_debug import (
    TensorBundleError,
    TensorComparisonError,
    TensorRunGroup,
    compare_run_groups,
)

from ._run_helpers import make_tensor_run


def _group_run(
    root: Path,
    *,
    name: str,
    group_id: str,
    rank: int,
    value: float,
    world_size: int = 2,
):
    return make_tensor_run(
        [("forward", [("output", torch.tensor([value]), "full")])],
        name=name,
        bundle_dir=root / f"rank-{rank:05d}.tcgd-tensor",
        rank=rank,
        group_id=group_id,
        world_size=world_size,
    )


def test_tensor_run_group_load_summary_and_compare(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    for rank in range(2):
        _group_run(
            reference_root,
            name="eager",
            group_id="eager-job",
            rank=rank,
            value=float(rank + 1),
        )
        _group_run(
            candidate_root,
            name="cuda-graph",
            group_id="graph-job",
            rank=rank,
            value=float(rank + 1),
        )

    reference = TensorRunGroup.load(reference_root)
    candidate = TensorRunGroup.load(candidate_root)
    summary = reference.summary()
    comparison = compare_run_groups(reference, candidate)

    assert reference.ranks == (0, 1)
    assert reference.missing_ranks == ()
    assert reference.complete is True
    assert comparison.conclusive is True
    assert reference.point_labels == ("forward",)
    assert len(summary.rank_points) == 2
    assert all(item.observation_count == 1 for item in summary.rank_points)
    assert comparison.ok
    comparison.assert_ok()
    assert "<!doctype html>" in summary.to_html()
    assert "<!doctype html>" in comparison.to_html()
    assert [item.status for item in comparison.rank_comparisons] == [
        "match",
        "match",
    ]

    summary_paths = summary.write(tmp_path / "summary")
    comparison_paths = comparison.write(tmp_path / "comparison")
    assert set(summary_paths) == {"text", "json", "html", "rank_points"}
    assert set(comparison_paths) == {"text", "json", "html", "rank_comparisons"}
    payload = json.loads(comparison_paths["json"].read_text(encoding="utf-8"))
    assert payload["status"] == "match"
    assert len(payload["rank_comparisons"]) == 2


def test_tensor_run_group_comparison_reports_rank_mismatch(tmp_path: Path) -> None:
    reference_runs = [
        make_tensor_run(
            [("forward", [("output", torch.tensor([float(rank)]), "full")])],
            name="eager",
            rank=rank,
            group_id="eager-job",
            world_size=2,
        )
        for rank in range(2)
    ]
    candidate_runs = [
        make_tensor_run(
            [
                (
                    "candidate_forward",
                    [
                        (
                            "output",
                            torch.tensor([99.0 if rank == 1 else float(rank)]),
                            "full",
                        )
                    ],
                )
            ],
            name="cuda-graph",
            rank=rank,
            group_id="graph-job",
            world_size=2,
        )
        for rank in range(2)
    ]

    comparison = compare_run_groups(
        TensorRunGroup.from_runs(reference_runs),
        TensorRunGroup.from_runs(candidate_runs),
        point_mapping={"forward": "candidate_forward"},
    )

    assert comparison.status == "mismatch"
    with pytest.raises(TensorComparisonError, match="run group comparison"):
        comparison.assert_ok()
    assert [item.status for item in comparison.rank_comparisons] == [
        "match",
        "mismatch",
    ]
    assert "rank 0" not in comparison.to_text(include_unchanged=False)
    assert "rank 1: mismatch" in comparison.to_text(include_unchanged=False)


def test_tensor_run_group_validates_rank_identity_and_completeness() -> None:
    rank0 = make_tensor_run(
        [("forward", [("output", torch.tensor([0.0]), "full")])],
        rank=0,
        group_id="job",
        world_size=3,
    )
    rank2 = make_tensor_run(
        [("forward", [("output", torch.tensor([2.0]), "full")])],
        rank=2,
        group_id="job",
        world_size=3,
    )
    group = TensorRunGroup.from_runs((rank0, rank2))
    assert any("missing ranks 1" in item for item in group.warnings)
    assert group.missing_ranks == (1,)
    assert group.complete is False

    with pytest.raises(TensorBundleError, match="duplicate rank"):
        TensorRunGroup.from_runs((rank0, rank0))
    with pytest.raises(TensorBundleError, match="point label sequences differ"):
        TensorRunGroup.from_runs(
            (
                rank0,
                make_tensor_run(
                    [("other", [("output", torch.tensor([1.0]), "full")])],
                    rank=1,
                    group_id="job",
                    world_size=3,
                ),
            )
        )


def _prefix_group_runs(*, name: str, group_id: str):
    rank0 = make_tensor_run(
        [
            ("a", [("output", torch.tensor([0.0]), "full")]),
            ("b", [("output", torch.tensor([1.0]), "full")]),
        ],
        name=name,
        rank=0,
        group_id=group_id,
        world_size=2,
    )
    rank1 = replace(
        make_tensor_run(
            [("a", [("output", torch.tensor([0.0]), "full")])],
            name=name,
            rank=1,
            group_id=group_id,
            world_size=2,
        ),
        complete=False,
    )
    return rank0, rank1


def test_tensor_run_group_accepts_incomplete_rank_with_prefix_labels() -> None:
    reference = TensorRunGroup.from_runs(
        _prefix_group_runs(name="eager", group_id="eager-job")
    )
    candidate = TensorRunGroup.from_runs(
        _prefix_group_runs(name="cuda-graph", group_id="graph-job")
    )

    assert reference.point_labels == ("a", "b")
    assert reference.complete is False
    assert any(
        "tensor bundles are incomplete for ranks 1" in item
        for item in reference.warnings
    )

    comparison = compare_run_groups(reference, candidate)
    assert comparison.status == "inconclusive"
    assert [item.status for item in comparison.rank_comparisons] == [
        "match",
        "match",
    ]


def test_crashed_rank_with_identical_shared_points_is_inconclusive() -> None:
    reference = TensorRunGroup.from_runs(
        make_tensor_run(
            [
                ("a", [("output", torch.tensor([0.0]), "full")]),
                ("b", [("output", torch.tensor([1.0]), "full")]),
            ],
            name="eager",
            rank=rank,
            group_id="eager-job",
            world_size=2,
        )
        for rank in range(2)
    )
    candidate = TensorRunGroup.from_runs(
        _prefix_group_runs(name="cuda-graph", group_id="graph-job")
    )

    comparison = compare_run_groups(reference, candidate)
    assert comparison.status == "inconclusive"
    assert [item.status for item in comparison.rank_comparisons] == [
        "match",
        "inconclusive",
    ]
    rank1 = comparison.rank_comparisons[1].comparison
    assert rank1.conclusive is False
    assert any("candidate run is incomplete" in item for item in rank1.warnings)


def test_crashed_rank_with_shared_point_divergence_stays_mismatch() -> None:
    reference = TensorRunGroup.from_runs(
        make_tensor_run(
            [
                ("a", [("output", torch.tensor([0.0]), "full")]),
                ("b", [("output", torch.tensor([1.0]), "full")]),
            ],
            name="eager",
            rank=rank,
            group_id="eager-job",
            world_size=2,
        )
        for rank in range(2)
    )
    healthy, _ = _prefix_group_runs(name="cuda-graph", group_id="graph-job")
    diverged_crash = replace(
        make_tensor_run(
            [("a", [("output", torch.tensor([9.0]), "full")])],
            name="cuda-graph",
            rank=1,
            group_id="graph-job",
            world_size=2,
        ),
        complete=False,
    )
    candidate = TensorRunGroup.from_runs((healthy, diverged_crash))

    comparison = compare_run_groups(reference, candidate)
    assert comparison.status == "mismatch"
    assert comparison.rank_comparisons[1].status == "mismatch"


def test_tensor_run_group_load_accepts_incomplete_prefix_rank_bundle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "group"
    make_tensor_run(
        [
            ("a", [("output", torch.tensor([0.0]), "full")]),
            ("b", [("output", torch.tensor([1.0]), "full")]),
        ],
        bundle_dir=root / "rank-00000.tcgd-tensor",
        rank=0,
        group_id="job",
        world_size=2,
    )
    make_tensor_run(
        [("a", [("output", torch.tensor([0.0]), "full")])],
        bundle_dir=root / "rank-00001.tcgd-tensor",
        rank=1,
        group_id="job",
        world_size=2,
    )
    manifest_path = root / "rank-00001.tcgd-tensor" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    group = TensorRunGroup.load(root)
    assert group.point_labels == ("a", "b")
    assert group.complete is False
    assert any(
        "tensor bundles are incomplete for ranks 1" in item for item in group.warnings
    )


def test_tensor_run_group_still_rejects_non_prefix_point_sequences() -> None:
    rank0 = make_tensor_run(
        [
            ("a", [("output", torch.tensor([0.0]), "full")]),
            ("b", [("output", torch.tensor([1.0]), "full")]),
        ],
        rank=0,
        group_id="job",
        world_size=2,
    )
    complete_prefix = make_tensor_run(
        [("a", [("output", torch.tensor([0.0]), "full")])],
        rank=1,
        group_id="job",
        world_size=2,
    )
    with pytest.raises(TensorBundleError, match="point label sequences differ"):
        TensorRunGroup.from_runs((rank0, complete_prefix))

    incomplete_non_prefix = replace(
        make_tensor_run(
            [("b", [("output", torch.tensor([1.0]), "full")])],
            rank=1,
            group_id="job",
            world_size=2,
        ),
        complete=False,
    )
    with pytest.raises(TensorBundleError, match="point label sequences differ"):
        TensorRunGroup.from_runs((rank0, incomplete_non_prefix))


def test_tensor_run_group_report_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    run = make_tensor_run(
        [("forward", [("output", torch.tensor([1.0]), "full")])],
        rank=0,
        group_id="job",
        world_size=1,
    )
    summary = TensorRunGroup.from_runs((run,)).summary()
    output = tmp_path / "report"
    summary.write(output)

    with pytest.raises(FileExistsError, match="not empty"):
        summary.write(output)
    summary.write(output, overwrite=True)


def test_tensor_run_group_comparison_is_inconclusive_when_rank_is_missing() -> None:
    reference = TensorRunGroup.from_runs(
        tuple(
            make_tensor_run(
                [("forward", [("output", torch.tensor([float(rank)]), "full")])],
                name="eager",
                rank=rank,
                group_id="eager-job",
                world_size=2,
            )
            for rank in range(2)
        )
    )
    candidate = TensorRunGroup.from_runs(
        (
            make_tensor_run(
                [("forward", [("output", torch.tensor([0.0]), "full")])],
                name="cuda-graph",
                rank=0,
                group_id="graph-job",
                world_size=2,
            ),
        )
    )

    comparison = compare_run_groups(reference, candidate)

    assert comparison.status == "inconclusive"
    assert comparison.ok is False
    assert comparison.conclusive is False
    assert comparison.reference_only_ranks == (1,)
    assert comparison.candidate_only_ranks == ()
    assert comparison.to_dict()["reference_only_ranks"] == [1]


def test_tensor_run_group_warns_for_partial_identity_and_nested_metadata() -> None:
    rank0 = make_tensor_run(
        [("forward", [("output", torch.tensor([0.0]), "full")])],
        rank=0,
        group_id="job",
        world_size=2,
    )
    rank1 = make_tensor_run(
        [("forward", [("output", torch.tensor([1.0]), "full")])],
        rank=1,
        group_id=None,
        world_size=2,
    )
    rank0 = replace(rank0, run_metadata={"nested": {"value": 1}})
    rank1 = replace(rank1, run_metadata={"nested": {"value": 2}})

    group = TensorRunGroup.from_runs((rank0, rank1))

    assert group.group_id == "job"
    assert any(
        "group identity is unavailable for ranks 1" in item for item in group.warnings
    )
    assert any("user run metadata differs" in item for item in group.warnings)
