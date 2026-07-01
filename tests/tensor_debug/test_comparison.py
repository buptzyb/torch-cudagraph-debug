from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from torch_cudagraph_debug.tensor_debug import (
    TensorCompareOptions,
    compare_points,
    compare_runs,
    compare_series,
)

from ._run_helpers import make_tensor_run


def test_allclose_reports_first_divergence_and_error_metrics() -> None:
    reference = make_tensor_run(
        [
            (
                "forward",
                [
                    ("first", torch.tensor([1.0, 2.0]), "full"),
                    ("second", torch.tensor([3.0, 4.0]), "full"),
                ],
            )
        ],
        name="reference",
    )
    candidate = make_tensor_run(
        [
            (
                "replay",
                [
                    ("first", torch.tensor([1.0, 2.0]), "full"),
                    ("second", torch.tensor([3.0, 5.0]), "full"),
                ],
            )
        ],
        name="candidate",
    )

    report = compare_points(reference["forward"], candidate["replay"])
    assert report.status == "mismatch"
    assert report.matched_count == 1
    assert report.mismatched_count == 1
    assert report.first_issue is not None
    assert report.first_issue.key == ("second", 0)
    assert report.first_issue.mismatch_count == 1
    assert report.first_issue.mismatch_fraction == pytest.approx(0.5)
    assert report.first_issue.max_abs_error == pytest.approx(1.0)
    assert report.first_issue.first_mismatch_index == (1,)
    assert report.first_issue.reference_value == 4.0
    assert report.first_issue.candidate_value == 5.0


def test_summary_comparison_is_strictly_three_state() -> None:
    reference = make_tensor_run(
        [("point", [("x", torch.tensor([1.0, 2.0]), "summary")])]
    )
    same = make_tensor_run([("point", [("x", torch.tensor([1.0, 2.0]), "summary")])])
    changed = make_tensor_run(
        [("point", [("x", torch.tensor([1.0, 2.000001]), "summary")])]
    )

    assert compare_points(reference["point"], same["point"]).status == "match"
    allclose = compare_points(reference["point"], changed["point"])
    assert allclose.status == "inconclusive"
    assert allclose.first_issue is not None
    assert allclose.first_issue.kind == "payload"

    exact = compare_points(
        reference["point"],
        changed["point"],
        options=TensorCompareOptions(mode="exact"),
    )
    assert exact.status == "mismatch"
    assert exact.first_issue is not None
    assert exact.first_issue.reason == "tensor digests differ"


def test_dtype_is_strict_by_default_and_can_be_promoted() -> None:
    reference = make_tensor_run(
        [("point", [("x", torch.tensor([1.0, 2.0], dtype=torch.float16), "full")])]
    )
    candidate = make_tensor_run(
        [("point", [("x", torch.tensor([1.0, 2.0], dtype=torch.float32), "full")])]
    )

    strict = compare_points(reference["point"], candidate["point"])
    assert strict.status == "mismatch"
    assert strict.first_issue is not None
    assert strict.first_issue.kind == "metadata"

    promoted = compare_points(
        reference["point"],
        candidate["point"],
        options=TensorCompareOptions(dtype_policy="promote"),
    )
    assert promoted.status == "match"


def test_exact_comparison_detects_signed_zero_bits() -> None:
    reference = make_tensor_run([("point", [("x", torch.tensor([-0.0]), "full")])])
    candidate = make_tensor_run([("point", [("x", torch.tensor([0.0]), "full")])])

    report = compare_points(
        reference["point"],
        candidate["point"],
        options=TensorCompareOptions(mode="exact"),
    )
    assert report.status == "mismatch"
    assert report.first_issue is not None
    assert report.first_issue.mismatch_count == 1


def test_missing_observations_and_reordered_keys_are_reported() -> None:
    reference = make_tensor_run(
        [
            (
                "point",
                [
                    ("a", torch.ones(1), "full"),
                    ("b", torch.ones(1), "full"),
                ],
            )
        ]
    )
    reordered = make_tensor_run(
        [
            (
                "point",
                [
                    ("b", torch.ones(1), "full"),
                    ("a", torch.ones(1), "full"),
                ],
            )
        ]
    )
    report = compare_points(reference["point"], reordered["point"])
    assert report.status == "match"
    assert report.warnings

    missing = make_tensor_run([("point", [("a", torch.ones(1), "full")])])
    report = compare_points(reference["point"], missing["point"])
    assert report.status == "mismatch"
    assert report.first_issue is not None
    assert report.first_issue.kind == "reference_only"


def test_run_and_series_comparison_compose_point_comparisons() -> None:
    reference = make_tensor_run(
        [
            ("step-1", [("x", torch.tensor([1.0]), "full")]),
            ("step-2", [("x", torch.tensor([2.0]), "full")]),
        ],
        name="eager",
    )
    candidate = make_tensor_run(
        [
            ("step-1", [("x", torch.tensor([1.0]), "full")]),
            ("step-2", [("x", torch.tensor([3.0]), "full")]),
            ("step-3", [("x", torch.tensor([4.0]), "full")]),
        ],
        name="cg",
    )

    runs = compare_runs(reference, candidate)
    assert runs.status == "mismatch"
    assert runs.candidate_only_points == ("step-3",)
    assert runs.first_issue is not None

    series = compare_series(reference["step-1"], candidate)
    assert series.status == "mismatch"
    assert series.conclusive
    assert len(series.comparisons) == 3
    assert series.comparisons[0].status == "match"


def test_series_rejects_an_empty_candidate_run() -> None:
    reference = make_tensor_run([("reference", [("x", torch.tensor([1.0]), "full")])])
    empty = make_tensor_run([], name="empty")

    with pytest.raises(ValueError, match="must be non-empty"):
        compare_series(reference["reference"], empty)


def test_series_first_issue_text_labels_inconclusive_summary() -> None:
    reference = make_tensor_run(
        [("reference", [("x", torch.tensor([1.0]), "summary")])]
    )
    candidate = make_tensor_run(
        [("replay", [("x", torch.tensor([1.000001]), "summary")])]
    )

    report = compare_series(reference["reference"], candidate)
    assert report.status == "inconclusive"
    assert not report.conclusive
    assert "[inconclusive/payload]" in report.to_text()
    assert report.to_dict()["conclusive"] is False


def test_reports_write_text_json_html_and_csv(tmp_path: Path) -> None:
    reference = make_tensor_run([("reference", [("x", torch.tensor([1.0]), "full")])])
    candidate = make_tensor_run([("candidate", [("x", torch.tensor([2.0]), "full")])])
    report = compare_points(reference["reference"], candidate["candidate"])

    paths = report.write(tmp_path / "report", include_matches=False)
    assert set(paths) == {"text", "json", "html", "observations"}
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["status"] == "mismatch"
    assert payload["differences"][0]["probe_name"] == "x"
    assert "Tensor comparison" in paths["text"].read_text(encoding="utf-8")
    assert "<table>" in paths["html"].read_text(encoding="utf-8")
