from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from torch_cudagraph_debug.tensor_debug import (
    TensorComparisonError,
    TensorComparisonOptions,
    TensorObservationKey,
    compare_point_series,
    compare_points,
    compare_runs,
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
    assert report.first_issue.key == TensorObservationKey("second", 0)
    assert report.first_issue.mismatch_count == 1
    assert report.first_issue.mismatch_fraction == pytest.approx(0.5)
    assert report.first_issue.max_abs_error == pytest.approx(1.0)
    assert report.first_issue.first_mismatch_index == (1,)
    assert report.first_issue.reference_value == 4.0
    assert report.first_issue.candidate_value == 5.0


def test_int64_error_metrics_preserve_differences_above_float64_precision() -> None:
    reference = make_tensor_run(
        [
            (
                "point",
                [("x", torch.tensor([2**53, 0], dtype=torch.int64), "full")],
            )
        ]
    )
    candidate = make_tensor_run(
        [
            (
                "point",
                [("x", torch.tensor([2**53 + 1, 0], dtype=torch.int64), "full")],
            )
        ]
    )

    report = compare_points(reference["point"], candidate["point"])
    issue = report.first_issue
    assert issue is not None
    assert issue.mismatch_count == 1
    assert issue.max_abs_error == 1.0
    assert issue.mean_abs_error == 0.5
    assert issue.reference_value == 2**53
    assert issue.candidate_value == 2**53 + 1

    minimum = make_tensor_run(
        [("point", [("x", torch.tensor([-(2**63)], dtype=torch.int64), "full")])]
    )
    maximum = make_tensor_run(
        [("point", [("x", torch.tensor([2**63 - 1], dtype=torch.int64), "full")])]
    )
    extreme = compare_points(minimum["point"], maximum["point"]).first_issue
    assert extreme is not None
    assert extreme.max_abs_error == 2**64 - 1


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
        options=TensorComparisonOptions(mode="exact"),
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
        options=TensorComparisonOptions(dtype_policy="promote"),
    )
    assert promoted.status == "match"


def test_exact_comparison_detects_signed_zero_bits() -> None:
    reference = make_tensor_run([("point", [("x", torch.tensor([-0.0]), "full")])])
    candidate = make_tensor_run([("point", [("x", torch.tensor([0.0]), "full")])])

    report = compare_points(
        reference["point"],
        candidate["point"],
        options=TensorComparisonOptions(mode="exact"),
    )
    assert report.status == "mismatch"
    assert report.first_issue is not None
    assert report.first_issue.mismatch_count == 1


def test_bit_identical_nans_do_not_fast_path_to_match_without_equal_nan() -> None:
    value = torch.tensor([float("nan"), 1.0])
    reference = make_tensor_run([("point", [("x", value, "full")])])
    candidate = make_tensor_run([("point", [("x", value.clone(), "full")])])
    assert (
        reference["point"].observation("x").sha256
        == candidate["point"].observation("x").sha256
    )

    report = compare_points(reference["point"], candidate["point"])
    assert report.status == "mismatch"
    assert report.first_issue is not None
    assert report.first_issue.mismatch_count == 1
    assert report.first_issue.first_mismatch_index == (0,)

    lenient = compare_points(
        reference["point"],
        candidate["point"],
        options=TensorComparisonOptions(equal_nan=True),
    )
    assert lenient.status == "match"


def test_exact_equal_nan_exempts_both_nan_positions_bitwise() -> None:
    reference_value = torch.tensor([float("nan"), 1.0], dtype=torch.float32)
    candidate_value = torch.tensor([0x7FC00001, 0], dtype=torch.int32).view(
        torch.float32
    )
    candidate_value[1] = 1.0
    assert torch.isnan(candidate_value[0])
    assert int(reference_value.view(torch.int32)[0]) != int(
        candidate_value.view(torch.int32)[0]
    )
    reference = make_tensor_run([("point", [("x", reference_value, "full")])])
    candidate = make_tensor_run([("point", [("x", candidate_value, "full")])])

    strict = compare_points(
        reference["point"],
        candidate["point"],
        options=TensorComparisonOptions(mode="exact"),
    )
    assert strict.status == "mismatch"

    lenient = compare_points(
        reference["point"],
        candidate["point"],
        options=TensorComparisonOptions(mode="exact", equal_nan=True),
    )
    assert lenient.status == "match"


def test_exact_equal_nan_still_detects_signed_zero_bits() -> None:
    reference = make_tensor_run([("point", [("x", torch.tensor([-0.0]), "full")])])
    candidate = make_tensor_run([("point", [("x", torch.tensor([0.0]), "full")])])

    report = compare_points(
        reference["point"],
        candidate["point"],
        options=TensorComparisonOptions(mode="exact", equal_nan=True),
    )
    assert report.status == "mismatch"
    assert report.first_issue is not None
    assert report.first_issue.mismatch_count == 1


def test_exact_promote_honors_equal_nan_across_dtypes() -> None:
    reference = make_tensor_run(
        [
            (
                "point",
                [("x", torch.tensor([float("nan"), 1.0], dtype=torch.float64), "full")],
            )
        ]
    )
    candidate = make_tensor_run(
        [
            (
                "point",
                [("x", torch.tensor([float("nan"), 1.0], dtype=torch.float32), "full")],
            )
        ]
    )

    report = compare_points(
        reference["point"],
        candidate["point"],
        options=TensorComparisonOptions(
            mode="exact", dtype_policy="promote", equal_nan=True
        ),
    )
    assert report.status == "match"


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
    with pytest.raises(TensorComparisonError, match="run comparison"):
        runs.assert_ok()
    assert runs.candidate_only_points == ("step-3",)
    assert runs.first_issue is not None
    assert runs.to_dict()["kind"] == "run-comparison"
    assert "<table>" in runs.to_html(include_unchanged=False)
    compact_runs = runs.to_text(include_unchanged=False)
    assert "    [mismatch]" in compact_runs
    assert "    [match]" not in compact_runs

    series = compare_point_series(reference["step-1"], candidate)
    assert series.status == "mismatch"
    with pytest.raises(TensorComparisonError, match="point-series comparison"):
        series.assert_ok()
    assert series.conclusive
    assert len(series.point_comparisons) == 3
    assert series.point_comparisons[0].status == "match"
    assert series.to_dict()["kind"] == "point-series-comparison"
    assert "<table>" in series.to_html(include_unchanged=False)
    compact_series = series.to_text(include_unchanged=False)
    assert "    [mismatch]" in compact_series
    assert "    [match]" not in compact_series


def test_point_mapping_merges_with_identical_label_auto_alignment() -> None:
    reference = make_tensor_run(
        [
            ("a", [("x", torch.tensor([1.0]), "full")]),
            ("b", [("x", torch.tensor([2.0]), "full")]),
        ],
        name="reference",
    )
    candidate = make_tensor_run(
        [
            ("a", [("x", torch.tensor([1.0]), "full")]),
            ("b", [("x", torch.tensor([2.0]), "full")]),
        ],
        name="candidate",
    )

    report = compare_runs(reference, candidate, point_mapping={"a": "a"})
    assert report.status == "match"
    assert report.reference_only_points == ()
    assert report.candidate_only_points == ()
    assert len(report.point_comparisons) == 2


def test_point_mapping_pairs_explicit_entries_and_auto_aligned_labels() -> None:
    reference = make_tensor_run(
        [
            ("a", [("x", torch.tensor([1.0]), "full")]),
            ("c", [("x", torch.tensor([2.0]), "full")]),
        ],
        name="reference",
    )
    candidate = make_tensor_run(
        [
            ("b", [("x", torch.tensor([1.0]), "full")]),
            ("c", [("x", torch.tensor([2.0]), "full")]),
        ],
        name="candidate",
    )

    report = compare_runs(reference, candidate, point_mapping={"a": "b"})
    assert report.status == "match"
    assert report.reference_only_points == ()
    assert report.candidate_only_points == ()
    assert {
        (item.reference.label, item.candidate.label)
        for item in report.point_comparisons
    } == {("a", "b"), ("c", "c")}


def test_point_mapping_rejects_candidate_label_claimed_twice() -> None:
    reference = make_tensor_run(
        [
            ("a", [("x", torch.tensor([1.0]), "full")]),
            ("b", [("x", torch.tensor([2.0]), "full")]),
        ],
        name="reference",
    )
    candidate = make_tensor_run(
        [("b", [("x", torch.tensor([1.0]), "full")])],
        name="candidate",
    )

    with pytest.raises(ValueError, match="one-to-one"):
        compare_runs(reference, candidate, point_mapping={"a": "b"})


def test_series_rejects_an_empty_candidate_run() -> None:
    reference = make_tensor_run([("reference", [("x", torch.tensor([1.0]), "full")])])
    empty = make_tensor_run([], name="empty")

    with pytest.raises(ValueError, match="must be non-empty"):
        compare_point_series(reference["reference"], empty)


def test_series_requires_a_real_candidate_run() -> None:
    reference = make_tensor_run([("reference", [("x", torch.ones(1), "full")])])

    with pytest.raises(TypeError, match="TensorRun"):
        compare_point_series(reference["reference"], reference.points)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"rtol": float("nan")},
        {"atol": float("inf")},
        {"limit": True},
        {"equal_nan": 1},
    ),
)
def test_comparison_options_reject_lossy_scalar_types(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        TensorComparisonOptions(**kwargs)  # type: ignore[arg-type]


def test_series_first_issue_text_labels_inconclusive_summary() -> None:
    reference = make_tensor_run(
        [("reference", [("x", torch.tensor([1.0]), "summary")])]
    )
    candidate = make_tensor_run(
        [("replay", [("x", torch.tensor([1.000001]), "summary")])]
    )

    report = compare_point_series(reference["reference"], candidate)
    assert report.status == "inconclusive"
    assert not report.conclusive
    assert "[inconclusive/payload]" in report.to_text()
    assert report.to_dict()["conclusive"] is False


def test_reports_write_text_json_html_and_csv(tmp_path: Path) -> None:
    reference = make_tensor_run([("reference", [("x", torch.tensor([1.0]), "full")])])
    candidate = make_tensor_run([("candidate", [("x", torch.tensor([2.0]), "full")])])
    report = compare_points(reference["reference"], candidate["candidate"])

    paths = report.write(tmp_path / "report", include_unchanged=False)
    assert set(paths) == {"text", "json", "html", "observations"}
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["status"] == "mismatch"
    assert payload["observation_comparisons"][0]["name"] == "x"
    assert "Tensor comparison" in paths["text"].read_text(encoding="utf-8")
    assert "<table>" in paths["html"].read_text(encoding="utf-8")


def test_layout_policy_uses_recorded_source_stride() -> None:
    reference_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    candidate_tensor = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]).t()
    assert torch.equal(reference_tensor, candidate_tensor)
    assert reference_tensor.stride() != candidate_tensor.stride()

    reference = make_tensor_run(
        [("point", [("hidden", reference_tensor, "full")])],
        name="reference",
    )
    candidate = make_tensor_run(
        [("point", [("hidden", candidate_tensor, "full")])],
        name="candidate",
    )

    strict = compare_points(reference["point"], candidate["point"])
    assert strict.status == "mismatch"
    assert strict.observation_comparisons[0].kind == "metadata"
    assert "stride" in strict.observation_comparisons[0].reason

    ignored = compare_points(
        reference["point"],
        candidate["point"],
        options=TensorComparisonOptions(layout_policy="ignore"),
    )
    assert ignored.ok
    assert ignored.to_dict()["options"]["layout_policy"] == "ignore"
