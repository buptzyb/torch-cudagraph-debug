"""Offline tensor point, run, and replay-series comparison."""

from __future__ import annotations

import csv
import html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import torch

from .errors import TensorComparisonError
from .recording import (
    TensorObservation,
    TensorObservationKey,
    TensorPoint,
    TensorRun,
)

from .snapshots import TensorProbeSnapshot

ComparisonMode = Literal["allclose", "exact"]
DTypePolicy = Literal["strict", "promote"]
ComparisonStatus = Literal["match", "mismatch", "inconclusive"]
ObservationComparisonKind = Literal[
    "match",
    "value",
    "metadata",
    "reference_only",
    "candidate_only",
    "payload",
]
_COMPARE_CHUNK_ELEMENTS = 1_000_000
REPORT_SCHEMA = "torch-cudagraph-debug/tensor-report"


@dataclass(frozen=True)
class TensorComparisonOptions:
    """Numerical and metadata policy for offline tensor comparison."""

    mode: ComparisonMode = "allclose"
    rtol: float = 1e-5
    atol: float = 1e-8
    equal_nan: bool = False
    dtype_policy: DTypePolicy = "strict"
    limit: int = 20

    def __post_init__(self) -> None:
        if self.mode not in {"allclose", "exact"}:
            raise ValueError('mode must be either "allclose" or "exact"')
        if self.rtol < 0:
            raise ValueError("rtol must be non-negative")
        if self.atol < 0:
            raise ValueError("atol must be non-negative")
        if self.dtype_policy not in {"strict", "promote"}:
            raise ValueError('dtype_policy must be either "strict" or "promote"')
        if self.limit < 1:
            raise ValueError("limit must be >= 1")


@dataclass(frozen=True)
class TensorObservationComparison:
    """Comparison result for one stable (probe name, invocation) key."""

    probe_name: str
    invocation_index: int
    status: ComparisonStatus
    kind: ObservationComparisonKind
    reason: str
    reference: TensorObservation | None
    candidate: TensorObservation | None
    mismatch_count: int | None = None
    total_count: int | None = None
    mismatch_fraction: float | None = None
    max_abs_error: float | None = None
    max_relative_error: float | None = None
    mean_abs_error: float | None = None
    first_mismatch_index: tuple[int, ...] | None = None
    reference_value: bool | int | float | str | None = None
    candidate_value: bool | int | float | str | None = None

    @property
    def key(self) -> TensorObservationKey:
        return TensorObservationKey(self.probe_name, self.invocation_index)

    @property
    def changed(self) -> bool:
        return self.status != "match"

    def to_row(self) -> dict[str, object]:
        reference = self.reference
        candidate = self.candidate
        return {
            "probe_name": self.probe_name,
            "invocation_index": self.invocation_index,
            "status": self.status,
            "kind": self.kind,
            "reason": self.reason,
            "reference_shape": _shape_text(reference),
            "candidate_shape": _shape_text(candidate),
            "reference_dtype": _dtype_text(reference),
            "candidate_dtype": _dtype_text(candidate),
            "reference_payload": reference.payload if reference else "",
            "candidate_payload": candidate.payload if candidate else "",
            "reference_sha256": reference.sha256 if reference else "",
            "candidate_sha256": candidate.sha256 if candidate else "",
            "mismatch_count": self.mismatch_count,
            "total_count": self.total_count,
            "mismatch_fraction": self.mismatch_fraction,
            "max_abs_error": self.max_abs_error,
            "max_relative_error": self.max_relative_error,
            "mean_abs_error": self.mean_abs_error,
            "first_mismatch_index": (
                list(self.first_mismatch_index)
                if self.first_mismatch_index is not None
                else None
            ),
            "reference_value": self.reference_value,
            "candidate_value": self.candidate_value,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.to_row(),
            "reference": self.reference.descriptor() if self.reference else None,
            "candidate": self.candidate.descriptor() if self.candidate else None,
        }


@dataclass(frozen=True)
class _TensorStateComparison:
    """Shared comparison behavior for Point and ProbeSnapshot states."""

    reference: TensorPoint | TensorProbeSnapshot
    candidate: TensorPoint | TensorProbeSnapshot
    options: TensorComparisonOptions
    _COMPARISON_KIND: ClassVar[str] = "state-comparison"
    _COMPARISON_TITLE: ClassVar[str] = "Tensor comparison"
    observation_comparisons: tuple[TensorObservationComparison, ...]
    warnings: tuple[str, ...] = ()

    @property
    def status(self) -> ComparisonStatus:
        if any(item.status == "mismatch" for item in self.observation_comparisons):
            return "mismatch"
        if any(item.status == "inconclusive" for item in self.observation_comparisons):
            return "inconclusive"
        return "match"

    @property
    def ok(self) -> bool:
        return self.status == "match"

    @property
    def conclusive(self) -> bool:
        return not any(
            item.status == "inconclusive" for item in self.observation_comparisons
        )

    @property
    def first_issue(self) -> TensorObservationComparison | None:
        return next(
            (item for item in self.observation_comparisons if item.changed), None
        )

    @property
    def matched_count(self) -> int:
        return sum(item.status == "match" for item in self.observation_comparisons)

    @property
    def mismatched_count(self) -> int:
        return sum(item.status == "mismatch" for item in self.observation_comparisons)

    @property
    def inconclusive_count(self) -> int:
        return sum(
            item.status == "inconclusive" for item in self.observation_comparisons
        )

    def worst_observation_comparisons(
        self,
        *,
        limit: int | None = None,
    ) -> tuple[TensorObservationComparison, ...]:
        selected = [
            item
            for item in self.observation_comparisons
            if (
                item.status == "mismatch"
                and item.kind == "value"
                and item.mismatch_fraction is not None
            )
        ]
        selected.sort(
            key=lambda item: (
                item.mismatch_fraction if item.mismatch_fraction is not None else -1.0,
                item.max_abs_error if item.max_abs_error is not None else -1.0,
            ),
            reverse=True,
        )
        return tuple(selected[: limit or self.options.limit])

    def assert_ok(self) -> None:
        if not self.ok:
            issue = self.first_issue
            detail = issue.reason if issue is not None else self.status
            raise TensorComparisonError(
                f"{self._COMPARISON_TITLE.lower()} {_state_display(self.reference)} -> "
                f"{_state_display(self.candidate)} is {self.status}: {detail}"
            )

    def observation_comparison_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.observation_comparisons
            if include_unchanged or item.changed
        ]

    def to_text(self, *, include_unchanged: bool = True) -> str:
        lines = [
            f"{self._COMPARISON_TITLE} {_state_display(self.reference)} -> "
            f"{_state_display(self.candidate)}: {self.status}",
            "  "
            f"matched={self.matched_count} mismatched={self.mismatched_count} "
            f"inconclusive={self.inconclusive_count}",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        issue = self.first_issue
        if issue is not None:
            lines.append(f"  first issue: {_issue_text(issue)}")
        worst = self.worst_observation_comparisons()
        if worst:
            lines.append("  worst value mismatches:")
            for item in worst:
                lines.append(
                    "    "
                    f"{_key_text(item.probe_name, item.invocation_index)} "
                    f"mismatch={item.mismatch_fraction:.3%} "
                    f"max_abs={item.max_abs_error!r}"
                )
        selected = [
            item
            for item in self.observation_comparisons
            if include_unchanged or item.changed
        ]
        lines.append("  observations:")
        if not selected:
            lines.append("    no changed observations")
        for item in selected:
            line = (
                f"    [{item.status}] "
                f"{_key_text(item.probe_name, item.invocation_index)}: "
                f"{item.reason}"
            )
            if item.mismatch_count is not None and item.total_count is not None:
                line += (
                    f"; mismatched={item.mismatch_count}/{item.total_count} "
                    f"({item.mismatch_fraction:.3%})"
                )
            if item.max_abs_error is not None:
                line += f"; max_abs={item.max_abs_error:.6g}"
            if item.max_relative_error is not None:
                line += f"; max_rel={item.max_relative_error:.6g}"
            if item.first_mismatch_index is not None:
                line += (
                    f"; first={item.first_mismatch_index} "
                    f"{item.reference_value!r}->{item.candidate_value!r}"
                )
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": self._COMPARISON_KIND,
            "status": self.status,
            "conclusive": self.conclusive,
            "reference": self.reference.descriptor(),
            "candidate": self.candidate.descriptor(),
            "options": _options_dict(self.options),
            "warnings": list(self.warnings),
            "summary": {
                "matched": self.matched_count,
                "mismatched": self.mismatched_count,
                "inconclusive": self.inconclusive_count,
            },
            "worst_observation_comparisons": [
                item.to_row() for item in self.worst_observation_comparisons()
            ],
            "observation_comparisons": [
                item.to_dict() for item in self.observation_comparisons
            ],
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        return _html_report(
            f"{self._COMPARISON_TITLE} {_state_label(self.reference)} to {_state_label(self.candidate)}",
            self.status,
            self.observation_comparison_rows(include_unchanged=include_unchanged),
            self.warnings,
        )

    def write(
        self,
        output_dir: str | Path,
        *,
        include_unchanged: bool = True,
    ) -> dict[str, Path]:
        return _write_report(
            output_dir,
            text=self.to_text(include_unchanged=include_unchanged),
            payload=self.to_dict(),
            html_text=self.to_html(include_unchanged=include_unchanged),
            rows=self.observation_comparison_rows(include_unchanged=include_unchanged),
        )


@dataclass(frozen=True)
class TensorPointComparison(_TensorStateComparison):
    """Difference between two tensor recording points."""

    reference: TensorPoint
    candidate: TensorPoint
    _COMPARISON_KIND: ClassVar[str] = "point-comparison"
    _COMPARISON_TITLE: ClassVar[str] = "Tensor comparison"


@dataclass(frozen=True)
class TensorSnapshotComparison(_TensorStateComparison):
    """Difference between two standalone tensor Probe snapshots."""

    reference: TensorProbeSnapshot
    candidate: TensorProbeSnapshot
    _COMPARISON_KIND: ClassVar[str] = "snapshot-comparison"
    _COMPARISON_TITLE: ClassVar[str] = "Tensor snapshot comparison"


@dataclass(frozen=True)
class TensorRunComparison:
    """Point-aligned comparison between two complete tensor runs."""

    reference: TensorRun
    candidate: TensorRun
    point_comparisons: tuple[TensorPointComparison, ...]
    reference_only_points: tuple[str, ...] = ()
    candidate_only_points: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def status(self) -> ComparisonStatus:
        if self.reference_only_points or self.candidate_only_points:
            return "mismatch"
        if any(item.status == "mismatch" for item in self.point_comparisons):
            return "mismatch"
        if any(item.status == "inconclusive" for item in self.point_comparisons):
            return "inconclusive"
        return "match"

    @property
    def ok(self) -> bool:
        return self.status == "match"

    @property
    def conclusive(self) -> bool:
        return not any(not item.conclusive for item in self.point_comparisons)

    @property
    def first_issue(self) -> tuple[str, TensorObservationComparison | None] | None:
        for comparison in self.point_comparisons:
            if not comparison.ok:
                return (comparison.candidate.label, comparison.first_issue)
        if self.reference_only_points:
            return (self.reference_only_points[0], None)
        if self.candidate_only_points:
            return (self.candidate_only_points[0], None)
        return None

    def to_text(self, *, include_unchanged: bool = True) -> str:
        lines = [
            f"Tensor run comparison {self.reference.name!r} -> "
            f"{self.candidate.name!r}: {self.status}",
            f"  compared points={len(self.point_comparisons)}",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        if self.reference_only_points:
            lines.append(
                f"  reference-only points: {', '.join(self.reference_only_points)}"
            )
        if self.candidate_only_points:
            lines.append(
                f"  candidate-only points: {', '.join(self.candidate_only_points)}"
            )
        for comparison in self.point_comparisons:
            lines.append(
                f"  point {comparison.reference.label!r} -> "
                f"{comparison.candidate.label!r}: {comparison.status} "
                f"(matched={comparison.matched_count}, "
                f"mismatched={comparison.mismatched_count}, "
                f"inconclusive={comparison.inconclusive_count})"
            )
            if comparison.first_issue is not None:
                issue = comparison.first_issue
                lines.append(f"    first issue: {_issue_text(issue)}")
            for observation_comparison in comparison.observation_comparisons:
                if not include_unchanged and not observation_comparison.changed:
                    continue
                lines.append(
                    f"    [{observation_comparison.status}] "
                    f"{_key_text(observation_comparison.probe_name, observation_comparison.invocation_index)}: "
                    f"{observation_comparison.reason}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "run-comparison",
            "status": self.status,
            "conclusive": self.conclusive,
            "reference": self.reference.descriptor(),
            "candidate": self.candidate.descriptor(),
            "reference_only_points": list(self.reference_only_points),
            "candidate_only_points": list(self.candidate_only_points),
            "warnings": list(self.warnings),
            "point_comparisons": [item.to_dict() for item in self.point_comparisons],
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        return _html_report(
            f"Tensor run comparison {self.reference.name} to {self.candidate.name}",
            self.status,
            _aggregate_rows(
                self.point_comparisons,
                include_unchanged=include_unchanged,
            ),
            self.warnings,
        )

    def write(
        self,
        output_dir: str | Path,
        *,
        include_unchanged: bool = True,
    ) -> dict[str, Path]:
        rows = _aggregate_rows(
            self.point_comparisons, include_unchanged=include_unchanged
        )
        return _write_report(
            output_dir,
            text=self.to_text(include_unchanged=include_unchanged),
            payload=self.to_dict(),
            html_text=self.to_html(include_unchanged=include_unchanged),
            rows=rows,
        )


@dataclass(frozen=True)
class TensorPointSeriesComparison:
    """One reference point compared with an ordered candidate point series."""

    reference: TensorPoint
    candidate_run: TensorRun
    point_comparisons: tuple[TensorPointComparison, ...]

    @property
    def status(self) -> ComparisonStatus:
        if any(item.status == "mismatch" for item in self.point_comparisons):
            return "mismatch"
        if any(item.status == "inconclusive" for item in self.point_comparisons):
            return "inconclusive"
        return "match"

    @property
    def ok(self) -> bool:
        return self.status == "match"

    @property
    def conclusive(self) -> bool:
        return not any(not item.conclusive for item in self.point_comparisons)

    @property
    def first_issue(self) -> tuple[str, TensorObservationComparison] | None:
        for comparison in self.point_comparisons:
            if comparison.first_issue is not None:
                return (comparison.candidate.label, comparison.first_issue)
        return None

    def to_text(self, *, include_unchanged: bool = True) -> str:
        lines = [
            f"Tensor point-series comparison reference={_state_display(self.reference)} "
            f"candidate_run={self.candidate_run.name!r}: {self.status}"
        ]
        for comparison in self.point_comparisons:
            lines.append(
                f"  {comparison.candidate.label}: {comparison.status} "
                f"(matched={comparison.matched_count}, "
                f"mismatched={comparison.mismatched_count}, "
                f"inconclusive={comparison.inconclusive_count})"
            )
            if comparison.first_issue is not None:
                issue = comparison.first_issue
                lines.append(f"    first issue: {_issue_text(issue)}")
            for observation_comparison in comparison.observation_comparisons:
                if not include_unchanged and not observation_comparison.changed:
                    continue
                lines.append(
                    f"    [{observation_comparison.status}] "
                    f"{_key_text(observation_comparison.probe_name, observation_comparison.invocation_index)}: "
                    f"{observation_comparison.reason}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "point-series-comparison",
            "status": self.status,
            "conclusive": self.conclusive,
            "reference": self.reference.descriptor(),
            "candidate_run": self.candidate_run.descriptor(),
            "point_comparisons": [item.to_dict() for item in self.point_comparisons],
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        return _html_report(
            f"Tensor point-series comparison against {_state_label(self.reference)}",
            self.status,
            _aggregate_rows(
                self.point_comparisons,
                include_unchanged=include_unchanged,
            ),
            (),
        )

    def write(
        self,
        output_dir: str | Path,
        *,
        include_unchanged: bool = True,
    ) -> dict[str, Path]:
        rows = _aggregate_rows(
            self.point_comparisons, include_unchanged=include_unchanged
        )
        return _write_report(
            output_dir,
            text=self.to_text(include_unchanged=include_unchanged),
            payload=self.to_dict(),
            html_text=self.to_html(include_unchanged=include_unchanged),
            rows=rows,
        )


def compare_snapshots(
    reference: TensorProbeSnapshot,
    candidate: TensorProbeSnapshot,
    *,
    options: TensorComparisonOptions | None = None,
) -> TensorSnapshotComparison:
    """Compare standalone tensor snapshots by probe/invocation key."""

    if (
        reference.probe_id == candidate.probe_id
        and candidate.replay_index <= reference.replay_index
    ):
        raise ValueError("candidate snapshot must follow reference snapshot")
    selected = options or TensorComparisonOptions()
    observation_comparisons, warnings = _compare_observation_sets(
        reference.observations,
        candidate.observations,
        selected,
    )
    return TensorSnapshotComparison(
        reference=reference,
        candidate=candidate,
        options=selected,
        observation_comparisons=observation_comparisons,
        warnings=warnings,
    )


def compare_points(
    reference: TensorPoint,
    candidate: TensorPoint,
    *,
    options: TensorComparisonOptions | None = None,
) -> TensorPointComparison:
    """Compare two tensor points using stable probe/invocation keys."""

    selected = options or TensorComparisonOptions()
    observation_comparisons, warnings = _compare_observation_sets(
        reference.observations,
        candidate.observations,
        selected,
    )
    return TensorPointComparison(
        reference=reference,
        candidate=candidate,
        options=selected,
        observation_comparisons=observation_comparisons,
        warnings=warnings,
    )


def _compare_observation_sets(
    reference: Sequence[TensorObservation],
    candidate: Sequence[TensorObservation],
    options: TensorComparisonOptions,
) -> tuple[tuple[TensorObservationComparison, ...], tuple[str, ...]]:
    reference_by_key = {item.key: item for item in reference}
    candidate_by_key = {item.key: item for item in candidate}
    observation_comparisons: list[TensorObservationComparison] = []
    matched_keys: set[TensorObservationKey] = set()
    for observation in reference:
        other = candidate_by_key.get(observation.key)
        if other is None:
            observation_comparisons.append(
                TensorObservationComparison(
                    probe_name=observation.probe_name,
                    invocation_index=observation.invocation_index,
                    status="mismatch",
                    kind="reference_only",
                    reason="observation is missing from candidate",
                    reference=observation,
                    candidate=None,
                )
            )
            continue
        matched_keys.add(observation.key)
        observation_comparisons.append(
            _compare_observations(observation, other, options)
        )
    for observation in candidate:
        if observation.key in matched_keys or observation.key in reference_by_key:
            continue
        observation_comparisons.append(
            TensorObservationComparison(
                probe_name=observation.probe_name,
                invocation_index=observation.invocation_index,
                status="mismatch",
                kind="candidate_only",
                reason="observation is missing from reference",
                reference=None,
                candidate=observation,
            )
        )

    warnings: list[str] = []
    reference_keys = [item.key for item in reference]
    candidate_keys = [item.key for item in candidate]
    if set(reference_keys) == set(candidate_keys) and reference_keys != candidate_keys:
        warnings.append(
            "candidate observation order differs from reference; stable keys were matched"
        )
    return tuple(observation_comparisons), tuple(warnings)


def compare_runs(
    reference: TensorRun,
    candidate: TensorRun,
    *,
    point_mapping: Mapping[str, str] | None = None,
    options: TensorComparisonOptions | None = None,
) -> TensorRunComparison:
    """Compare aligned points from two runs."""

    reference_by_label = {point.label: point for point in reference.points}
    candidate_by_label = {point.label: point for point in candidate.points}
    if point_mapping is None:
        mapping = {
            point.label: point.label
            for point in reference.points
            if point.label in candidate_by_label
        }
    else:
        mapping = dict(point_mapping)
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("point_mapping candidate labels must be one-to-one")
        missing_reference = set(mapping) - set(reference_by_label)
        missing_candidate = set(mapping.values()) - set(candidate_by_label)
        if missing_reference:
            raise KeyError(
                f"reference point labels do not exist: {sorted(missing_reference)!r}"
            )
        if missing_candidate:
            raise KeyError(
                f"candidate point labels do not exist: {sorted(missing_candidate)!r}"
            )

    point_comparisons = tuple(
        compare_points(
            reference_by_label[reference_label],
            candidate_by_label[candidate_label],
            options=options,
        )
        for reference_label, candidate_label in mapping.items()
    )
    matched_reference = set(mapping)
    matched_candidate = set(mapping.values())
    reference_only = tuple(
        point.label
        for point in reference.points
        if point.label not in matched_reference
    )
    candidate_only = tuple(
        point.label
        for point in candidate.points
        if point.label not in matched_candidate
    )
    warnings = []
    if reference.execution != candidate.execution:
        warnings.append(
            f"execution differs: {reference.execution} -> {candidate.execution}"
        )
    if reference.provenance != candidate.provenance:
        warnings.append("runtime provenance differs between tensor runs")
    if reference.run_metadata != candidate.run_metadata:
        warnings.append("run metadata differs between tensor runs")
    return TensorRunComparison(
        reference=reference,
        candidate=candidate,
        point_comparisons=point_comparisons,
        reference_only_points=reference_only,
        candidate_only_points=candidate_only,
        warnings=tuple(warnings),
    )


def compare_point_series(
    reference: TensorPoint,
    candidates: TensorRun | Sequence[TensorPoint],
    *,
    options: TensorComparisonOptions | None = None,
) -> TensorPointSeriesComparison:
    """Compare one reference point against an ordered candidate series."""

    points = (
        candidates.points if isinstance(candidates, TensorRun) else tuple(candidates)
    )
    if not points:
        raise ValueError("candidate series must be non-empty")
    if isinstance(candidates, TensorRun):
        candidate_run = candidates
    else:
        run_ids = {point.run_id for point in points}
        if len(run_ids) != 1:
            raise ValueError("candidate series points must belong to one run")
        candidate_run = TensorRun(
            run_id=points[0].run_id,
            name="candidate-series",
            execution="eager",
            rank=None,
            created_at=points[0].timestamp,
            finished_at=None,
            complete=False,
            default_payload="full",
            points=tuple(points),
        )
    return TensorPointSeriesComparison(
        reference=reference,
        candidate_run=candidate_run,
        point_comparisons=tuple(
            compare_points(reference, point, options=options) for point in points
        ),
    )


def _compare_observations(
    reference: TensorObservation,
    candidate: TensorObservation,
    options: TensorComparisonOptions,
) -> TensorObservationComparison:
    if reference.shape != candidate.shape:
        return _metadata_comparison(
            reference,
            candidate,
            f"shape differs: {reference.shape} -> {candidate.shape}",
        )
    dtype_changed = reference.dtype != candidate.dtype
    if dtype_changed and options.dtype_policy == "strict":
        return _metadata_comparison(
            reference,
            candidate,
            f"dtype differs: {reference.dtype} -> {candidate.dtype}",
        )

    if not dtype_changed and reference.sha256 == candidate.sha256:
        return TensorObservationComparison(
            probe_name=reference.probe_name,
            invocation_index=reference.invocation_index,
            status="match",
            kind="match",
            reason="tensor values are identical",
            reference=reference,
            candidate=candidate,
            mismatch_count=0,
            total_count=reference.summary.numel,
            mismatch_fraction=0.0,
            max_abs_error=0.0,
            max_relative_error=0.0,
            mean_abs_error=0.0,
        )

    if options.mode == "exact" and not dtype_changed:
        if not reference.has_payload or not candidate.has_payload:
            return TensorObservationComparison(
                probe_name=reference.probe_name,
                invocation_index=reference.invocation_index,
                status="mismatch",
                kind="value",
                reason="tensor digests differ",
                reference=reference,
                candidate=candidate,
                total_count=reference.summary.numel,
            )
    elif not reference.has_payload or not candidate.has_payload:
        return TensorObservationComparison(
            probe_name=reference.probe_name,
            invocation_index=reference.invocation_index,
            status="inconclusive",
            kind="payload",
            reason=(
                "tensor digests differ but full payload is unavailable for "
                f"{options.mode} comparison"
            ),
            reference=reference,
            candidate=candidate,
            total_count=reference.summary.numel,
        )

    return _compare_full_payloads(reference, candidate, options)


def _metadata_comparison(
    reference: TensorObservation,
    candidate: TensorObservation,
    reason: str,
) -> TensorObservationComparison:
    return TensorObservationComparison(
        probe_name=reference.probe_name,
        invocation_index=reference.invocation_index,
        status="mismatch",
        kind="metadata",
        reason=reason,
        reference=reference,
        candidate=candidate,
    )


def _compare_full_payloads(
    reference: TensorObservation,
    candidate: TensorObservation,
    options: TensorComparisonOptions,
) -> TensorObservationComparison:
    reference_tensor = reference.tensor().reshape(-1)
    candidate_tensor = candidate.tensor().reshape(-1)
    numel = reference_tensor.numel()
    mismatch_count = 0
    first_flat_index: int | None = None
    reference_value: bool | int | float | str | None = None
    candidate_value: bool | int | float | str | None = None
    max_abs = 0.0
    max_relative = 0.0
    total_abs = 0.0
    finite_error_count = 0
    nonfinite_error = False

    compare_dtype = (
        torch.promote_types(reference.dtype, candidate.dtype)
        if options.dtype_policy == "promote"
        else reference.dtype
    )
    for start in range(0, numel, _COMPARE_CHUNK_ELEMENTS):
        reference_chunk = reference_tensor[start : start + _COMPARE_CHUNK_ELEMENTS]
        candidate_chunk = candidate_tensor[start : start + _COMPARE_CHUNK_ELEMENTS]
        if options.mode == "exact" and reference.dtype == candidate.dtype:
            item_size = reference_chunk.element_size()
            reference_bytes = reference_chunk.contiguous().view(torch.uint8)
            candidate_bytes = candidate_chunk.contiguous().view(torch.uint8)
            mismatch = (
                reference_bytes.reshape(-1, item_size)
                != candidate_bytes.reshape(-1, item_size)
            ).any(dim=1)
        else:
            candidate_values = candidate_chunk.to(compare_dtype)
            reference_values = reference_chunk.to(compare_dtype)
            if options.mode == "exact" or not (
                candidate_values.is_floating_point()
                or reference_values.is_floating_point()
            ):
                close = candidate_values == reference_values
                if options.equal_nan and candidate_values.is_floating_point():
                    close = close | (
                        torch.isnan(candidate_values) & torch.isnan(reference_values)
                    )
            else:
                close = torch.isclose(
                    candidate_values,
                    reference_values,
                    rtol=options.rtol,
                    atol=options.atol,
                    equal_nan=options.equal_nan,
                )
            mismatch = ~close

        chunk_mismatches = int(mismatch.sum().item())
        mismatch_count += chunk_mismatches
        if chunk_mismatches and first_flat_index is None:
            local = int(torch.nonzero(mismatch, as_tuple=False)[0].item())
            first_flat_index = start + local
            reference_value = _scalar_value(reference_chunk[local])
            candidate_value = _scalar_value(candidate_chunk[local])

        reference_values = reference_chunk.to(torch.float64)
        candidate_values = candidate_chunk.to(torch.float64)
        absolute = torch.abs(candidate_values - reference_values)
        finite = torch.isfinite(absolute)
        if bool(finite.any()):
            finite_values = absolute[finite]
            max_abs = max(max_abs, float(finite_values.max().item()))
            total_abs += float(finite_values.sum().item())
            finite_error_count += finite_values.numel()
        if not bool(finite.all()):
            nonfinite_error = True

        denominator = torch.abs(reference_values)
        relative = torch.where(
            denominator == 0,
            torch.where(
                absolute == 0,
                torch.zeros_like(absolute),
                torch.full_like(absolute, math.inf),
            ),
            absolute / denominator,
        )
        finite_relative = relative[torch.isfinite(relative)]
        if finite_relative.numel():
            max_relative = max(max_relative, float(finite_relative.max().item()))
        if bool(torch.isinf(relative).any()):
            max_relative = math.inf

    status: ComparisonStatus = "match" if mismatch_count == 0 else "mismatch"
    reason = (
        f"tensor values satisfy {options.mode} comparison"
        if status == "match"
        else f"tensor values fail {options.mode} comparison"
    )
    return TensorObservationComparison(
        probe_name=reference.probe_name,
        invocation_index=reference.invocation_index,
        status=status,
        kind="match" if status == "match" else "value",
        reason=reason,
        reference=reference,
        candidate=candidate,
        mismatch_count=mismatch_count,
        total_count=numel,
        mismatch_fraction=(mismatch_count / numel if numel else 0.0),
        max_abs_error=None if nonfinite_error else max_abs,
        max_relative_error=(None if not math.isfinite(max_relative) else max_relative),
        mean_abs_error=(
            total_abs / finite_error_count
            if finite_error_count and not nonfinite_error
            else (0.0 if numel == 0 else None)
        ),
        first_mismatch_index=(
            _unravel_index(first_flat_index, reference.shape)
            if first_flat_index is not None
            else None
        ),
        reference_value=reference_value,
        candidate_value=candidate_value,
    )


def _unravel_index(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    coordinates = [0] * len(shape)
    value = index
    for axis in range(len(shape) - 1, -1, -1):
        coordinates[axis] = value % shape[axis]
        value //= shape[axis]
    return tuple(coordinates)


def _scalar_value(tensor: torch.Tensor) -> bool | int | float | str:
    value = tensor.item()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return value


def _shape_text(observation: TensorObservation | None) -> str:
    return str(observation.shape) if observation is not None else ""


def _dtype_text(observation: TensorObservation | None) -> str:
    return str(observation.dtype) if observation is not None else ""


def _key_text(name: str, invocation: int) -> str:
    return f"{name}[{invocation}]"


def _state_label(state: TensorPoint | TensorProbeSnapshot) -> str:
    if isinstance(state, TensorPoint):
        return state.label
    return f"{state.probe_name} replay {state.replay_index}"


def _state_display(state: TensorPoint | TensorProbeSnapshot) -> str:
    if isinstance(state, TensorPoint):
        return repr(state.label)
    return f"{state.probe_name!r} replay={state.replay_index}"


def _issue_text(issue: TensorObservationComparison) -> str:
    return (
        f"{_key_text(issue.probe_name, issue.invocation_index)} "
        f"[{issue.status}/{issue.kind}] {issue.reason}"
    )


def _options_dict(options: TensorComparisonOptions) -> dict[str, object]:
    return {
        "mode": options.mode,
        "rtol": options.rtol,
        "atol": options.atol,
        "equal_nan": options.equal_nan,
        "dtype_policy": options.dtype_policy,
        "limit": options.limit,
    }


def _aggregate_rows(
    point_comparisons: Sequence[TensorPointComparison],
    *,
    include_unchanged: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for comparison in point_comparisons:
        for row in comparison.observation_comparison_rows(
            include_unchanged=include_unchanged
        ):
            rows.append(
                {
                    "reference_point": comparison.reference.label,
                    "candidate_point": comparison.candidate.label,
                    **row,
                }
            )
    return rows


def _write_report(
    output_dir: str | Path,
    *,
    text: str,
    payload: Mapping[str, object],
    html_text: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Path]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    text_path = root / "report.txt"
    json_path = root / "report.json"
    html_path = root / "report.html"
    csv_path = root / "observations.csv"
    text_path.write_text(text + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(html_text, encoding="utf-8")
    _write_csv(csv_path, rows)
    return {
        "text": text_path,
        "json": json_path,
        "html": html_path,
        "observations": csv_path,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _html_report(
    title: str,
    status: str,
    rows: Sequence[Mapping[str, object]],
    warnings: Sequence[str],
) -> str:
    warning_html = "".join(f"<li>{html.escape(warning)}</li>" for warning in warnings)
    if rows:
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        header = "".join(f"<th>{html.escape(name)}</th>" for name in columns)
        body = "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(_cell_text(row.get(name)))}</td>" for name in columns
            )
            + "</tr>"
            for row in rows
        )
        table = f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
    else:
        table = "<p>No selected observations.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 24px; color: #1f2933; }}
h1 {{ font-size: 22px; }}
.status {{ font-weight: 700; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #d8dee4; padding: 6px; text-align: left; }}
th {{ background: #f3f4f6; position: sticky; top: 0; }}
tr:nth-child(even) {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="status">Status: {html.escape(status)}</p>
<ul>{warning_html}</ul>
{table}
</body>
</html>
"""


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    return str(value)
