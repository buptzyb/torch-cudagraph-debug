"""Multi-rank TensorRun validation, summaries, and comparisons."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from types import MappingProxyType

from .._reporting import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    prepare_output_dir,
)
from .._validation import comparable_provenance, json_signature
from .comparison import (
    REPORT_SCHEMA,
    ComparisonStatus,
    TensorComparisonOptions,
    TensorRunComparison,
    compare_runs,
)
from .errors import TensorBundleError, TensorComparisonError
from .recording import ExecutionMode, TensorRun


@dataclass(frozen=True)
class TensorRankPointSummary:
    """Payload inventory for one rank and point in a tensor run group."""

    rank: int
    point_index: int
    point_label: str
    observation_count: int
    full_payload_count: int
    summary_payload_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "point_index": self.point_index,
            "point_label": self.point_label,
            "observation_count": self.observation_count,
            "full_payload_count": self.full_payload_count,
            "summary_payload_count": self.summary_payload_count,
        }


@dataclass(frozen=True)
class TensorRunGroup:
    """Validated rank-indexed collection of compatible tensor runs."""

    name: str
    execution: ExecutionMode
    runs: Mapping[int, TensorRun]
    point_labels: tuple[str, ...]
    group_id: str | None
    world_size: int | None
    warnings: tuple[str, ...] = ()
    root: Path | None = field(default=None, compare=False)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(self.runs)

    @property
    def missing_ranks(self) -> tuple[int, ...]:
        if self.world_size is None:
            return ()
        return tuple(sorted(set(range(self.world_size)) - set(self.runs)))

    @property
    def complete(self) -> bool:
        return (
            self.world_size is not None
            and not self.missing_ranks
            and all(run.complete for run in self.runs.values())
        )

    def __len__(self) -> int:
        return len(self.runs)

    def __getitem__(self, rank: int) -> TensorRun:
        return self.runs[rank]

    def descriptor(self) -> dict[str, object]:
        return {
            "name": self.name,
            "execution": self.execution,
            "group_id": self.group_id,
            "world_size": self.world_size,
            "ranks": list(self.ranks),
            "missing_ranks": list(self.missing_ranks),
            "complete": self.complete,
            "point_labels": list(self.point_labels),
            "warnings": list(self.warnings),
            "runs": {str(rank): run.descriptor() for rank, run in self.runs.items()},
        }

    def summary(self) -> "TensorRunGroupSummary":
        rows = []
        for rank, run in self.runs.items():
            for point in run.points:
                full_count = sum(item.payload == "full" for item in point.observations)
                rows.append(
                    TensorRankPointSummary(
                        rank=rank,
                        point_index=point.index,
                        point_label=point.label,
                        observation_count=len(point.observations),
                        full_payload_count=full_count,
                        summary_payload_count=len(point.observations) - full_count,
                    )
                )
        return TensorRunGroupSummary(
            run_group=self,
            rank_points=tuple(rows),
            warnings=self.warnings,
        )

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        cache_tensors: bool = False,
    ) -> "TensorRunGroup":
        """Load every direct child bundle directory as one tensor run group.

        Scans ``root`` for direct child directories that contain a
        ``manifest.json``, loads each as a ``TensorRun`` (forwarding
        ``cache_tensors``), and validates them through ``from_runs``.
        Raises ``TensorBundleError`` when ``root`` is not a directory or
        holds no bundles.
        """

        group_root = Path(root).resolve()
        if not group_root.is_dir():
            raise TensorBundleError(
                f"tensor run group directory does not exist: {group_root}"
            )
        bundles = tuple(
            sorted(
                path
                for path in group_root.iterdir()
                if path.is_dir() and (path / "manifest.json").is_file()
            )
        )
        if not bundles:
            raise TensorBundleError(
                f"tensor run group {group_root} has no direct child bundles"
            )
        return cls.from_runs(
            (TensorRun.load(path, cache_tensors=cache_tensors) for path in bundles),
            root=group_root,
        )

    @classmethod
    def from_runs(
        cls,
        runs: Iterable[TensorRun],
        *,
        root: str | Path | None = None,
    ) -> "TensorRunGroup":
        """Build a validated rank-indexed group from per-rank tensor runs.

        Raises ``TensorBundleError`` for an empty iterable, a run without a
        rank, a duplicate rank, differing run names or execution modes, or
        point labels that diverge from the longest rank's sequence other
        than as a strict prefix. An incomplete (crashed) rank whose labels
        are a strict prefix is accepted with a warning.
        """

        materialized = tuple(runs)
        if not materialized:
            raise TensorBundleError("tensor run group requires at least one run")
        by_rank: dict[int, TensorRun] = {}
        for run in materialized:
            if run.rank is None:
                raise TensorBundleError(f"tensor run {run.name!r} has no rank")
            if run.rank in by_rank:
                raise TensorBundleError(
                    f"tensor run group contains duplicate rank {run.rank}"
                )
            by_rank[run.rank] = run
        names = {run.name for run in materialized}
        executions = {run.execution for run in materialized}
        if len(names) != 1:
            raise TensorBundleError("tensor run group names differ across ranks")
        if len(executions) != 1:
            raise TensorBundleError("tensor execution modes differ across ranks")
        # The canonical point-label sequence is the longest across ranks. An
        # incomplete rank whose labels are a strict prefix of it (a crashed
        # rank) is accepted and handled by the incomplete-bundle path below.
        point_labels = max(
            (tuple(point.label for point in run.points) for run in materialized),
            key=len,
        )
        for run in materialized:
            labels = tuple(point.label for point in run.points)
            if labels == point_labels:
                continue
            if run.complete or labels != point_labels[: len(labels)]:
                raise TensorBundleError(
                    "tensor point label sequences differ across ranks"
                )
        warnings: list[str] = []
        group_ids = {run.group_id for run in materialized if run.group_id is not None}
        if len(group_ids) > 1:
            raise TensorBundleError("tensor group IDs differ across ranks")
        group_id = next(iter(group_ids), None)
        missing_group_ids = sorted(
            run.rank for run in materialized if run.group_id is None
        )
        if missing_group_ids:
            warnings.append(
                "group identity is unavailable for ranks "
                + ", ".join(str(rank) for rank in missing_group_ids)
            )

        world_sizes = {
            run.world_size for run in materialized if run.world_size is not None
        }
        if len(world_sizes) > 1:
            raise TensorBundleError("tensor world sizes differ across ranks")
        world_size = next(iter(world_sizes), None)
        missing_world_sizes = sorted(
            run.rank for run in materialized if run.world_size is None
        )
        if missing_world_sizes:
            warnings.append(
                "world size is unavailable for ranks "
                + ", ".join(str(rank) for rank in missing_world_sizes)
            )
        if world_size is not None:
            invalid = sorted(rank for rank in by_rank if rank >= world_size)
            if invalid:
                raise TensorBundleError(
                    f"ranks {invalid} are outside declared world size {world_size}"
                )
            absent = sorted(set(range(world_size)) - set(by_rank))
            if absent:
                warnings.append(
                    "tensor run group is missing ranks "
                    + ", ".join(str(rank) for rank in absent)
                )

        incomplete = sorted(run.rank for run in materialized if not run.complete)
        if incomplete:
            warnings.append(
                "tensor bundles are incomplete for ranks "
                + ", ".join(str(rank) for rank in incomplete)
            )
        missing_provenance = sorted(
            run.rank for run in materialized if not run.provenance
        )
        if missing_provenance:
            warnings.append(
                "runtime provenance is unavailable for ranks "
                + ", ".join(str(rank) for rank in missing_provenance)
            )
        provenance_signatures = {
            json_signature(comparable_provenance(run.provenance))
            for run in materialized
            if run.provenance
        }
        if len(provenance_signatures) > 1:
            warnings.append("runtime provenance differs across ranks")
        metadata_signatures = {json_signature(run.run_metadata) for run in materialized}
        if len(metadata_signatures) > 1:
            warnings.append("user run metadata differs across ranks")

        first = materialized[0]
        return cls(
            name=first.name,
            execution=first.execution,
            runs=MappingProxyType(dict(sorted(by_rank.items()))),
            point_labels=point_labels,
            group_id=group_id,
            world_size=world_size,
            warnings=tuple(dict.fromkeys(warnings)),
            root=Path(root).resolve() if root is not None else None,
        )


@dataclass(frozen=True)
class TensorRunGroupSummary:
    """Per-rank point inventory for one validated tensor run group."""

    run_group: TensorRunGroup
    rank_points: tuple[TensorRankPointSummary, ...]
    warnings: tuple[str, ...] = ()

    def to_text(self) -> str:
        lines = [
            f"Tensor run group summary {self.run_group.name!r} "
            f"ranks={list(self.run_group.ranks)}"
        ]
        lines.extend(f"  warning: {item}" for item in self.warnings)
        for row in self.rank_points:
            lines.append(
                f"  rank {row.rank} [{row.point_index}] {row.point_label}: "
                f"observations={row.observation_count}, "
                f"full={row.full_payload_count}, summary={row.summary_payload_count}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "run-group-summary",
            "run_group": self.run_group.descriptor(),
            "warnings": list(self.warnings),
            "rank_points": [item.to_dict() for item in self.rank_points],
        }

    def to_html(self) -> str:
        return _text_html(self.to_text())

    def write(
        self, output_dir: str | Path, *, overwrite: bool = False
    ) -> dict[str, Path]:
        return _write_group_report(
            output_dir,
            text=self.to_text(),
            payload=self.to_dict(),
            html=self.to_html(),
            rows=[item.to_dict() for item in self.rank_points],
            csv_name="rank_points",
            overwrite=overwrite,
        )


@dataclass(frozen=True)
class TensorRankRunComparison:
    """One rank-aligned tensor run comparison."""

    rank: int
    comparison: TensorRunComparison

    @property
    def status(self) -> ComparisonStatus:
        return self.comparison.status

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "status": self.status,
            "comparison": self.comparison.to_dict(),
        }


@dataclass(frozen=True)
class TensorRunGroupComparison:
    """Rank-aligned comparison between two tensor run groups."""

    reference_group: TensorRunGroup
    candidate_group: TensorRunGroup
    rank_comparisons: tuple[TensorRankRunComparison, ...]
    warnings: tuple[str, ...] = ()

    @property
    def reference_only_ranks(self) -> tuple[int, ...]:
        return tuple(
            sorted(set(self.reference_group.runs) - set(self.candidate_group.runs))
        )

    @property
    def candidate_only_ranks(self) -> tuple[int, ...]:
        return tuple(
            sorted(set(self.candidate_group.runs) - set(self.reference_group.runs))
        )

    @property
    def status(self) -> ComparisonStatus:
        statuses = {item.status for item in self.rank_comparisons}
        if "mismatch" in statuses:
            return "mismatch"
        if (
            "inconclusive" in statuses
            or self.reference_only_ranks
            or self.candidate_only_ranks
            or not self.reference_group.complete
            or not self.candidate_group.complete
        ):
            return "inconclusive"
        return "match"

    @property
    def conclusive(self) -> bool:
        return self.status != "inconclusive"

    @property
    def ok(self) -> bool:
        return self.status == "match"

    def assert_ok(self) -> None:
        if not self.ok:
            raise TensorComparisonError(f"tensor run group comparison is {self.status}")

    def to_text(self, *, include_unchanged: bool = True) -> str:
        lines = [
            f"Tensor run group comparison {self.reference_group.name!r} vs "
            f"{self.candidate_group.name!r}: {self.status}"
        ]
        lines.extend(f"  warning: {item}" for item in self.warnings)
        selected = [
            item
            for item in self.rank_comparisons
            if include_unchanged or item.status != "match"
        ]
        if not selected:
            lines.append("  no changed ranks")
        for item in selected:
            lines.append(f"  rank {item.rank}: {item.status}")
            nested = item.comparison.to_text(
                include_unchanged=include_unchanged
            ).splitlines()[1:]
            lines.extend(f"    {line}" for line in nested)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "run-group-comparison",
            "status": self.status,
            "conclusive": self.conclusive,
            "reference_group": self.reference_group.descriptor(),
            "candidate_group": self.candidate_group.descriptor(),
            "warnings": list(self.warnings),
            "reference_only_ranks": list(self.reference_only_ranks),
            "candidate_only_ranks": list(self.candidate_only_ranks),
            "rank_comparisons": [item.to_dict() for item in self.rank_comparisons],
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        return _text_html(self.to_text(include_unchanged=include_unchanged))

    def write(
        self,
        output_dir: str | Path,
        *,
        include_unchanged: bool = True,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        rows = [
            {"rank": item.rank, "status": item.status}
            for item in self.rank_comparisons
            if include_unchanged or item.status != "match"
        ]
        return _write_group_report(
            output_dir,
            text=self.to_text(include_unchanged=include_unchanged),
            payload=self.to_dict(),
            html=self.to_html(include_unchanged=include_unchanged),
            rows=rows,
            csv_name="rank_comparisons",
            overwrite=overwrite,
        )


def compare_run_groups(
    reference: TensorRunGroup,
    candidate: TensorRunGroup,
    *,
    point_mapping: Mapping[str, str] | None = None,
    options: TensorComparisonOptions | None = None,
) -> TensorRunGroupComparison:
    """Compare two run groups rank by rank over their common ranks.

    Raises ``TensorBundleError`` when the groups share no ranks; missing
    ranks, unknown world size, or incomplete bundles make the group verdict
    ``inconclusive``.
    """

    common_ranks = tuple(sorted(set(reference.runs) & set(candidate.runs)))
    if not common_ranks:
        raise TensorBundleError("tensor run groups have no common ranks")
    warnings = [
        *(f"reference: {item}" for item in reference.warnings),
        *(f"candidate: {item}" for item in candidate.warnings),
    ]
    missing_candidate = sorted(set(reference.runs) - set(candidate.runs))
    missing_reference = sorted(set(candidate.runs) - set(reference.runs))
    if missing_candidate:
        warnings.append(f"candidate group is missing ranks {missing_candidate}")
    if missing_reference:
        warnings.append(f"reference group is missing ranks {missing_reference}")
    comparisons = tuple(
        TensorRankRunComparison(
            rank=rank,
            comparison=compare_runs(
                reference[rank],
                candidate[rank],
                point_mapping=point_mapping,
                options=options,
            ),
        )
        for rank in common_ranks
    )
    return TensorRunGroupComparison(
        reference_group=reference,
        candidate_group=candidate,
        rank_comparisons=comparisons,
        warnings=tuple(warnings),
    )


def _text_html(text: str) -> str:
    return "<!doctype html><html><body><pre>" + escape(text) + "</pre></body></html>\n"


def _write_group_report(
    output_dir: str | Path,
    *,
    text: str,
    payload: Mapping[str, object],
    html: str,
    rows: list[dict[str, object]],
    csv_name: str,
    overwrite: bool,
) -> dict[str, Path]:
    root = prepare_output_dir(output_dir, overwrite=overwrite)
    text_path = root / "report.txt"
    json_path = root / "report.json"
    csv_path = root / f"{csv_name}.csv"
    html_path = root / "report.html"
    atomic_write_text(text_path, text + "\n")
    atomic_write_json(json_path, payload)
    atomic_write_csv(csv_path, rows)
    atomic_write_text(html_path, html)
    return {
        "text": text_path,
        "json": json_path,
        csv_name: csv_path,
        "html": html_path,
    }
