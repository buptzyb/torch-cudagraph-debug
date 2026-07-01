"""Multi-rank memory-run validation and per-rank aggregation."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .core import AttributionOptions, MemoryRun, compare_phases
from .errors import MemoryBundleError
from .reports import GroupPhaseComparison, MemoryGroupSummary
from .totals import ALLOCATOR_SCOPES


GROUP_MEMORY_METRICS = (
    "reserved_bytes",
    "allocated_bytes",
    "active_bytes",
    "inactive_bytes",
    "requested_bytes",
    "fragmentation_bytes",
)
PHASE_COMPONENTS = (
    "start_delta_bytes",
    "baseline_growth_bytes",
    "candidate_growth_bytes",
    "growth_delta_bytes",
    "end_delta_bytes",
)


@dataclass(frozen=True)
class MemoryRunGroup:
    """Validated rank-indexed collection of compatible memory runs."""

    name: str
    runs: Mapping[int, MemoryRun]
    point_labels: tuple[str, ...]
    group_id: str | None
    world_size: int | None
    warnings: tuple[str, ...] = ()
    root: Path | None = field(default=None, compare=False)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(self.runs)

    def __len__(self) -> int:
        return len(self.runs)

    def __getitem__(self, rank: int) -> MemoryRun:
        return self.runs[rank]

    def descriptor(self) -> dict[str, object]:
        return {
            "name": self.name,
            "group_id": self.group_id,
            "world_size": self.world_size,
            "ranks": list(self.ranks),
            "point_labels": list(self.point_labels),
            "warnings": list(self.warnings),
            "runs": {str(rank): run.descriptor() for rank, run in self.runs.items()},
        }

    def summary(self) -> MemoryGroupSummary:
        """Summarize point states and cross-rank skew without summing GPUs."""

        rank_rows = _rank_point_rows(self)
        return MemoryGroupSummary(
            group=self,
            rank_points=rank_rows,
            point_summary=_aggregate_point_rows(rank_rows),
            warnings=self.warnings,
        )

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        cache_snapshots: bool = False,
    ) -> "MemoryRunGroup":
        """Load direct bundles without retaining raw snapshots by default."""

        group_root = Path(root).resolve()
        if not group_root.is_dir():
            raise MemoryBundleError(
                f"memory run group directory does not exist: {group_root}"
            )
        bundles = tuple(
            sorted(path for path in group_root.glob("*.tcgd-memory") if path.is_dir())
        )
        if not bundles:
            raise MemoryBundleError(
                f"memory run group {group_root} has no direct *.tcgd-memory bundles"
            )
        return cls.from_runs(
            (
                MemoryRun.load(bundle, cache_snapshots=cache_snapshots)
                for bundle in bundles
            ),
            root=group_root,
        )

    @classmethod
    def from_runs(
        cls,
        runs: Iterable[MemoryRun],
        *,
        root: str | Path | None = None,
    ) -> "MemoryRunGroup":
        materialized = tuple(runs)
        if not materialized:
            raise MemoryBundleError("memory run group requires at least one run")

        by_rank: dict[int, MemoryRun] = {}
        for run in materialized:
            if run.rank is None:
                raise MemoryBundleError(
                    f"memory run {run.name!r} ({run.run_id}) has no rank"
                )
            if run.rank < 0:
                raise MemoryBundleError(
                    f"memory run rank must be non-negative: {run.rank}"
                )
            if run.rank in by_rank:
                raise MemoryBundleError(
                    f"memory run group contains duplicate rank {run.rank}"
                )
            by_rank[run.rank] = run

        names = {run.name for run in materialized}
        if len(names) != 1:
            raise MemoryBundleError(
                "memory run group names differ across ranks: "
                + ", ".join(sorted(repr(name) for name in names))
            )
        name = materialized[0].name

        point_sequences = {
            tuple(point.label for point in run.points) for run in materialized
        }
        if len(point_sequences) != 1:
            details = "; ".join(
                f"rank {rank}: {tuple(point.label for point in run.points)!r}"
                for rank, run in sorted(by_rank.items())
            )
            raise MemoryBundleError(
                f"memory point label sequences differ across ranks: {details}"
            )
        point_labels = tuple(point.label for point in materialized[0].points)

        warnings: list[str] = []
        group_ids = {run.group_id for run in materialized if run.group_id is not None}
        if len(group_ids) > 1:
            raise MemoryBundleError(
                "memory group IDs differ across ranks: "
                + ", ".join(sorted(repr(value) for value in group_ids))
            )
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
            raise MemoryBundleError(
                "memory world sizes differ across ranks: "
                + ", ".join(str(value) for value in sorted(world_sizes))
            )
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
                raise MemoryBundleError(
                    f"ranks {invalid} are outside declared world size {world_size}"
                )
            absent = sorted(set(range(world_size)) - set(by_rank))
            if absent:
                warnings.append(
                    "memory run group is missing ranks "
                    + ", ".join(str(rank) for rank in absent)
                )

        incomplete = sorted(run.rank for run in materialized if not run.complete)
        if incomplete:
            warnings.append(
                "memory bundles are incomplete for ranks "
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
            _signature(_comparable_provenance(run.provenance))
            for run in materialized
            if run.provenance
        }
        if len(provenance_signatures) > 1:
            warnings.append("runtime provenance differs across ranks")

        metadata_signatures = {
            _signature(dict(run.run_metadata)) for run in materialized
        }
        if len(metadata_signatures) > 1:
            warnings.append("user run metadata differs across ranks")

        ordered = MappingProxyType(dict(sorted(by_rank.items())))
        return cls(
            name=name,
            runs=ordered,
            point_labels=point_labels,
            group_id=group_id,
            world_size=world_size,
            warnings=tuple(dict.fromkeys(warnings)),
            root=Path(root).resolve() if root is not None else None,
        )


def compare_group_phases(
    baseline: MemoryRunGroup,
    candidate: MemoryRunGroup,
    *,
    baseline_start: str | int,
    baseline_end: str | int,
    candidate_start: str | int,
    candidate_end: str | int,
    attribution: AttributionOptions | None = None,
) -> GroupPhaseComparison:
    """Compare four-point phase equations rank by rank and report skew."""

    common_ranks = tuple(sorted(set(baseline.runs) & set(candidate.runs)))
    if not common_ranks:
        raise MemoryBundleError(
            "baseline and candidate memory groups have no common ranks"
        )

    warnings = [
        *(f"baseline: {warning}" for warning in baseline.warnings),
        *(f"candidate: {warning}" for warning in candidate.warnings),
    ]
    baseline_only = sorted(set(baseline.runs) - set(candidate.runs))
    candidate_only = sorted(set(candidate.runs) - set(baseline.runs))
    if baseline_only:
        warnings.append(
            "candidate group is missing baseline ranks "
            + ", ".join(str(rank) for rank in baseline_only)
        )
    if candidate_only:
        warnings.append(
            "baseline group is missing candidate ranks "
            + ", ".join(str(rank) for rank in candidate_only)
        )

    comparisons = {}
    rank_rows: list[dict[str, object]] = []
    options = attribution or AttributionOptions()
    for rank in common_ranks:
        phase = compare_phases(
            baseline[rank].between(baseline_start, baseline_end),
            candidate[rank].between(candidate_start, candidate_end),
            attribution=options,
        )
        comparisons[rank] = phase
        rank_rows.extend({"rank": rank, **row} for row in phase.total_decomposition)
        warnings.extend(f"rank {rank}: {warning}" for warning in phase.warnings)

    immutable_comparisons = MappingProxyType(comparisons)
    frozen_rows = tuple(rank_rows)
    return GroupPhaseComparison(
        baseline_group=baseline,
        candidate_group=candidate,
        rank_comparisons=immutable_comparisons,
        rank_phase=frozen_rows,
        phase_summary=_aggregate_phase_rows(frozen_rows),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _rank_point_rows(group: MemoryRunGroup) -> tuple[dict[str, object], ...]:
    rows = []
    for rank, run in group.runs.items():
        for point in run.points:
            for scope in ALLOCATOR_SCOPES:
                rows.append(
                    {
                        "rank": rank,
                        "run_id": run.run_id,
                        "point_index": point.index,
                        "point_label": point.label,
                        "scope": scope,
                        **point.totals[scope].to_dict(),
                    }
                )
    return tuple(rows)


def _aggregate_point_rows(
    rows: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[int, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (int(row["point_index"]), str(row["point_label"]), str(row["scope"]))
        ].append(row)

    result = []
    for (point_index, point_label, scope), members in sorted(grouped.items()):
        for metric in GROUP_MEMORY_METRICS:
            values = sorted((int(item[metric]), int(item["rank"])) for item in members)
            minimum, min_rank = values[0]
            maximum, max_rank = values[-1]
            result.append(
                {
                    "point_index": point_index,
                    "point_label": point_label,
                    "scope": scope,
                    "metric": metric,
                    "rank_count": len(values),
                    "min_value": minimum,
                    "min_rank": min_rank,
                    "max_value": maximum,
                    "max_rank": max_rank,
                    "worst_rank": max_rank,
                    "spread_value": maximum - minimum,
                }
            )
    return tuple(result)


def _aggregate_phase_rows(
    rows: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scope"]), str(row["metric"]))].append(row)

    result = []
    scope_order = {scope: index for index, scope in enumerate(ALLOCATOR_SCOPES)}
    for (scope, metric), members in sorted(
        grouped.items(), key=lambda item: (scope_order[item[0][0]], item[0][1])
    ):
        summary: dict[str, object] = {
            "scope": scope,
            "metric": metric,
            "rank_count": len(members),
            "identity_holds": all(bool(item["identity_holds"]) for item in members),
        }
        for component in PHASE_COMPONENTS:
            values = sorted(
                (int(item[component]), int(item["rank"])) for item in members
            )
            minimum, min_rank = values[0]
            maximum, max_rank = values[-1]
            prefix = component.removesuffix("_bytes")
            summary.update(
                {
                    f"{prefix}_min_bytes": minimum,
                    f"{prefix}_min_rank": min_rank,
                    f"{prefix}_max_bytes": maximum,
                    f"{prefix}_max_rank": max_rank,
                    f"{prefix}_spread_bytes": maximum - minimum,
                }
            )
        result.append(summary)
    return tuple(result)


def _comparable_provenance(value: Mapping[str, Any]) -> dict[str, object]:
    producer = value.get("producer")
    runtime = value.get("runtime")
    device = value.get("device")
    result: dict[str, object] = {}
    if isinstance(producer, Mapping):
        result["producer"] = dict(producer)
    if isinstance(runtime, Mapping):
        result["runtime"] = {
            key: runtime.get(key)
            for key in ("python", "platform", "torch", "cuda")
            if key in runtime
        }
    if isinstance(device, Mapping):
        result["device"] = {
            key: device.get(key)
            for key in ("name", "capability", "total_memory_bytes")
            if key in device
        }
    return result


def _signature(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
