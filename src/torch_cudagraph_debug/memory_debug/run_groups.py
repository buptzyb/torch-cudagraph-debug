"""Multi-rank memory-run validation and per-rank aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from .._validation import comparable_provenance, json_signature
from ._pool_identity import MemoryPoolKey
from .aggregation import ALLOCATOR_SCOPES, summarize_devices
from .attribution import MemoryAttributionOptions
from .comparison import compare_phases
from .comparison_models import (
    MemoryAllocatorScopePhaseDecomposition,
    MemoryDevicePhaseDecomposition,
    MemoryPoolPhaseDecomposition,
    PhaseMetric,
)
from .errors import MemoryBundleError
from .recording import MemoryRun
from .reports import MemoryRunGroupPhaseComparison, MemoryRunGroupSummary
from .stats import (
    MEMORY_STAT_METRICS,
    AllocatorScope,
    DeviceMemorySample,
    MemoryStatMetric,
    MemoryStats,
)

GROUP_MEMORY_METRICS: tuple[MemoryStatMetric, ...] = MEMORY_STAT_METRICS
PHASE_COMPONENTS = (
    "start_gap_bytes",
    "baseline_change_bytes",
    "candidate_change_bytes",
    "change_gap_bytes",
    "end_gap_bytes",
)


@dataclass(frozen=True)
class MemoryRankPointState:
    """Absolute allocator state for one rank, point, and scope."""

    rank: int
    run_id: str
    point_index: int
    point_label: str
    scope: AllocatorScope
    stats: MemoryStats

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "run_id": self.run_id,
            "point_index": self.point_index,
            "point_label": self.point_label,
            "scope": self.scope,
            **self.stats.to_dict(),
        }


@dataclass(frozen=True)
class MemoryRankDevicePointState:
    """Device-wide CUDA Runtime state for one rank and point."""

    rank: int
    run_id: str
    point_index: int
    point_label: str
    device_index: int
    sample: DeviceMemorySample
    allocator_reserved_bytes: int

    @property
    def cuda_allocator_residual_bytes(self) -> int:
        return self.sample.used_bytes - self.allocator_reserved_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "run_id": self.run_id,
            "point_index": self.point_index,
            "point_label": self.point_label,
            "device_index": self.device_index,
            **self.sample.to_dict(),
            "allocator_reserved_bytes": self.allocator_reserved_bytes,
            "cuda_allocator_residual_bytes": self.cuda_allocator_residual_bytes,
        }


@dataclass(frozen=True)
class MemoryRankPointAggregate:
    """Cross-rank extrema for one point/scope metric."""

    point_index: int
    point_label: str
    scope: AllocatorScope
    metric: MemoryStatMetric
    rank_count: int
    min_value: int
    min_rank: int
    max_value: int
    max_rank: int

    @property
    def worst_rank(self) -> int:
        return self.max_rank

    @property
    def spread_value(self) -> int:
        return self.max_value - self.min_value

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_label": self.point_label,
            "scope": self.scope,
            "metric": self.metric,
            "rank_count": self.rank_count,
            "min_value": self.min_value,
            "min_rank": self.min_rank,
            "max_value": self.max_value,
            "max_rank": self.max_rank,
            "worst_rank": self.worst_rank,
            "spread_value": self.spread_value,
        }


@dataclass(frozen=True)
class MemoryRankPhaseDecomposition:
    """Allocator-scope phase equation for one rank."""

    rank: int
    decomposition: MemoryAllocatorScopePhaseDecomposition

    @property
    def changed(self) -> bool:
        return self.decomposition.changed

    def to_dict(self) -> dict[str, object]:
        return {"rank": self.rank, **self.decomposition.to_dict()}


@dataclass(frozen=True)
class MemoryRankPoolPhaseDecomposition:
    """Mapped-pool phase equation for one rank."""

    rank: int
    decomposition: MemoryPoolPhaseDecomposition

    @property
    def changed(self) -> bool:
        return self.decomposition.changed

    def to_dict(self) -> dict[str, object]:
        return {"rank": self.rank, **self.decomposition.to_dict()}


@dataclass(frozen=True)
class MemoryRankDevicePhaseDecomposition:
    """Device-wide phase equation for one rank."""

    rank: int
    decomposition: MemoryDevicePhaseDecomposition

    @property
    def changed(self) -> bool:
        return self.decomposition.changed

    def to_dict(self) -> dict[str, object]:
        return {"rank": self.rank, **self.decomposition.to_dict()}


@dataclass(frozen=True)
class MemoryMetricExtrema:
    """Per-rank extrema and spread for one phase component."""

    min_bytes: int
    min_rank: int
    max_bytes: int
    max_rank: int

    @property
    def spread_bytes(self) -> int:
        return self.max_bytes - self.min_bytes

    def to_dict(self, prefix: str) -> dict[str, object]:
        return {
            f"{prefix}_min_bytes": self.min_bytes,
            f"{prefix}_min_rank": self.min_rank,
            f"{prefix}_max_bytes": self.max_bytes,
            f"{prefix}_max_rank": self.max_rank,
            f"{prefix}_spread_bytes": self.spread_bytes,
        }


@dataclass(frozen=True)
class MemoryRunGroupPhaseAggregate:
    """Cross-rank extrema for one allocator-scope phase equation."""

    scope: AllocatorScope
    metric: PhaseMetric
    rank_count: int
    identity_holds: bool
    start_gap: MemoryMetricExtrema
    baseline_change: MemoryMetricExtrema
    candidate_change: MemoryMetricExtrema
    change_gap: MemoryMetricExtrema
    end_gap: MemoryMetricExtrema

    @property
    def changed(self) -> bool:
        return any(
            extrema.min_bytes or extrema.max_bytes
            for extrema in (
                self.start_gap,
                self.baseline_change,
                self.candidate_change,
                self.change_gap,
                self.end_gap,
            )
        )

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "scope": self.scope,
            "metric": self.metric,
            "rank_count": self.rank_count,
            "identity_holds": self.identity_holds,
        }
        for name in (
            "start_gap",
            "baseline_change",
            "candidate_change",
            "change_gap",
            "end_gap",
        ):
            row.update(getattr(self, name).to_dict(name))
        return row


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

    @property
    def missing_ranks(self) -> tuple[int, ...]:
        if self.world_size is None:
            return ()
        return tuple(sorted(set(range(self.world_size)) - set(self.runs)))

    @property
    def complete(self) -> bool:
        """True only when the world size is known, every rank is present,
        and every run finished with ``complete=True``."""

        return (
            self.world_size is not None
            and not self.missing_ranks
            and all(run.complete for run in self.runs.values())
        )

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
            "missing_ranks": list(self.missing_ranks),
            "complete": self.complete,
            "point_labels": list(self.point_labels),
            "warnings": list(self.warnings),
            "runs": {str(rank): run.descriptor() for rank, run in self.runs.items()},
        }

    def summary(self) -> MemoryRunGroupSummary:
        """Summarize point states and cross-rank skew without summing GPUs."""

        rank_rows = _rank_point_rows(self)
        point_warnings = [
            f"rank {rank} point [{point.index}] {point.label}: {warning}"
            for rank, run in self.runs.items()
            for point in run.points
            for warning in point.warnings
        ]
        return MemoryRunGroupSummary(
            run_group=self,
            rank_points=rank_rows,
            rank_devices=_rank_device_point_rows(self),
            point_aggregates=_aggregate_point_rows(rank_rows),
            warnings=tuple(dict.fromkeys((*self.warnings, *point_warnings))),
        )

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        cache_snapshots: bool = False,
    ) -> "MemoryRunGroup":
        """Load direct bundles without retaining allocator payloads by default."""

        group_root = Path(root).resolve()
        if not group_root.is_dir():
            raise MemoryBundleError(
                f"memory run group directory does not exist: {group_root}"
            )
        bundles = tuple(
            sorted(
                path
                for path in group_root.iterdir()
                if path.is_dir() and (path / "manifest.json").is_file()
            )
        )
        if not bundles:
            raise MemoryBundleError(
                f"memory run group {group_root} has no direct child memory bundles"
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

        # The canonical point-label sequence is the longest across ranks. An
        # incomplete rank whose labels are a strict prefix of it (a crashed
        # rank) is accepted and surfaced by the incomplete-bundle warning
        # below; a complete rank with fewer points or any non-prefix sequence
        # is a genuine structural difference.
        point_labels = max(
            (tuple(point.label for point in run.points) for run in materialized),
            key=len,
        )
        for run in materialized:
            labels = tuple(point.label for point in run.points)
            if labels == point_labels:
                continue
            if run.complete or labels != point_labels[: len(labels)]:
                details = "; ".join(
                    f"rank {rank}: {tuple(point.label for point in item.points)!r}"
                    for rank, item in sorted(by_rank.items())
                )
                raise MemoryBundleError(
                    f"memory point label sequences differ across ranks: {details}"
                )

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
            json_signature(comparable_provenance(run.provenance))
            for run in materialized
            if run.provenance
        }
        if len(provenance_signatures) > 1:
            warnings.append("runtime provenance differs across ranks")

        metadata_signatures = {json_signature(run.run_metadata) for run in materialized}
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


def compare_run_group_phases(
    baseline: MemoryRunGroup,
    candidate: MemoryRunGroup,
    *,
    baseline_start: str | int,
    baseline_end: str | int,
    candidate_start: str | int,
    candidate_end: str | int,
    pool_mappings: Mapping[int, Mapping[MemoryPoolKey, MemoryPoolKey]] | None = None,
    device_mappings: Mapping[int, Mapping[int, int]] | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemoryRunGroupPhaseComparison:
    """Compare four-point phase equations rank by rank and report skew."""

    common_ranks = tuple(sorted(set(baseline.runs) & set(candidate.runs)))
    if not common_ranks:
        raise MemoryBundleError(
            "baseline and candidate memory groups have no common ranks"
        )
    unknown_mapping_ranks = set(pool_mappings or {}) - set(common_ranks)
    if unknown_mapping_ranks:
        raise ValueError(
            "pool_mappings contains ranks outside the comparison: "
            + ", ".join(str(rank) for rank in sorted(unknown_mapping_ranks))
        )

    unknown_device_mapping_ranks = set(device_mappings or {}) - set(common_ranks)
    if unknown_device_mapping_ranks:
        raise ValueError(
            "device_mappings contains ranks outside the comparison: "
            + ", ".join(str(rank) for rank in sorted(unknown_device_mapping_ranks))
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

    rank_comparisons = {}
    rank_decomposition: list[MemoryRankPhaseDecomposition] = []
    rank_pool_decomposition: list[MemoryRankPoolPhaseDecomposition] = []
    rank_device_decomposition: list[MemoryRankDevicePhaseDecomposition] = []
    options = attribution or MemoryAttributionOptions()
    for rank in common_ranks:
        phase = compare_phases(
            baseline[rank].between(baseline_start, baseline_end),
            candidate[rank].between(candidate_start, candidate_end),
            pool_mapping=(pool_mappings or {}).get(rank),
            device_mapping=(device_mappings or {}).get(rank),
            attribution=options,
        )
        rank_comparisons[rank] = phase
        rank_decomposition.extend(
            MemoryRankPhaseDecomposition(rank, item)
            for item in phase.allocator_scope_decomposition
        )
        rank_pool_decomposition.extend(
            MemoryRankPoolPhaseDecomposition(rank, item)
            for item in phase.pool_decomposition
        )
        rank_device_decomposition.extend(
            MemoryRankDevicePhaseDecomposition(rank, item)
            for item in phase.device_decomposition
        )
        warnings.extend(f"rank {rank}: {warning}" for warning in phase.warnings)

    frozen_rank_decomposition = tuple(rank_decomposition)
    return MemoryRunGroupPhaseComparison(
        baseline_group=baseline,
        candidate_group=candidate,
        rank_comparisons=MappingProxyType(rank_comparisons),
        rank_decomposition=frozen_rank_decomposition,
        rank_pool_decomposition=tuple(rank_pool_decomposition),
        rank_device_decomposition=tuple(rank_device_decomposition),
        phase_aggregates=_aggregate_phase_rows(frozen_rank_decomposition),
        warnings=tuple(dict.fromkeys(warnings)),
        display_stack_depth=options.display.stack_depth,
        display_limit=options.display.limit,
    )


def _rank_point_rows(group: MemoryRunGroup) -> tuple[MemoryRankPointState, ...]:
    rows = []
    for rank, run in group.runs.items():
        for point in run.points:
            for scope in ALLOCATOR_SCOPES:
                rows.append(
                    MemoryRankPointState(
                        rank=rank,
                        run_id=run.run_id,
                        point_index=point.index,
                        point_label=point.label,
                        scope=scope,
                        stats=point.allocator_scope_stats[scope],
                    )
                )
    return tuple(rows)


def _rank_device_point_rows(
    group: MemoryRunGroup,
) -> tuple[MemoryRankDevicePointState, ...]:
    rows = []
    empty = MemoryStats()
    for rank, run in group.runs.items():
        for point in run.points:
            allocator_by_device = summarize_devices(point.pool_stats)
            for device, sample in point.device_memory.items():
                rows.append(
                    MemoryRankDevicePointState(
                        rank=rank,
                        run_id=run.run_id,
                        point_index=point.index,
                        point_label=point.label,
                        device_index=device,
                        sample=sample,
                        allocator_reserved_bytes=allocator_by_device.get(
                            device, empty
                        ).reserved_bytes,
                    )
                )
    return tuple(rows)


def _aggregate_point_rows(
    rows: Iterable[MemoryRankPointState],
) -> tuple[MemoryRankPointAggregate, ...]:
    grouped: dict[
        tuple[int, str, AllocatorScope],
        list[MemoryRankPointState],
    ] = defaultdict(list)
    for row in rows:
        grouped[(row.point_index, row.point_label, row.scope)].append(row)

    result = []
    for (point_index, point_label, scope), members in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            ALLOCATOR_SCOPES.index(item[0][2]),
        ),
    ):
        for metric in GROUP_MEMORY_METRICS:
            values = sorted(
                (int(getattr(item.stats, metric)), item.rank) for item in members
            )
            minimum, min_rank = values[0]
            maximum, max_rank = values[-1]
            result.append(
                MemoryRankPointAggregate(
                    point_index=point_index,
                    point_label=point_label,
                    scope=scope,
                    metric=metric,
                    rank_count=len(values),
                    min_value=minimum,
                    min_rank=min_rank,
                    max_value=maximum,
                    max_rank=max_rank,
                )
            )
    return tuple(result)


def _aggregate_phase_rows(
    rows: Iterable[MemoryRankPhaseDecomposition],
) -> tuple[MemoryRunGroupPhaseAggregate, ...]:
    grouped: dict[
        tuple[AllocatorScope, PhaseMetric],
        list[MemoryRankPhaseDecomposition],
    ] = defaultdict(list)
    for row in rows:
        grouped[(row.decomposition.scope, row.decomposition.components.metric)].append(
            row
        )

    result = []
    scope_order = {scope: index for index, scope in enumerate(ALLOCATOR_SCOPES)}
    for (scope, metric), members in sorted(
        grouped.items(),
        key=lambda item: (scope_order[item[0][0]], item[0][1]),
    ):
        extrema: dict[str, MemoryMetricExtrema] = {}
        for component in (
            "start_gap_bytes",
            "baseline_change_bytes",
            "candidate_change_bytes",
            "change_gap_bytes",
            "end_gap_bytes",
        ):
            values = sorted(
                (
                    int(getattr(item.decomposition.components, component)),
                    item.rank,
                )
                for item in members
            )
            minimum, min_rank = values[0]
            maximum, max_rank = values[-1]
            extrema[component.removesuffix("_bytes")] = MemoryMetricExtrema(
                min_bytes=minimum,
                min_rank=min_rank,
                max_bytes=maximum,
                max_rank=max_rank,
            )
        result.append(
            MemoryRunGroupPhaseAggregate(
                scope=scope,
                metric=metric,
                rank_count=len(members),
                identity_holds=all(
                    item.decomposition.components.identity_holds for item in members
                ),
                start_gap=extrema["start_gap"],
                baseline_change=extrema["baseline_change"],
                candidate_change=extrema["candidate_change"],
                change_gap=extrema["change_gap"],
                end_gap=extrema["end_gap"],
            )
        )
    return tuple(result)
