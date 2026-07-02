"""Structured text, JSON, CSV, and HTML reports for memory debugging."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from .lifetimes import AllocationCohort
from .comparison_models import (
    MemoryAllocatorScopeComparison,
    MemoryObservationComparison,
    MemoryPoolComparison,
)
from .stacks import AllocationStackCoverage, AllocationStackDelta
from .events import AllocatorEventSummary
from .allocator_snapshot import (
    format_comparison,
    format_bytes,
    format_delta_bytes,
    pool_id_label,
    stream_label,
)

if TYPE_CHECKING:
    from .timeline import (
        MemoryAllocatorScopeTimelineEntry,
        MemoryObservationTimelineEntry,
        MemoryPoolTimelineEntry,
    )

REPORT_SCHEMA = "torch-cudagraph-debug/memory-report"


@dataclass(frozen=True)
class MemoryAllocationLifetimeAnalysis:
    """Point-by-point allocation cohorts from one Run or Probe interval."""

    source_kind: Literal["run", "probe"]
    source_id: str
    source_name: str
    start: Any
    end: Any
    active_at: Any | None
    born_between: tuple[Any, Any] | None
    cohorts: tuple[AllocationCohort, ...]
    history_requested: bool
    history_available: bool
    history_complete: bool
    total_instance_bytes: int
    attributed_instance_bytes: int
    warnings: tuple[str, ...] = ()

    def cohort_rows(self) -> list[dict[str, object]]:
        return [item.to_row() for item in self.cohorts]

    def point_rows(self) -> list[dict[str, object]]:
        return [
            item.to_row(cohort.cohort_id)
            for cohort in self.cohorts
            for item in cohort.points
        ]

    def size_rows(self) -> list[dict[str, object]]:
        return [
            item.to_row(cohort.cohort_id)
            for cohort in self.cohorts
            for item in cohort.size_histogram
        ]

    def release_rows(self) -> list[dict[str, object]]:
        return [
            item.to_row(cohort.cohort_id)
            for cohort in self.cohorts
            for item in cohort.releases
        ]

    def birth_rows(self) -> list[dict[str, object]]:
        return [
            item.to_row(cohort.cohort_id)
            for cohort in self.cohorts
            for item in cohort.births
        ]

    def summary_lines(self, *, indent: str = "") -> list[str]:
        lines = [f"{indent}top allocation cohorts:"]
        if not self.cohorts:
            lines.append(f"{indent}  no active allocation cohorts")
            return lines
        for item in self.cohorts:
            device = "unknown" if item.device is None else str(item.device)
            lines.append(
                f"{indent}  {item.cohort_id} device[{device}] "
                f"{pool_id_label(item.pool_id)} peak={format_bytes(item.peak_active_bytes)} "
                f"event_peak={format_bytes(item.peak_live_bytes)} "
                f"impact={format_bytes(item.impact_bytes)} "
                f"blocks={item.peak_block_count} at {item.stack_key}"
            )
            lines.append(
                f"{indent}    born exact={format_bytes(item.event_exact_birth_bytes)} "
                f"inferred={format_bytes(item.snapshot_inferred_birth_bytes)}; "
                f"released exact={format_bytes(item.event_exact_release_bytes)} "
                f"inferred={format_bytes(item.snapshot_inferred_release_bytes)}, "
                f"still active={format_bytes(item.still_active_bytes)}"
            )
        return lines

    def to_text(self, *, include_unchanged: bool = True) -> str:
        del include_unchanged
        if self.active_at is not None:
            selection = f"active_at={_state_label(self.active_at)!r}"
        elif self.born_between is not None:
            selection = (
                f"born_between=({_state_label(self.born_between[0])!r}, "
                f"{_state_label(self.born_between[1])!r}]"
            )
        else:
            selection = "all cohorts"
        lines = [
            f"Allocation cohort lifetimes {self.source_name!r}: "
            f"{_state_label(self.start)!r} -> {_state_label(self.end)!r} ({selection})"
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        lines.append(
            "  instance stack coverage: "
            f"{format_bytes(self.attributed_instance_bytes)} / "
            f"{format_bytes(self.total_instance_bytes)}"
        )
        if self.history_requested:
            lines.append(
                "  allocator history: "
                f"available={self.history_available}, complete={self.history_complete}"
            )
        lines.extend(self.summary_lines(indent="  "))
        for cohort in self.cohorts:
            lines.append(f"  {cohort.cohort_id} point states:")
            for point in cohort.points:
                lines.append(
                    "    "
                    f"[{point.point_index}] {point.point_label}: "
                    f"active={format_bytes(point.active_bytes)}, "
                    f"requested={format_bytes(point.requested_bytes)}, "
                    f"blocks={point.block_count}"
                )
            if cohort.size_histogram:
                sizes = ", ".join(
                    f"{format_bytes(item.size_bytes)} x {item.count}"
                    for item in cohort.size_histogram
                )
                lines.append(f"    sizes: {sizes}")
            for birth in cohort.births:
                lines.append(
                    "    birth "
                    f"{birth.start_label!r} -> {birth.end_label!r}: "
                    f"{format_bytes(birth.size_bytes)} in {birth.count} blocks "
                    f"[{birth.confidence}] at {birth.stack_key}"
                )
            for release in cohort.releases:
                lines.append(
                    "    release "
                    f"{release.start_label!r} -> {release.end_label!r}: "
                    f"{format_bytes(release.size_bytes)} in {release.count} blocks "
                    f"[{release.confidence}] at {release.stack_key}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "allocation-lifetime-analysis",
            "source": {
                "kind": self.source_kind,
                "id": self.source_id,
                "name": self.source_name,
            },
            "start": self.start.descriptor(),
            "end": self.end.descriptor(),
            "active_at": self.active_at.descriptor() if self.active_at else None,
            "born_between": (
                [item.descriptor() for item in self.born_between]
                if self.born_between is not None
                else None
            ),
            "history_requested": self.history_requested,
            "history_available": self.history_available,
            "history_complete": self.history_complete,
            "total_instance_bytes": self.total_instance_bytes,
            "attributed_instance_bytes": self.attributed_instance_bytes,
            "warnings": list(self.warnings),
            "cohorts": [item.to_dict() for item in self.cohorts],
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        del include_unchanged
        cohort_rows = self.cohort_rows()
        point_rows = self.point_rows()
        size_rows = self.size_rows()
        birth_rows = self.birth_rows()
        release_rows = self.release_rows()
        sections = [
            "<h2>Active Bytes by Cohort</h2>",
            _cohort_timeline_svg(point_rows),
            "<h2>Cohorts</h2>",
            _render_table(cohort_rows, "No allocation cohorts"),
            "<h2>Cohort Point States</h2>",
            _render_table(point_rows, "No cohort point states"),
            "<h2>Size Histograms</h2>",
            _render_table(size_rows, "No allocation sizes"),
        ]
        if birth_rows:
            sections.extend(
                [
                    "<h2>Birth Stacks</h2>",
                    _render_table(birth_rows, "No allocation births"),
                ]
            )
        if release_rows:
            sections.extend(
                [
                    "<h2>Release Stacks</h2>",
                    _render_table(release_rows, "No releases"),
                ]
            )
        return _html_document(
            f"Allocation cohort lifetimes for {self.source_name}",
            sections,
            self.warnings,
        )

    def write(
        self,
        output_dir: str | Path,
        *,
        include_unchanged: bool = True,
    ) -> dict[str, Path]:
        root = _prepare_output(output_dir)
        paths = _write_report_documents(
            root,
            text=self.to_text(include_unchanged=include_unchanged),
            payload=self.to_dict(),
            html=self.to_html(include_unchanged=include_unchanged),
        )
        paths.update(self.write_csv_files(root))
        return paths

    def write_csv_files(self, root: Path) -> dict[str, Path]:
        cohort_rows = self.cohort_rows()
        point_rows = self.point_rows()
        size_rows = self.size_rows()
        birth_rows = self.birth_rows()
        release_rows = self.release_rows()
        paths = {
            "cohorts": _write_csv(root / "cohorts.csv", cohort_rows),
            "cohort_points": _write_csv(root / "cohort_points.csv", point_rows),
            "size_histograms": _write_csv(root / "size_histograms.csv", size_rows),
        }
        if birth_rows:
            paths["birth_stacks"] = _write_csv(root / "birth_stacks.csv", birth_rows)
        if release_rows:
            paths["release_stacks"] = _write_csv(
                root / "release_stacks.csv", release_rows
            )
        return paths


@dataclass(frozen=True)
class _MemoryStateComparison:
    """Shared comparison behavior for Point and ProbeSnapshot states."""

    reference: Any
    candidate: Any
    _COMPARISON_KIND: ClassVar[str] = "state-comparison"
    allocator_scope_comparisons: tuple[MemoryAllocatorScopeComparison, ...]
    pool_comparisons: tuple[MemoryPoolComparison, ...]
    observation_comparisons: tuple[MemoryObservationComparison, ...]
    allocation_stack_comparisons: tuple[AllocationStackDelta, ...] = ()
    allocation_stack_observation_comparisons: tuple[AllocationStackDelta, ...] = ()
    reference_stack_coverage: AllocationStackCoverage | None = None
    candidate_stack_coverage: AllocationStackCoverage | None = None
    allocator_events: tuple[AllocatorEventSummary, ...] = ()
    allocation_lifetimes: MemoryAllocationLifetimeAnalysis | None = None
    events_available: bool = False
    events_complete: bool = True
    lifecycle_available: bool = False
    warnings: tuple[str, ...] = ()

    def allocator_scope_comparison_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.allocator_scope_comparisons
            if include_unchanged or item.changed
        ]

    def pool_comparison_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.pool_comparisons
            if include_unchanged or item.changed
        ]

    def observation_comparison_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.observation_comparisons
            if include_unchanged or item.changed
        ]

    def allocation_stack_comparison_rows(self) -> list[dict[str, object]]:
        rows = [
            {"scope": "pool", **item.to_row()}
            for item in self.allocation_stack_comparisons
        ]
        rows.extend(
            {"scope": "observation", **item.to_row()}
            for item in self.allocation_stack_observation_comparisons
        )
        return rows

    def event_rows(self) -> list[dict[str, object]]:
        return [
            item.to_row(
                reference_label=_state_label(self.reference),
                candidate_label=_state_label(self.candidate),
            )
            for item in self.allocator_events
        ]

    def to_text(self, *, include_unchanged: bool = True) -> str:
        reference_label = _state_label(self.reference)
        candidate_label = _state_label(self.candidate)
        lines = [
            f"Memory comparison {reference_label!r} -> {candidate_label!r} "
            f"({_state_scope(self.reference, self.candidate)})"
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        lines.append("  allocator totals:")
        selected_allocator_scopes = [
            item
            for item in self.allocator_scope_comparisons
            if include_unchanged or item.changed
        ]
        if not selected_allocator_scopes:
            lines.append("    no changed totals")
        for item in selected_allocator_scopes:
            lines.extend(_scope_text(item))
        lines.append("  pools:")
        selected_pool_comparisons = [
            item for item in self.pool_comparisons if include_unchanged or item.changed
        ]
        if not selected_pool_comparisons:
            lines.append("    no changed pools")
        for item in selected_pool_comparisons:
            lines.extend(_pool_text(item))
        lines.append("  pool/stream observations:")
        selected_observation_comparisons = [
            item
            for item in self.observation_comparisons
            if include_unchanged or item.changed
        ]
        if not selected_observation_comparisons:
            lines.append("    no changed pool/stream observations")
        for item in selected_observation_comparisons:
            lines.extend(_observation_text(item))

        if (
            self.reference_stack_coverage is not None
            and self.candidate_stack_coverage is not None
        ):
            lines.append(
                "  allocation stack coverage: "
                f"{self.reference_stack_coverage.ratio:.1%} -> "
                f"{self.candidate_stack_coverage.ratio:.1%}"
            )
        if self.allocation_stack_comparisons:
            lines.append("  top allocation stack deltas:")
            for item in self.allocation_stack_comparisons:
                lines.append(
                    "    "
                    f"{pool_id_label(item.pool_id)} "
                    f"{format_bytes(item.reference_size_bytes)} -> "
                    f"{format_bytes(item.candidate_size_bytes)} "
                    f"(delta {format_delta_bytes(item.delta_size_bytes)}) at {item.stack_key}"
                )
        if self.allocator_events:
            lines.append("  allocator events:")
            for item in self.allocator_events:
                lines.append(
                    "    "
                    f"{pool_id_label(item.pool_id)} {stream_label(item.stream)} "
                    f"{item.action}: {format_bytes(item.size_bytes)} "
                    f"in {item.count} events at {item.stack_key}"
                )
        if self.allocation_lifetimes is not None:
            lines.extend(self.allocation_lifetimes.summary_lines(indent="  "))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": self._COMPARISON_KIND,
            "reference": self.reference.descriptor(),
            "candidate": self.candidate.descriptor(),
            "lifecycle_available": self.lifecycle_available,
            "events_available": self.events_available,
            "events_complete": self.events_complete,
            "warnings": list(self.warnings),
            "allocator_scope_comparisons": [
                item.to_dict() for item in self.allocator_scope_comparisons
            ],
            "stack_coverage": {
                "reference": (
                    self.reference_stack_coverage.to_row()
                    if self.reference_stack_coverage is not None
                    else None
                ),
                "candidate": (
                    self.candidate_stack_coverage.to_row()
                    if self.candidate_stack_coverage is not None
                    else None
                ),
            },
            "pool_comparisons": [item.to_dict() for item in self.pool_comparisons],
            "observation_comparisons": [
                item.to_dict() for item in self.observation_comparisons
            ],
            "allocation_stack_comparisons": [
                item.to_row() for item in self.allocation_stack_comparisons
            ],
            "allocation_stack_observation_comparisons": [
                item.to_row() for item in self.allocation_stack_observation_comparisons
            ],
            "events": self.event_rows(),
            "allocation_lifetimes": (
                self.allocation_lifetimes.to_dict()
                if self.allocation_lifetimes is not None
                else None
            ),
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        sections = [
            "<h2>Allocator Totals</h2>",
            _render_table(
                self.allocator_scope_comparison_rows(
                    include_unchanged=include_unchanged
                ),
                "No changed totals",
            ),
            "<h2>Pools</h2>",
            _render_table(
                self.pool_comparison_rows(include_unchanged=include_unchanged),
                "No changed pools",
            ),
            "<h2>Pool/Stream Observations</h2>",
            _render_table(
                self.observation_comparison_rows(include_unchanged=include_unchanged),
                "No changed pool/stream observations",
            ),
        ]
        if self.allocation_stack_comparisons:
            sections.extend(
                [
                    "<h2>Allocation Stacks</h2>",
                    _render_table(
                        self.allocation_stack_comparison_rows(), "No stack deltas"
                    ),
                ]
            )
        if self.allocator_events:
            sections.extend(
                [
                    "<h2>Allocator Events</h2>",
                    _render_table(self.event_rows(), "No allocator events"),
                ]
            )
        if self.allocation_lifetimes is not None:
            sections.extend(
                [
                    "<h2>Allocation Cohorts</h2>",
                    _render_table(
                        self.allocation_lifetimes.cohort_rows(),
                        "No allocation cohorts",
                    ),
                ]
            )
        return _html_document(
            "Memory comparison "
            f"{_state_label(self.reference)} to "
            f"{_state_label(self.candidate)}",
            sections,
            self.warnings,
        )

    def write(
        self,
        output_dir: str | Path,
        *,
        include_unchanged: bool = True,
    ) -> dict[str, Path]:
        root = _prepare_output(output_dir)
        paths = _write_common(
            root,
            text=self.to_text(include_unchanged=include_unchanged),
            payload=self.to_dict(),
            html=self.to_html(include_unchanged=include_unchanged),
            allocator_scopes=self.allocator_scope_comparison_rows(
                include_unchanged=include_unchanged
            ),
            pools=self.pool_comparison_rows(include_unchanged=include_unchanged),
            observations=self.observation_comparison_rows(
                include_unchanged=include_unchanged
            ),
        )
        if self.allocation_stack_comparisons:
            paths["allocation_stack_comparisons"] = _write_csv(
                root / "allocation_stack_comparisons.csv",
                self.allocation_stack_comparison_rows(),
            )
        if self.allocator_events:
            paths["events"] = _write_csv(root / "events.csv", self.event_rows())
        if self.allocation_lifetimes is not None:
            paths.update(self.allocation_lifetimes.write_csv_files(root))
        return paths


@dataclass(frozen=True)
class MemoryPointComparison(_MemoryStateComparison):
    """State and optional attribution comparison between recording points."""

    _COMPARISON_KIND: ClassVar[str] = "point-comparison"


@dataclass(frozen=True)
class MemorySnapshotComparison(_MemoryStateComparison):
    """State and optional attribution comparison between Probe snapshots."""

    _COMPARISON_KIND: ClassVar[str] = "snapshot-comparison"


def _state_label(value: Any) -> str:
    label = getattr(value, "label", None)
    if label is not None:
        return str(label)
    return f"{value.probe_name}@snapshot-{value.index}"


def _state_scope(reference: Any, candidate: Any) -> str:
    reference_probe = getattr(reference, "probe_id", None)
    candidate_probe = getattr(candidate, "probe_id", None)
    if reference_probe is not None or candidate_probe is not None:
        return "same probe" if reference_probe == candidate_probe else "cross probe"
    return "same run" if reference.run_id == candidate.run_id else "cross run"


@dataclass(frozen=True)
class MemoryTimeline:
    """Absolute states and adjacent deltas for every point in one run."""

    run: Any
    allocator_scope_entries: tuple[MemoryAllocatorScopeTimelineEntry, ...]
    pool_entries: tuple[MemoryPoolTimelineEntry, ...]
    observation_entries: tuple[MemoryObservationTimelineEntry, ...]
    point_comparisons: tuple[MemoryPointComparison, ...]
    allocation_lifetimes: MemoryAllocationLifetimeAnalysis | None = None

    @property
    def warnings(self) -> tuple[str, ...]:
        values = [
            warning
            for comparison in self.point_comparisons
            for warning in comparison.warnings
        ]
        if self.allocation_lifetimes is not None:
            values.extend(self.allocation_lifetimes.warnings)
        return tuple(dict.fromkeys(values))

    def allocator_scope_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.allocator_scope_entries
            if include_unchanged or item.delta.changed
        ]

    def pool_rows(self, *, include_unchanged: bool = True) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.pool_entries
            if include_unchanged or item.delta.changed
        ]

    def observation_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.observation_entries
            if include_unchanged or item.delta.changed
        ]

    def to_text(self, *, include_unchanged: bool = True) -> str:
        lines = [f"CUDA allocator memory timeline {self.run.name!r}"]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        allocator_scopes_by_point: dict[
            int, list[MemoryAllocatorScopeTimelineEntry]
        ] = {}
        pools_by_point: dict[int, list[MemoryPoolTimelineEntry]] = {}
        observations_by_point: dict[int, list[MemoryObservationTimelineEntry]] = {}
        for item in self.allocator_scope_entries:
            allocator_scopes_by_point.setdefault(item.point_index, []).append(item)
        for item in self.pool_entries:
            pools_by_point.setdefault(item.point_index, []).append(item)
        for item in self.observation_entries:
            observations_by_point.setdefault(item.point_index, []).append(item)
        for point in self.run.points:
            lines.append(f"  [{point.index}] {point.label}")
            for item in allocator_scopes_by_point.get(point.index, []):
                if not include_unchanged and not item.delta.changed:
                    continue
                lines.append(
                    "    "
                    f"total[{item.scope}] "
                    f"allocated={format_bytes(item.stats.allocated_bytes)} "
                    f"(delta {format_delta_bytes(item.delta.allocated_bytes)}), "
                    f"reserved={format_bytes(item.stats.reserved_bytes)} "
                    f"(delta {format_delta_bytes(item.delta.reserved_bytes)})"
                )
            for item in pools_by_point.get(point.index, []):
                if not include_unchanged and not item.delta.changed:
                    continue
                lines.append(
                    "    "
                    f"{pool_id_label(item.pool_id)} "
                    f"allocated={format_bytes(item.stats.allocated_bytes)} "
                    f"(delta {format_delta_bytes(item.delta.allocated_bytes)}), "
                    f"reserved={format_bytes(item.stats.reserved_bytes)} "
                    f"(delta {format_delta_bytes(item.delta.reserved_bytes)})"
                )
            for item in observations_by_point.get(point.index, []):
                if not include_unchanged and not item.delta.changed:
                    continue
                lines.append(
                    "      "
                    f"{pool_id_label(item.pool_id)} {stream_label(item.stream)} "
                    f"allocated={format_bytes(item.stats.allocated_bytes)} "
                    f"(delta {format_delta_bytes(item.delta.allocated_bytes)}), "
                    f"reserved={format_bytes(item.stats.reserved_bytes)} "
                    f"(delta {format_delta_bytes(item.delta.reserved_bytes)})"
                )
        if self.allocation_lifetimes is not None:
            lines.extend(self.allocation_lifetimes.summary_lines(indent="  "))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "timeline",
            "run": self.run.descriptor(),
            "points": [point.descriptor() for point in self.run.points],
            "warnings": list(self.warnings),
            "allocator_scope_entries": [
                item.to_dict() for item in self.allocator_scope_entries
            ],
            "pool_entries": [item.to_dict() for item in self.pool_entries],
            "observation_entries": [
                item.to_dict() for item in self.observation_entries
            ],
            "point_comparisons": [item.to_dict() for item in self.point_comparisons],
            "allocation_lifetimes": (
                self.allocation_lifetimes.to_dict()
                if self.allocation_lifetimes is not None
                else None
            ),
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        pool_entries = self.pool_rows(include_unchanged=include_unchanged)
        sections = [
            "<h2>Allocator Totals</h2>",
            _render_table(
                self.allocator_scope_rows(include_unchanged=include_unchanged),
                "No memory points",
            ),
            "<h2>Allocated Memory</h2>",
            _timeline_svg(pool_entries, "state_allocated_bytes"),
            "<h2>Reserved Memory</h2>",
            _timeline_svg(pool_entries, "state_reserved_bytes"),
            "<h2>Pool Timeline</h2>",
            _render_table(pool_entries, "No memory points"),
            "<h2>Pool/Stream Timeline</h2>",
            _render_table(
                self.observation_rows(include_unchanged=include_unchanged),
                "No memory points",
            ),
        ]
        if self.allocation_lifetimes is not None:
            sections.extend(
                [
                    "<h2>Allocation Cohorts</h2>",
                    _cohort_timeline_svg(self.allocation_lifetimes.point_rows()),
                    _render_table(
                        self.allocation_lifetimes.cohort_rows(),
                        "No allocation cohorts",
                    ),
                ]
            )
        return _html_document(
            f"Memory timeline {self.run.name}", sections, self.warnings
        )

    def write(
        self,
        output_dir: str | Path,
        *,
        include_unchanged: bool = True,
    ) -> dict[str, Path]:
        root = _prepare_output(output_dir)
        paths = _write_common(
            root,
            text=self.to_text(include_unchanged=include_unchanged),
            payload=self.to_dict(),
            html=self.to_html(include_unchanged=include_unchanged),
            allocator_scopes=self.allocator_scope_rows(
                include_unchanged=include_unchanged
            ),
            pools=self.pool_rows(include_unchanged=include_unchanged),
            observations=self.observation_rows(include_unchanged=include_unchanged),
        )
        allocation_stack_comparison_rows = [
            {
                "reference": item.reference.label,
                "candidate": item.candidate.label,
                **row,
            }
            for item in self.point_comparisons
            for row in item.allocation_stack_comparison_rows()
        ]
        event_rows = [
            row for item in self.point_comparisons for row in item.event_rows()
        ]
        if allocation_stack_comparison_rows:
            paths["allocation_stack_comparisons"] = _write_csv(
                root / "allocation_stack_comparisons.csv",
                allocation_stack_comparison_rows,
            )
        if event_rows:
            paths["events"] = _write_csv(root / "events.csv", event_rows)
        if self.allocation_lifetimes is not None:
            paths.update(self.allocation_lifetimes.write_csv_files(root))
        return paths


@dataclass(frozen=True)
class MemoryPhaseComparison:
    """Baseline/candidate phase decomposition built from four points."""

    baseline_name: str
    candidate_name: str
    baseline_change: MemoryPointComparison
    candidate_change: MemoryPointComparison
    start_gap: MemoryPointComparison
    end_gap: MemoryPointComparison
    allocator_scope_decomposition: tuple[dict[str, object], ...]
    pool_decomposition: tuple[dict[str, object], ...]

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.baseline_change.warnings,
                    *self.candidate_change.warnings,
                    *self.start_gap.warnings,
                    *self.end_gap.warnings,
                )
            )
        )

    def allocator_scope_decomposition_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            row
            for row in self.allocator_scope_decomposition
            if include_unchanged or _phase_row_changed(row)
        ]

    def pool_decomposition_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            row
            for row in self.pool_decomposition
            if include_unchanged or _phase_row_changed(row)
        ]

    def to_text(self, *, include_unchanged: bool = True) -> str:
        lines = [
            f"Phase comparison {self.baseline_name!r} vs {self.candidate_name!r}",
            "  end_gap = start_gap + candidate_change - baseline_change",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        lines.append("  allocator totals:")
        for row in self.allocator_scope_decomposition_rows(
            include_unchanged=include_unchanged
        ):
            lines.append(
                "    "
                f"total[{row['scope']}] {row['metric']}: "
                f"end_gap {format_delta_bytes(int(row['end_gap_bytes']))} = "
                f"start_gap {format_delta_bytes(int(row['start_gap_bytes']))} + "
                f"candidate_change {format_delta_bytes(int(row['candidate_change_bytes']))} - "
                f"baseline_change {format_delta_bytes(int(row['baseline_change_bytes']))}"
            )
        lines.append("  matched pools:")
        for row in self.pool_decomposition_rows(include_unchanged=include_unchanged):
            lines.append(
                "  "
                f"{row['pool']} {row['metric']}: "
                f"end_gap {format_delta_bytes(int(row['end_gap_bytes']))} = "
                f"start_gap {format_delta_bytes(int(row['start_gap_bytes']))} + "
                f"candidate_change {format_delta_bytes(int(row['candidate_change_bytes']))} - "
                f"baseline_change {format_delta_bytes(int(row['baseline_change_bytes']))}"
            )
        for name, comparison in (
            ("baseline change", self.baseline_change),
            ("candidate change", self.candidate_change),
        ):
            if comparison.allocation_lifetimes is None:
                continue
            lines.append(f"  {name}:")
            lines.extend(comparison.allocation_lifetimes.summary_lines(indent="    "))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "phase-comparison",
            "baseline_name": self.baseline_name,
            "candidate_name": self.candidate_name,
            "warnings": list(self.warnings),
            "allocator_scope_decomposition": list(self.allocator_scope_decomposition),
            "pool_decomposition": list(self.pool_decomposition),
            "baseline_change": self.baseline_change.to_dict(),
            "candidate_change": self.candidate_change.to_dict(),
            "start_gap": self.start_gap.to_dict(),
            "end_gap": self.end_gap.to_dict(),
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        sections = [
            "<h2>Allocator-Scope Decomposition</h2>",
            _render_table(
                self.allocator_scope_decomposition_rows(
                    include_unchanged=include_unchanged
                ),
                "No allocator-scope decomposition rows",
            ),
            "<h2>Pool Decomposition</h2>",
            _render_table(
                self.pool_decomposition_rows(include_unchanged=include_unchanged),
                "No pool decomposition rows",
            ),
        ]
        for name, comparison in (
            ("Baseline Change Cohorts", self.baseline_change),
            ("Candidate Change Cohorts", self.candidate_change),
        ):
            if comparison.allocation_lifetimes is None:
                continue
            sections.extend(
                [
                    f"<h2>{name}</h2>",
                    _render_table(
                        comparison.allocation_lifetimes.cohort_rows(),
                        "No allocation cohorts",
                    ),
                ]
            )
        return _html_document(
            f"Phase comparison {self.baseline_name} vs {self.candidate_name}",
            sections,
            self.warnings,
        )

    def write(
        self,
        output_dir: str | Path,
        *,
        include_unchanged: bool = True,
    ) -> dict[str, Path]:
        root = _prepare_output(output_dir)
        phase_comparisons = (
            ("baseline_change", self.baseline_change),
            ("candidate_change", self.candidate_change),
            ("start_gap", self.start_gap),
            ("end_gap", self.end_gap),
        )
        pool_comparison_rows = [
            {"comparison": name, **row}
            for name, comparison in phase_comparisons
            for row in comparison.pool_comparison_rows(
                include_unchanged=include_unchanged
            )
        ]
        observation_comparison_rows = [
            {"comparison": name, **row}
            for name, comparison in phase_comparisons
            for row in comparison.observation_comparison_rows(
                include_unchanged=include_unchanged
            )
        ]
        allocator_scope_comparison_rows = [
            {"comparison": name, **row}
            for name, comparison in phase_comparisons
            for row in comparison.allocator_scope_comparison_rows(
                include_unchanged=include_unchanged
            )
        ]
        paths = _write_common(
            root,
            text=self.to_text(include_unchanged=include_unchanged),
            payload=self.to_dict(),
            html=self.to_html(include_unchanged=include_unchanged),
            allocator_scopes=allocator_scope_comparison_rows,
            pools=pool_comparison_rows,
            observations=observation_comparison_rows,
        )
        paths["pool_decomposition"] = _write_csv(
            root / "pool_decomposition.csv",
            self.pool_decomposition_rows(include_unchanged=include_unchanged),
        )
        paths["allocator_scope_decomposition"] = _write_csv(
            root / "allocator_scope_decomposition.csv",
            self.allocator_scope_decomposition_rows(
                include_unchanged=include_unchanged
            ),
        )
        allocation_stack_comparison_rows = [
            {"comparison": name, **row}
            for name, comparison in phase_comparisons
            for row in comparison.allocation_stack_comparison_rows()
        ]
        event_rows = [
            {"comparison": name, **row}
            for name, comparison in phase_comparisons
            for row in comparison.event_rows()
        ]
        if allocation_stack_comparison_rows:
            paths["allocation_stack_comparisons"] = _write_csv(
                root / "allocation_stack_comparisons.csv",
                allocation_stack_comparison_rows,
            )
        if event_rows:
            paths["events"] = _write_csv(root / "events.csv", event_rows)
        lifetime_reports = [
            (name, comparison.allocation_lifetimes)
            for name, comparison in phase_comparisons
            if comparison.allocation_lifetimes is not None
        ]
        cohort_rows = [
            {"comparison": name, **row}
            for name, report in lifetime_reports
            for row in report.cohort_rows()
        ]
        cohort_point_rows = [
            {"comparison": name, **row}
            for name, report in lifetime_reports
            for row in report.point_rows()
        ]
        size_rows = [
            {"comparison": name, **row}
            for name, report in lifetime_reports
            for row in report.size_rows()
        ]
        birth_rows = [
            {"comparison": name, **row}
            for name, report in lifetime_reports
            for row in report.birth_rows()
        ]
        release_rows = [
            {"comparison": name, **row}
            for name, report in lifetime_reports
            for row in report.release_rows()
        ]
        if cohort_rows:
            paths["cohorts"] = _write_csv(root / "cohorts.csv", cohort_rows)
            paths["cohort_points"] = _write_csv(
                root / "cohort_points.csv", cohort_point_rows
            )
            paths["size_histograms"] = _write_csv(
                root / "size_histograms.csv", size_rows
            )
        if birth_rows:
            paths["birth_stacks"] = _write_csv(root / "birth_stacks.csv", birth_rows)
        if release_rows:
            paths["release_stacks"] = _write_csv(
                root / "release_stacks.csv", release_rows
            )
        return paths


@dataclass(frozen=True)
class MemoryRunGroupSummary:
    """Per-rank point states and cross-rank extrema for one run group."""

    run_group: Any
    rank_point_entries: tuple[dict[str, object], ...]
    point_aggregates: tuple[dict[str, object], ...]
    warnings: tuple[str, ...] = ()

    def to_text(self, *, include_unchanged: bool = True) -> str:
        del include_unchanged
        lines = [
            f"Memory run group summary {self.run_group.name!r} ranks={list(self.run_group.ranks)}",
            "  values are per rank; GPU memory is not summed across ranks",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        visible_metrics = {"reserved_bytes", "allocated_bytes", "active_bytes"}
        current_point: tuple[int, str] | None = None
        for row in self.point_aggregates:
            if row["metric"] not in visible_metrics:
                continue
            point = (int(row["point_index"]), str(row["point_label"]))
            if point != current_point:
                lines.append(f"  [{point[0]}] {point[1]}")
                current_point = point
            lines.append(
                "    "
                f"total[{row['scope']}] {row['metric']}: "
                f"min {format_bytes(int(row['min_value']))} on rank {row['min_rank']}, "
                f"max {format_bytes(int(row['max_value']))} on rank {row['max_rank']}, "
                f"spread {format_bytes(int(row['spread_value']))}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "run-group-summary",
            "aggregation": "per_rank_extrema_no_sum",
            "run_group": self.run_group.descriptor(),
            "warnings": list(self.warnings),
            "rank_point_entries": list(self.rank_point_entries),
            "point_aggregates": list(self.point_aggregates),
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        del include_unchanged
        return _html_document(
            f"Memory run group summary {self.run_group.name}",
            (
                "<p>Values are per rank; GPU memory is not summed across ranks.</p>",
                "<h2>Cross-Rank Point Summary</h2>",
                _render_table(self.point_aggregates, "No point summaries"),
                "<h2>Per-Rank Point States</h2>",
                _render_table(self.rank_point_entries, "No rank point states"),
            ),
            self.warnings,
        )

    def write(
        self,
        output_dir: str | Path,
        *,
        include_unchanged: bool = True,
    ) -> dict[str, Path]:
        root = _prepare_output(output_dir)
        paths = _write_report_documents(
            root,
            text=self.to_text(include_unchanged=include_unchanged),
            payload=self.to_dict(),
            html=self.to_html(include_unchanged=include_unchanged),
        )
        paths["rank_point_entries"] = _write_csv(
            root / "rank_point_entries.csv", self.rank_point_entries
        )
        paths["point_aggregates"] = _write_csv(
            root / "point_aggregates.csv", self.point_aggregates
        )
        return paths


@dataclass(frozen=True)
class MemoryRunGroupPhaseComparison:
    """Rank-paired four-point phase equations and cross-rank skew."""

    baseline_group: Any
    candidate_group: Any
    rank_comparisons: Mapping[int, MemoryPhaseComparison]
    rank_decomposition: tuple[dict[str, object], ...]
    phase_aggregates: tuple[dict[str, object], ...]
    warnings: tuple[str, ...] = ()

    def rank_decomposition_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            row
            for row in self.rank_decomposition
            if include_unchanged or _phase_row_changed(row)
        ]

    def phase_aggregate_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            row
            for row in self.phase_aggregates
            if include_unchanged or _group_phase_row_changed(row)
        ]

    def to_text(self, *, include_unchanged: bool = True) -> str:
        lines = [
            f"Group phase comparison {self.baseline_group.name!r} vs "
            f"{self.candidate_group.name!r} ranks={list(self.rank_comparisons)}",
            "  values are per rank; GPU memory is not summed across ranks",
            "  end_gap = start_gap + candidate_change - baseline_change",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        visible_metrics = {"reserved_bytes", "allocated_bytes", "active_bytes"}
        for row in self.phase_aggregate_rows(include_unchanged=include_unchanged):
            if row["metric"] not in visible_metrics:
                continue
            lines.append(
                "  "
                f"total[{row['scope']}] {row['metric']}: "
                f"end gap max {format_delta_bytes(int(row['end_gap_max_bytes']))} "
                f"on rank {row['end_gap_max_rank']} "
                f"(spread {format_bytes(int(row['end_gap_spread_bytes']))}); "
                f"change gap max "
                f"{format_delta_bytes(int(row['change_gap_max_bytes']))} "
                f"on rank {row['change_gap_max_rank']} "
                f"(spread {format_bytes(int(row['change_gap_spread_bytes']))})"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "run-group-phase-comparison",
            "aggregation": "per_rank_extrema_no_sum",
            "baseline_group": self.baseline_group.descriptor(),
            "candidate_group": self.candidate_group.descriptor(),
            "warnings": list(self.warnings),
            "rank_decomposition": list(self.rank_decomposition),
            "phase_aggregates": list(self.phase_aggregates),
            "rank_comparisons": {
                str(rank): comparison.to_dict()
                for rank, comparison in self.rank_comparisons.items()
            },
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        return _html_document(
            f"Group phase comparison {self.baseline_group.name} vs "
            f"{self.candidate_group.name}",
            (
                "<p>Values are per rank; GPU memory is not summed across ranks.</p>",
                "<h2>Cross-Rank Phase Summary</h2>",
                _render_table(
                    self.phase_aggregate_rows(include_unchanged=include_unchanged),
                    "No phase summaries",
                ),
                "<h2>Per-Rank Phase Equations</h2>",
                _render_table(
                    self.rank_decomposition_rows(include_unchanged=include_unchanged),
                    "No rank phase rows",
                ),
            ),
            self.warnings,
        )

    def write(
        self,
        output_dir: str | Path,
        *,
        include_unchanged: bool = True,
    ) -> dict[str, Path]:
        root = _prepare_output(output_dir)
        paths = _write_report_documents(
            root,
            text=self.to_text(include_unchanged=include_unchanged),
            payload=self.to_dict(),
            html=self.to_html(include_unchanged=include_unchanged),
        )
        paths["rank_decomposition"] = _write_csv(
            root / "rank_decomposition.csv",
            self.rank_decomposition_rows(include_unchanged=include_unchanged),
        )
        paths["phase_aggregates"] = _write_csv(
            root / "phase_aggregates.csv",
            self.phase_aggregate_rows(include_unchanged=include_unchanged),
        )
        return paths


def _group_phase_row_changed(row: Mapping[str, object]) -> bool:
    return any(
        int(row.get(key, 0))
        for key in (
            "start_gap_max_bytes",
            "baseline_change_max_bytes",
            "candidate_change_max_bytes",
            "end_gap_max_bytes",
        )
    )


def _phase_row_changed(row: Mapping[str, object]) -> bool:
    return any(
        int(row.get(key, 0))
        for key in (
            "start_gap_bytes",
            "baseline_change_bytes",
            "candidate_change_bytes",
            "end_gap_bytes",
        )
    )


def _pool_text(item: MemoryPoolComparison) -> list[str]:
    reference = (
        pool_id_label(item.reference_pool_id)
        if item.reference_pool_id is not None
        else "<none>"
    )
    candidate = (
        pool_id_label(item.candidate_pool_id)
        if item.candidate_pool_id is not None
        else "<none>"
    )
    return [
        f"    {reference} -> {candidate} [{item.match}]",
        "      allocated: "
        + format_comparison(
            item.reference.allocated_bytes,
            item.candidate.allocated_bytes,
            item.delta.allocated_bytes,
        )
        + ", reserved: "
        + format_comparison(
            item.reference.reserved_bytes,
            item.candidate.reserved_bytes,
            item.delta.reserved_bytes,
        ),
        "      active: "
        + format_comparison(
            item.reference.active_bytes,
            item.candidate.active_bytes,
            item.delta.active_bytes,
        )
        + ", requested: "
        + format_comparison(
            item.reference.requested_bytes,
            item.candidate.requested_bytes,
            item.delta.requested_bytes,
        ),
    ]


def _observation_text(item: MemoryObservationComparison) -> list[str]:
    reference = (
        f"{pool_id_label(item.reference_pool_id)} {stream_label(item.reference_stream)}"
        if item.reference_pool_id is not None
        else "<none>"
    )
    candidate = (
        f"{pool_id_label(item.candidate_pool_id)} {stream_label(item.candidate_stream)}"
        if item.candidate_pool_id is not None
        else "<none>"
    )
    return [
        f"    {reference} -> {candidate} [{item.match}]",
        "      allocated: "
        + format_comparison(
            item.reference.allocated_bytes,
            item.candidate.allocated_bytes,
            item.delta.allocated_bytes,
        )
        + ", reserved: "
        + format_comparison(
            item.reference.reserved_bytes,
            item.candidate.reserved_bytes,
            item.delta.reserved_bytes,
        ),
    ]


def _scope_text(item: MemoryAllocatorScopeComparison) -> list[str]:
    return [
        f"    total[{item.scope}]",
        "      allocated: "
        + format_comparison(
            item.reference.allocated_bytes,
            item.candidate.allocated_bytes,
            item.delta.allocated_bytes,
        )
        + ", reserved: "
        + format_comparison(
            item.reference.reserved_bytes,
            item.candidate.reserved_bytes,
            item.delta.reserved_bytes,
        ),
        "      active: "
        + format_comparison(
            item.reference.active_bytes,
            item.candidate.active_bytes,
            item.delta.active_bytes,
        )
        + ", requested: "
        + format_comparison(
            item.reference.requested_bytes,
            item.candidate.requested_bytes,
            item.delta.requested_bytes,
        ),
    ]


def _prepare_output(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_report_documents(
    root: Path,
    *,
    text: str,
    payload: Mapping[str, object],
    html: str,
) -> dict[str, Path]:
    text_path = root / "report.txt"
    json_path = root / "report.json"
    html_path = root / "report.html"
    text_path.write_text(text + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(html, encoding="utf-8")
    return {"text": text_path, "json": json_path, "html": html_path}


def _write_common(
    root: Path,
    *,
    text: str,
    payload: Mapping[str, object],
    html: str,
    allocator_scopes: Sequence[Mapping[str, object]],
    pools: Sequence[Mapping[str, object]],
    observations: Sequence[Mapping[str, object]],
) -> dict[str, Path]:
    paths = _write_report_documents(
        root,
        text=text,
        payload=payload,
        html=html,
    )
    paths.update(
        {
            "allocator_scopes": _write_csv(
                root / "allocator_scopes.csv", allocator_scopes
            ),
            "pools": _write_csv(root / "pools.csv", pools),
            "observations": _write_csv(root / "observations.csv", observations),
        }
    )
    return paths


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields or ["row_type"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _html_document(
    title: str,
    sections: Sequence[str],
    warnings: Sequence[str],
) -> str:
    warning_html = "".join(
        f"<li>{escape(warning)}</li>" for warning in dict.fromkeys(warnings)
    )
    warnings_section = (
        f"<h2>Warnings</h2><ul>{warning_html}</ul>" if warning_html else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    h1 {{ font-size: 24px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    .chart {{ max-width: 1120px; overflow-x: auto; border: 1px solid #d1d5db; padding: 12px; }}
    table {{ border-collapse: collapse; margin-top: 12px; font-size: 13px; min-width: 900px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 7px 9px; text-align: right; white-space: nowrap; }}
    th {{ background: #f3f4f6; color: #374151; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .empty {{ color: #6b7280; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  {warnings_section}
  {"".join(sections)}
</body>
</html>
"""


def _render_table(rows: Sequence[Mapping[str, object]], empty: str) -> str:
    if not rows:
        return f'<p class="empty">{escape(empty)}</p>'
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    header = "".join(f"<th>{escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(field, '')))}</td>" for field in fields)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _timeline_svg(rows: Sequence[Mapping[str, object]], metric: str) -> str:
    if not rows:
        return '<p class="empty">No memory points</p>'
    width, height = 960, 300
    left, right, top, bottom = 80, 20, 20, 50
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(int(row.get(metric, 0) or 0) for row in rows) or 1
    max_index = max(int(row.get("point_index", 0) or 0) for row in rows) or 1
    colors = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706", "#0891b2")
    pools = sorted({str(row.get("pool_id", "")) for row in rows})
    lines = []
    for color_index, pool in enumerate(pools):
        points = []
        for row in rows:
            if str(row.get("pool_id", "")) != pool:
                continue
            x = left + int(row.get("point_index", 0) or 0) / max_index * plot_width
            y = (
                top
                + plot_height
                - int(row.get(metric, 0) or 0) / max_value * plot_height
            )
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            lines.append(
                f'<polyline fill="none" stroke="{colors[color_index % len(colors)]}" '
                f'stroke-width="2" points="{" ".join(points)}"><title>{escape(pool)}</title></polyline>'
            )
    svg = (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" '
        f'y2="{top + plot_height}" stroke="#9ca3af" />'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#9ca3af" />'
        + "".join(lines)
        + f'<text x="8" y="{top + 12}" font-size="12">{escape(metric)}</text>'
        + "</svg>"
    )
    return f'<div class="chart">{svg}</div>'


def _cohort_timeline_svg(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return _timeline_svg((), "state_active_bytes")
    first_index = min(int(row.get("point_index", 0) or 0) for row in rows)
    chart_rows = [
        {
            **row,
            "point_index": int(row.get("point_index", 0) or 0) - first_index,
            "pool_id": row.get("cohort_id", ""),
            "state_active_bytes": row.get("active_bytes", 0),
        }
        for row in rows
    ]
    return _timeline_svg(chart_rows, "state_active_bytes")
