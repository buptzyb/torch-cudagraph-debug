"""Structured text, JSON, CSV, and HTML reports for memory debugging."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from .lifetimes import AllocationCohort
from .models import (
    AllocatorScopeComparison,
    PointAllocatorScopeState,
    PointPoolState,
    PointPoolStreamState,
    PoolComparison,
    PoolStreamComparison,
)
from .stacks import AllocationStackCoverage, AllocationStackDelta
from .events import AllocatorEventSummary
from .summary import (
    format_before_after,
    format_bytes,
    format_delta_bytes,
    pool_id_label,
    stream_label,
)

REPORT_SCHEMA = "torch-cudagraph-debug/memory-report"


@dataclass(frozen=True)
class AllocationLifetimeReport:
    """Point-by-point lifetime report for allocation cohorts in one run."""

    run: Any
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
            selection = f"active_at={self.active_at.label!r}"
        elif self.born_between is not None:
            selection = (
                f"born_between=({self.born_between[0].label!r}, "
                f"{self.born_between[1].label!r}]"
            )
        else:
            selection = "all cohorts"
        lines = [
            f"Allocation cohort lifetimes {self.run.name!r}: "
            f"{self.start.label!r} -> {self.end.label!r} ({selection})"
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
                    f"{birth.before_label!r} -> {birth.after_label!r}: "
                    f"{format_bytes(birth.size_bytes)} in {birth.count} blocks "
                    f"[{birth.confidence}] at {birth.stack_key}"
                )
            for release in cohort.releases:
                lines.append(
                    "    release "
                    f"{release.before_label!r} -> {release.after_label!r}: "
                    f"{format_bytes(release.size_bytes)} in {release.count} blocks "
                    f"[{release.confidence}] at {release.stack_key}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "allocation_lifetimes",
            "run": self.run.descriptor(),
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
            f"Allocation cohort lifetimes for {self.run.name}",
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
            paths["birth_stacks"] = _write_csv(
                root / "birth_stacks.csv", birth_rows
            )
        if release_rows:
            paths["release_stacks"] = _write_csv(
                root / "release_stacks.csv", release_rows
            )
        return paths


@dataclass(frozen=True)
class MemoryComparison:
    """State and optional attribution difference between two memory points."""

    before: Any
    after: Any
    totals: tuple[AllocatorScopeComparison, ...]
    pools: tuple[PoolComparison, ...]
    pool_streams: tuple[PoolStreamComparison, ...]
    allocation_stacks: tuple[AllocationStackDelta, ...] = ()
    allocation_stacks_by_stream: tuple[AllocationStackDelta, ...] = ()
    before_stack_coverage: AllocationStackCoverage | None = None
    after_stack_coverage: AllocationStackCoverage | None = None
    allocator_events: tuple[AllocatorEventSummary, ...] = ()
    allocation_lifetimes: AllocationLifetimeReport | None = None
    events_available: bool = False
    events_complete: bool = True
    lifecycle_available: bool = False
    warnings: tuple[str, ...] = ()

    def total_rows(self, *, include_unchanged: bool = True) -> list[dict[str, object]]:
        return [
            item.to_row() for item in self.totals if include_unchanged or item.changed
        ]

    def pool_rows(self, *, include_unchanged: bool = True) -> list[dict[str, object]]:
        return [
            item.to_row() for item in self.pools if include_unchanged or item.changed
        ]

    def pool_stream_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.pool_streams
            if include_unchanged or item.changed
        ]

    def allocation_stack_rows(self) -> list[dict[str, object]]:
        rows = [{"scope": "pool", **item.to_row()} for item in self.allocation_stacks]
        rows.extend(
            {"scope": "pool_stream", **item.to_row()}
            for item in self.allocation_stacks_by_stream
        )
        return rows

    def event_rows(self) -> list[dict[str, object]]:
        return [
            item.to_row(
                before_label=self.before.label,
                after_label=self.after.label,
            )
            for item in self.allocator_events
        ]

    def to_text(self, *, include_unchanged: bool = True) -> str:
        scope = "same run" if self.before.run_id == self.after.run_id else "cross run"
        lines = [
            f"Memory comparison {self.before.label!r} -> {self.after.label!r} ({scope})"
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        lines.append("  allocator totals:")
        selected_totals = [
            item for item in self.totals if include_unchanged or item.changed
        ]
        if not selected_totals:
            lines.append("    no changed totals")
        for item in selected_totals:
            lines.extend(_scope_text(item))
        lines.append("  pools:")
        selected_pools = [
            item for item in self.pools if include_unchanged or item.changed
        ]
        if not selected_pools:
            lines.append("    no changed pools")
        for item in selected_pools:
            lines.extend(_pool_text(item))
        lines.append("  pool/stream groups:")
        selected_groups = [
            item for item in self.pool_streams if include_unchanged or item.changed
        ]
        if not selected_groups:
            lines.append("    no changed pool/stream groups")
        for item in selected_groups:
            lines.extend(_group_text(item))

        if (
            self.before_stack_coverage is not None
            and self.after_stack_coverage is not None
        ):
            lines.append(
                "  allocation stack coverage: "
                f"{self.before_stack_coverage.ratio:.1%} -> "
                f"{self.after_stack_coverage.ratio:.1%}"
            )
        if self.allocation_stacks:
            lines.append("  top allocation stack deltas:")
            for item in self.allocation_stacks:
                lines.append(
                    "    "
                    f"{pool_id_label(item.pool_id)} "
                    f"{format_bytes(item.before_size_bytes)} -> "
                    f"{format_bytes(item.after_size_bytes)} "
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
            "kind": "comparison",
            "before": self.before.descriptor(),
            "after": self.after.descriptor(),
            "lifecycle_available": self.lifecycle_available,
            "events_available": self.events_available,
            "events_complete": self.events_complete,
            "warnings": list(self.warnings),
            "totals": [item.to_dict() for item in self.totals],
            "stack_coverage": {
                "before": (
                    self.before_stack_coverage.to_row()
                    if self.before_stack_coverage is not None
                    else None
                ),
                "after": (
                    self.after_stack_coverage.to_row()
                    if self.after_stack_coverage is not None
                    else None
                ),
            },
            "pools": [item.to_dict() for item in self.pools],
            "pool_streams": [item.to_dict() for item in self.pool_streams],
            "allocation_stacks": [item.to_row() for item in self.allocation_stacks],
            "allocation_stacks_by_stream": [
                item.to_row() for item in self.allocation_stacks_by_stream
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
                self.total_rows(include_unchanged=include_unchanged),
                "No changed totals",
            ),
            "<h2>Pools</h2>",
            _render_table(
                self.pool_rows(include_unchanged=include_unchanged),
                "No changed pools",
            ),
            "<h2>Pool/Stream Groups</h2>",
            _render_table(
                self.pool_stream_rows(include_unchanged=include_unchanged),
                "No changed pool/stream groups",
            ),
        ]
        if self.allocation_stacks:
            sections.extend(
                [
                    "<h2>Allocation Stacks</h2>",
                    _render_table(self.allocation_stack_rows(), "No stack deltas"),
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
            f"Memory comparison {self.before.label} to {self.after.label}",
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
            totals=self.total_rows(include_unchanged=include_unchanged),
            pools=self.pool_rows(include_unchanged=include_unchanged),
            pool_streams=self.pool_stream_rows(include_unchanged=include_unchanged),
        )
        if self.allocation_stacks:
            paths["allocation_stacks"] = _write_csv(
                root / "allocation_stacks.csv", self.allocation_stack_rows()
            )
        if self.allocator_events:
            paths["events"] = _write_csv(root / "events.csv", self.event_rows())
        if self.allocation_lifetimes is not None:
            paths.update(self.allocation_lifetimes.write_csv_files(root))
        return paths


@dataclass(frozen=True)
class MemoryTimeline:
    """Absolute states and adjacent deltas for every point in one run."""

    run: Any
    totals: tuple[PointAllocatorScopeState, ...]
    pools: tuple[PointPoolState, ...]
    pool_streams: tuple[PointPoolStreamState, ...]
    adjacent: tuple[MemoryComparison, ...]
    allocation_lifetimes: AllocationLifetimeReport | None = None

    @property
    def warnings(self) -> tuple[str, ...]:
        values = [
            warning
            for comparison in self.adjacent
            for warning in comparison.warnings
        ]
        if self.allocation_lifetimes is not None:
            values.extend(self.allocation_lifetimes.warnings)
        return tuple(dict.fromkeys(values))

    def total_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.totals
            if include_unchanged or item.delta.changed
        ]

    def pool_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.pools
            if include_unchanged or item.delta.changed
        ]

    def pool_stream_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.pool_streams
            if include_unchanged or item.delta.changed
        ]

    def to_text(self, *, include_unchanged: bool = True) -> str:
        lines = [f"CUDA allocator memory timeline {self.run.name!r}"]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        totals_by_point: dict[int, list[PointAllocatorScopeState]] = {}
        by_point: dict[int, list[PointPoolState]] = {}
        groups_by_point: dict[int, list[PointPoolStreamState]] = {}
        for item in self.totals:
            totals_by_point.setdefault(item.point_index, []).append(item)
        for item in self.pools:
            by_point.setdefault(item.point_index, []).append(item)
        for item in self.pool_streams:
            groups_by_point.setdefault(item.point_index, []).append(item)
        for point in self.run.points:
            lines.append(f"  [{point.index}] {point.label}")
            for item in totals_by_point.get(point.index, []):
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
            for item in by_point.get(point.index, []):
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
            for item in groups_by_point.get(point.index, []):
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
            "totals": [item.to_dict() for item in self.totals],
            "pools": [item.to_dict() for item in self.pools],
            "pool_streams": [item.to_dict() for item in self.pool_streams],
            "adjacent_comparisons": [item.to_dict() for item in self.adjacent],
            "allocation_lifetimes": (
                self.allocation_lifetimes.to_dict()
                if self.allocation_lifetimes is not None
                else None
            ),
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        pool_rows = self.pool_rows(include_unchanged=include_unchanged)
        sections = [
            "<h2>Allocator Totals</h2>",
            _render_table(
                self.total_rows(include_unchanged=include_unchanged),
                "No memory points",
            ),
            "<h2>Allocated Memory</h2>",
            _timeline_svg(pool_rows, "state_allocated_bytes"),
            "<h2>Reserved Memory</h2>",
            _timeline_svg(pool_rows, "state_reserved_bytes"),
            "<h2>Pool Timeline</h2>",
            _render_table(pool_rows, "No memory points"),
            "<h2>Pool/Stream Timeline</h2>",
            _render_table(
                self.pool_stream_rows(include_unchanged=include_unchanged),
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
            totals=self.total_rows(include_unchanged=include_unchanged),
            pools=self.pool_rows(include_unchanged=include_unchanged),
            pool_streams=self.pool_stream_rows(
                include_unchanged=include_unchanged
            ),
        )
        stack_rows = [
            {"before": item.before.label, "after": item.after.label, **row}
            for item in self.adjacent
            for row in item.allocation_stack_rows()
        ]
        event_rows = [row for item in self.adjacent for row in item.event_rows()]
        if stack_rows:
            paths["allocation_stacks"] = _write_csv(
                root / "allocation_stacks.csv", stack_rows
            )
        if event_rows:
            paths["events"] = _write_csv(root / "events.csv", event_rows)
        if self.allocation_lifetimes is not None:
            paths.update(self.allocation_lifetimes.write_csv_files(root))
        return paths


@dataclass(frozen=True)
class PhaseComparison:
    """Baseline/candidate phase decomposition built from four points."""

    baseline_name: str
    candidate_name: str
    baseline_growth: MemoryComparison
    candidate_growth: MemoryComparison
    start_delta: MemoryComparison
    end_delta: MemoryComparison
    total_decomposition: tuple[dict[str, object], ...]
    decomposition: tuple[dict[str, object], ...]

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.baseline_growth.warnings,
                    *self.candidate_growth.warnings,
                    *self.start_delta.warnings,
                    *self.end_delta.warnings,
                )
            )
        )

    def total_decomposition_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            row
            for row in self.total_decomposition
            if include_unchanged or _phase_row_changed(row)
        ]

    def decomposition_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            row
            for row in self.decomposition
            if include_unchanged or _phase_row_changed(row)
        ]

    def to_text(self, *, include_unchanged: bool = True) -> str:
        lines = [
            f"Phase comparison {self.baseline_name!r} vs {self.candidate_name!r}",
            "  end delta = start delta + candidate growth - baseline growth",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        lines.append("  allocator totals:")
        for row in self.total_decomposition_rows(
            include_unchanged=include_unchanged
        ):
            lines.append(
                "    "
                f"total[{row['scope']}] {row['metric']}: "
                f"end {format_delta_bytes(int(row['end_delta_bytes']))} = "
                f"start {format_delta_bytes(int(row['start_delta_bytes']))} + "
                f"candidate {format_delta_bytes(int(row['candidate_growth_bytes']))} - "
                f"baseline {format_delta_bytes(int(row['baseline_growth_bytes']))}"
            )
        lines.append("  matched pools:")
        for row in self.decomposition_rows(include_unchanged=include_unchanged):
            lines.append(
                "  "
                f"{row['pool']} {row['metric']}: "
                f"end {format_delta_bytes(int(row['end_delta_bytes']))} = "
                f"start {format_delta_bytes(int(row['start_delta_bytes']))} + "
                f"candidate {format_delta_bytes(int(row['candidate_growth_bytes']))} - "
                f"baseline {format_delta_bytes(int(row['baseline_growth_bytes']))}"
            )
        for name, comparison in (
            ("baseline growth", self.baseline_growth),
            ("candidate growth", self.candidate_growth),
        ):
            if comparison.allocation_lifetimes is None:
                continue
            lines.append(f"  {name}:")
            lines.extend(comparison.allocation_lifetimes.summary_lines(indent="    "))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "phase_comparison",
            "baseline_name": self.baseline_name,
            "candidate_name": self.candidate_name,
            "warnings": list(self.warnings),
            "total_decomposition": list(self.total_decomposition),
            "decomposition": list(self.decomposition),
            "baseline_growth": self.baseline_growth.to_dict(),
            "candidate_growth": self.candidate_growth.to_dict(),
            "start_delta": self.start_delta.to_dict(),
            "end_delta": self.end_delta.to_dict(),
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        sections = [
            "<h2>Allocator Total Decomposition</h2>",
            _render_table(
                self.total_decomposition_rows(
                    include_unchanged=include_unchanged
                ),
                "No allocator total phase rows",
            ),
            "<h2>Phase Decomposition</h2>",
            _render_table(
                self.decomposition_rows(include_unchanged=include_unchanged),
                "No phase rows",
            ),
        ]
        for name, comparison in (
            ("Baseline Growth Cohorts", self.baseline_growth),
            ("Candidate Growth Cohorts", self.candidate_growth),
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
        comparisons = (
            ("baseline_growth", self.baseline_growth),
            ("candidate_growth", self.candidate_growth),
            ("start_delta", self.start_delta),
            ("end_delta", self.end_delta),
        )
        pool_rows = [
            {"comparison": name, **row}
            for name, comparison in comparisons
            for row in comparison.pool_rows(include_unchanged=include_unchanged)
        ]
        group_rows = [
            {"comparison": name, **row}
            for name, comparison in comparisons
            for row in comparison.pool_stream_rows(include_unchanged=include_unchanged)
        ]
        total_rows = [
            {"comparison": name, **row}
            for name, comparison in comparisons
            for row in comparison.total_rows(include_unchanged=include_unchanged)
        ]
        paths = _write_common(
            root,
            text=self.to_text(include_unchanged=include_unchanged),
            payload=self.to_dict(),
            html=self.to_html(include_unchanged=include_unchanged),
            totals=total_rows,
            pools=pool_rows,
            pool_streams=group_rows,
        )
        paths["phase"] = _write_csv(
            root / "phase.csv",
            self.decomposition_rows(include_unchanged=include_unchanged),
        )
        paths["phase_totals"] = _write_csv(
            root / "phase_totals.csv",
            self.total_decomposition_rows(include_unchanged=include_unchanged),
        )
        stack_rows = [
            {"comparison": name, **row}
            for name, comparison in comparisons
            for row in comparison.allocation_stack_rows()
        ]
        event_rows = [
            {"comparison": name, **row}
            for name, comparison in comparisons
            for row in comparison.event_rows()
        ]
        if stack_rows:
            paths["allocation_stacks"] = _write_csv(
                root / "allocation_stacks.csv", stack_rows
            )
        if event_rows:
            paths["events"] = _write_csv(root / "events.csv", event_rows)
        lifetime_reports = [
            (name, comparison.allocation_lifetimes)
            for name, comparison in comparisons
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
class MemoryGroupSummary:
    """Per-rank point states and cross-rank extrema for one run group."""

    group: Any
    rank_points: tuple[dict[str, object], ...]
    point_summary: tuple[dict[str, object], ...]
    warnings: tuple[str, ...] = ()

    def to_text(self, *, include_unchanged: bool = True) -> str:
        del include_unchanged
        lines = [
            f"Memory run group summary {self.group.name!r} ranks={list(self.group.ranks)}",
            "  values are per rank; GPU memory is not summed across ranks",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        visible_metrics = {"reserved_bytes", "allocated_bytes", "active_bytes"}
        current_point: tuple[int, str] | None = None
        for row in self.point_summary:
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
            "kind": "group_summary",
            "aggregation": "per_rank_extrema_no_sum",
            "group": self.group.descriptor(),
            "warnings": list(self.warnings),
            "rank_points": list(self.rank_points),
            "point_summary": list(self.point_summary),
        }

    def to_html(self, *, include_unchanged: bool = True) -> str:
        del include_unchanged
        return _html_document(
            f"Memory run group summary {self.group.name}",
            (
                "<p>Values are per rank; GPU memory is not summed across ranks.</p>",
                "<h2>Cross-Rank Point Summary</h2>",
                _render_table(self.point_summary, "No point summaries"),
                "<h2>Per-Rank Point States</h2>",
                _render_table(self.rank_points, "No rank point states"),
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
        paths["rank_points"] = _write_csv(root / "rank_points.csv", self.rank_points)
        paths["point_summary"] = _write_csv(
            root / "point_summary.csv", self.point_summary
        )
        return paths


@dataclass(frozen=True)
class GroupPhaseComparison:
    """Rank-paired four-point phase equations and cross-rank skew."""

    baseline_group: Any
    candidate_group: Any
    rank_comparisons: Mapping[int, PhaseComparison]
    rank_phase: tuple[dict[str, object], ...]
    phase_summary: tuple[dict[str, object], ...]
    warnings: tuple[str, ...] = ()

    def rank_phase_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            row
            for row in self.rank_phase
            if include_unchanged or _phase_row_changed(row)
        ]

    def phase_summary_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            row
            for row in self.phase_summary
            if include_unchanged or _group_phase_row_changed(row)
        ]

    def to_text(self, *, include_unchanged: bool = True) -> str:
        lines = [
            f"Group phase comparison {self.baseline_group.name!r} vs "
            f"{self.candidate_group.name!r} ranks={list(self.rank_comparisons)}",
            "  values are per rank; GPU memory is not summed across ranks",
            "  end delta = start delta + candidate growth - baseline growth",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        visible_metrics = {"reserved_bytes", "allocated_bytes", "active_bytes"}
        for row in self.phase_summary_rows(
            include_unchanged=include_unchanged
        ):
            if row["metric"] not in visible_metrics:
                continue
            lines.append(
                "  "
                f"total[{row['scope']}] {row['metric']}: "
                f"end max {format_delta_bytes(int(row['end_delta_max_bytes']))} "
                f"on rank {row['end_delta_max_rank']} "
                f"(spread {format_bytes(int(row['end_delta_spread_bytes']))}); "
                f"growth delta max "
                f"{format_delta_bytes(int(row['growth_delta_max_bytes']))} "
                f"on rank {row['growth_delta_max_rank']} "
                f"(spread {format_bytes(int(row['growth_delta_spread_bytes']))})"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "group_phase_comparison",
            "aggregation": "per_rank_extrema_no_sum",
            "baseline_group": self.baseline_group.descriptor(),
            "candidate_group": self.candidate_group.descriptor(),
            "warnings": list(self.warnings),
            "rank_phase": list(self.rank_phase),
            "phase_summary": list(self.phase_summary),
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
                    self.phase_summary_rows(
                        include_unchanged=include_unchanged
                    ),
                    "No phase summaries",
                ),
                "<h2>Per-Rank Phase Equations</h2>",
                _render_table(
                    self.rank_phase_rows(include_unchanged=include_unchanged),
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
        paths["rank_phase"] = _write_csv(
            root / "rank_phase.csv",
            self.rank_phase_rows(include_unchanged=include_unchanged),
        )
        paths["phase_summary"] = _write_csv(
            root / "phase_summary.csv",
            self.phase_summary_rows(include_unchanged=include_unchanged),
        )
        return paths


def _group_phase_row_changed(row: Mapping[str, object]) -> bool:
    return any(
        int(row.get(key, 0))
        for key in (
            "start_delta_max_bytes",
            "baseline_growth_max_bytes",
            "candidate_growth_max_bytes",
            "end_delta_max_bytes",
        )
    )


def _phase_row_changed(row: Mapping[str, object]) -> bool:
    return any(
        int(row.get(key, 0))
        for key in (
            "start_delta_bytes",
            "baseline_growth_bytes",
            "candidate_growth_bytes",
            "end_delta_bytes",
        )
    )


def _pool_text(item: PoolComparison) -> list[str]:
    before = (
        pool_id_label(item.before_pool_id)
        if item.before_pool_id is not None
        else "<none>"
    )
    after = (
        pool_id_label(item.after_pool_id)
        if item.after_pool_id is not None
        else "<none>"
    )
    return [
        f"    {before} -> {after} [{item.match}]",
        "      allocated: "
        + format_before_after(
            item.before.allocated_bytes,
            item.after.allocated_bytes,
            item.delta.allocated_bytes,
        )
        + ", reserved: "
        + format_before_after(
            item.before.reserved_bytes,
            item.after.reserved_bytes,
            item.delta.reserved_bytes,
        ),
        "      active: "
        + format_before_after(
            item.before.active_bytes,
            item.after.active_bytes,
            item.delta.active_bytes,
        )
        + ", requested: "
        + format_before_after(
            item.before.requested_bytes,
            item.after.requested_bytes,
            item.delta.requested_bytes,
        ),
    ]


def _group_text(item: PoolStreamComparison) -> list[str]:
    before = (
        f"{pool_id_label(item.before_pool_id)} {stream_label(item.before_stream)}"
        if item.before_pool_id is not None
        else "<none>"
    )
    after = (
        f"{pool_id_label(item.after_pool_id)} {stream_label(item.after_stream)}"
        if item.after_pool_id is not None
        else "<none>"
    )
    return [
        f"    {before} -> {after} [{item.match}]",
        "      allocated: "
        + format_before_after(
            item.before.allocated_bytes,
            item.after.allocated_bytes,
            item.delta.allocated_bytes,
        )
        + ", reserved: "
        + format_before_after(
            item.before.reserved_bytes,
            item.after.reserved_bytes,
            item.delta.reserved_bytes,
        ),
    ]


def _scope_text(item: AllocatorScopeComparison) -> list[str]:
    return [
        f"    total[{item.scope}]",
        "      allocated: "
        + format_before_after(
            item.before.allocated_bytes,
            item.after.allocated_bytes,
            item.delta.allocated_bytes,
        )
        + ", reserved: "
        + format_before_after(
            item.before.reserved_bytes,
            item.after.reserved_bytes,
            item.delta.reserved_bytes,
        ),
        "      active: "
        + format_before_after(
            item.before.active_bytes,
            item.after.active_bytes,
            item.delta.active_bytes,
        )
        + ", requested: "
        + format_before_after(
            item.before.requested_bytes,
            item.after.requested_bytes,
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
    totals: Sequence[Mapping[str, object]],
    pools: Sequence[Mapping[str, object]],
    pool_streams: Sequence[Mapping[str, object]],
) -> dict[str, Path]:
    paths = _write_report_documents(
        root,
        text=text,
        payload=payload,
        html=html,
    )
    paths.update(
        {
            "totals": _write_csv(root / "totals.csv", totals),
            "pools": _write_csv(root / "pools.csv", pools),
            "pool_streams": _write_csv(
                root / "pool_streams.csv", pool_streams
            ),
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
