"""Structured text, JSON, CSV, and HTML reports for memory debugging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from .._reporting import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    prepare_output_dir,
)
from .allocator_snapshot import (
    format_bytes,
    format_comparison,
    format_delta_bytes,
    pool_id_label,
    stream_label,
)
from .attribution import MemoryAttributionStatus
from .comparison_models import (
    PHASE_METRICS,
    MemoryAllocatorScopeComparison,
    MemoryAllocatorScopePhaseDecomposition,
    MemoryObservationComparison,
    MemoryPoolComparison,
    MemoryPoolPhaseDecomposition,
)
from .events import AllocatorEventSummary
from .lifetimes import AllocationCohort
from .stacks import AllocationStackCoverage, AllocationStackDelta
from .stats import MemoryStats, MemoryStatsDelta

if TYPE_CHECKING:
    from .run_groups import (
        MemoryRankPhaseDecomposition,
        MemoryRankPointAggregate,
        MemoryRankPointState,
        MemoryRankPoolPhaseDecomposition,
        MemoryRunGroup,
        MemoryRunGroupPhaseAggregate,
    )
    from .timeline import (
        MemoryAllocatorScopeTimelineEntry,
        MemoryObservationTimelineEntry,
        MemoryPoolTimelineEntry,
    )

REPORT_SCHEMA = "torch-cudagraph-debug/memory-report"
_CORE_MEMORY_METRICS = (
    "allocated_bytes",
    "reserved_bytes",
    "active_bytes",
    "requested_bytes",
)


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
    display_stack_depth: int
    display_limit: int
    warnings: tuple[str, ...] = ()

    def cohort_rows(self) -> list[dict[str, object]]:
        """Return every cohort; display limits never truncate structured data."""
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

    def size_outcome_rows(self) -> list[dict[str, object]]:
        return [
            item.to_row(cohort.cohort_id)
            for cohort in self.cohorts
            for item in cohort.size_outcomes
        ]

    def birth_rows(self) -> list[dict[str, object]]:
        return [
            item.to_row(cohort.cohort_id)
            for cohort in self.cohorts
            for item in cohort.births
        ]

    def free_request_rows(self) -> list[dict[str, object]]:
        return [
            item.to_row(cohort.cohort_id)
            for cohort in self.cohorts
            for item in cohort.free_requests
        ]

    def free_completion_rows(self) -> list[dict[str, object]]:
        return [
            item.to_row(cohort.cohort_id)
            for cohort in self.cohorts
            for item in cohort.free_completions
        ]

    def summary_lines(
        self,
        *,
        indent: str = "",
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> list[str]:
        visible = self._visible_cohorts(limit)
        depth = self._display_stack_depth(stack_depth)
        lines = [
            f"{indent}allocation cohorts "
            f"(showing {len(visible)} of {len(self.cohorts)}):"
        ]
        if not self.cohorts:
            lines.append(f"{indent}  no allocation cohorts")
            return lines
        for item in visible:
            device = "unknown" if item.device is None else str(item.device)
            lines.append(
                f"{indent}  #{item.display_rank} {item.cohort_id} device[{device}] "
                f"{pool_id_label(item.pool_id)} "
                f"snapshot_peak={format_bytes(item.peak_active_bytes)} "
                f"owner_event_peak={format_bytes(item.event_owner_peak_bytes)} "
                f"unreusable_event_peak={format_bytes(item.event_unreusable_peak_bytes)} "
                f"snapshot_blocks={item.peak_block_count} at "
                f"{item.display_stack(depth)}"
            )
            lines.append(
                f"{indent}    born={format_bytes(item.born_bytes)}; "
                f"free requested={format_bytes(item.free_requested_bytes)}; "
                f"free completed={format_bytes(item.free_completed_bytes)}; "
                f"end owner active={format_bytes(item.owner_active_at_end_bytes)}, "
                f"awaiting free={format_bytes(item.awaiting_free_at_end_bytes)}"
            )
        return lines

    def to_text(
        self,
        *,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> str:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        if self.active_at is not None:
            selection = f"active_at={_state_label(self.active_at)!r}"
        elif self.born_between is not None:
            selection = (
                f"born_between=({_state_label(self.born_between[0])!r}, "
                f"{_state_label(self.born_between[1])!r}]"
            )
        else:
            selection = "all cohorts"
        visible = self._visible_cohorts(limit)
        depth = self._display_stack_depth(stack_depth)
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
        lines.extend(
            self.summary_lines(indent="  ", limit=limit, stack_depth=stack_depth)
        )
        for cohort in visible:
            lines.append(f"  #{cohort.display_rank} {cohort.cohort_id} point states:")
            for point in cohort.points:
                lines.append(
                    "    "
                    f"[{point.point_index}] {point.point_label}: "
                    f"owner_active={format_bytes(point.owner_active_bytes)} "
                    f"({point.owner_active_count} blocks), "
                    f"awaiting_free={format_bytes(point.awaiting_free_bytes)} "
                    f"({point.awaiting_free_count} blocks), "
                    f"active={format_bytes(point.active_bytes)}, "
                    f"requested={format_bytes(point.requested_bytes)}"
                )
            if cohort.size_histogram:
                sizes = ", ".join(
                    f"{format_bytes(item.size_bytes)} requested="
                    f"{format_bytes(item.requested_bytes)} x {item.count}"
                    for item in cohort.size_histogram
                )
                lines.append(f"    sizes: {sizes}")
            if cohort.size_outcomes:
                outcomes = ", ".join(
                    f"{format_bytes(item.size_bytes)} {item.terminal_state} "
                    f"x {item.count}"
                    for item in cohort.size_outcomes
                )
                lines.append(f"    size outcomes: {outcomes}")
            for birth in cohort.births:
                lines.append(
                    "    birth "
                    f"{birth.start_label!r} -> {birth.end_label!r}: "
                    f"{format_bytes(birth.size_bytes)} in {birth.count} blocks "
                    f"[{birth.origin}] at {birth.display_stack(depth)}"
                )
            for request in cohort.free_requests:
                lines.append(
                    "    free requested "
                    f"{request.start_label!r} -> {request.end_label!r}: "
                    f"{format_bytes(request.size_bytes)} in {request.count} blocks "
                    f"[{request.origin}] at {request.display_stack(depth)}"
                )
            for completion in cohort.free_completions:
                lines.append(
                    "    free completed "
                    f"{completion.start_label!r} -> {completion.end_label!r}: "
                    f"{format_bytes(completion.size_bytes)} in {completion.count} blocks "
                    f"[{completion.origin}] at {completion.display_stack(depth)}"
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
            "display_stack_depth": self.display_stack_depth,
            "display_limit": self.display_limit,
            "warnings": list(self.warnings),
            "cohorts": [item.to_dict() for item in self.cohorts],
        }

    def to_html(
        self,
        *,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> str:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        visible = self._visible_cohorts(limit)
        depth = self._display_stack_depth(stack_depth)
        cohort_rows = self._display_cohort_rows(limit, stack_depth)
        point_rows = self._display_point_rows(limit)
        size_rows = [
            item.to_row(cohort.cohort_id)
            for cohort in visible
            for item in cohort.size_histogram
        ]
        outcome_rows = [
            item.to_row(cohort.cohort_id)
            for cohort in visible
            for item in cohort.size_outcomes
        ]
        birth_rows = self._display_transition_rows(visible, "births", depth)
        request_rows = self._display_transition_rows(visible, "free_requests", depth)
        completion_rows = self._display_transition_rows(
            visible, "free_completions", depth
        )
        sections = [
            f"<p>Showing {len(visible)} of {len(self.cohorts)} cohorts.</p>",
            "<h2>Active Bytes by Cohort</h2>",
            _cohort_timeline_svg(point_rows),
            "<h2>Cohorts</h2>",
            _render_table(cohort_rows, "No allocation cohorts"),
            "<h2>Cohort Point States</h2>",
            _render_table(point_rows, "No cohort point states"),
            "<h2>Size Histograms</h2>",
            _render_table(size_rows, "No allocation sizes"),
            "<h2>Size Outcomes</h2>",
            _render_table(outcome_rows, "No allocation outcomes"),
        ]
        if birth_rows:
            sections.extend(
                ["<h2>Birth Stacks</h2>", _render_table(birth_rows, "No births")]
            )
        if request_rows:
            sections.extend(
                [
                    "<h2>Free Request Stacks</h2>",
                    _render_table(request_rows, "No free requests"),
                ]
            )
        if completion_rows:
            sections.extend(
                [
                    "<h2>Free Completion Stacks</h2>",
                    _render_table(completion_rows, "No free completions"),
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
        limit: int | None = None,
        stack_depth: int | None = None,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        root = _prepare_output(output_dir, overwrite=overwrite)
        paths = _write_report_documents(
            root,
            text=self.to_text(limit=limit, stack_depth=stack_depth),
            payload=self.to_dict(),
            html=self.to_html(limit=limit, stack_depth=stack_depth),
        )
        paths.update(self.write_csv_files(root))
        return paths

    def write_csv_files(self, root: Path) -> dict[str, Path]:
        cohort_rows = self.cohort_rows()
        point_rows = self.point_rows()
        size_rows = self.size_rows()
        outcome_rows = self.size_outcome_rows()
        birth_rows = self.birth_rows()
        request_rows = self.free_request_rows()
        completion_rows = self.free_completion_rows()
        paths = {
            "cohorts": _write_csv(root / "cohorts.csv", cohort_rows),
            "cohort_points": _write_csv(root / "cohort_points.csv", point_rows),
            "size_histograms": _write_csv(root / "size_histograms.csv", size_rows),
            "size_outcomes": _write_csv(root / "size_outcomes.csv", outcome_rows),
        }
        if birth_rows:
            paths["birth_stacks"] = _write_csv(root / "birth_stacks.csv", birth_rows)
        if request_rows:
            paths["free_request_stacks"] = _write_csv(
                root / "free_request_stacks.csv", request_rows
            )
        if completion_rows:
            paths["free_completion_stacks"] = _write_csv(
                root / "free_completion_stacks.csv", completion_rows
            )
        return paths

    def _visible_cohorts(self, limit: int | None) -> tuple[AllocationCohort, ...]:
        resolved = self.display_limit if limit is None else limit
        if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 1:
            raise ValueError("limit must be an integer >= 1")
        return self.cohorts[:resolved]

    def _display_stack_depth(self, stack_depth: int | None) -> int:
        resolved = self.display_stack_depth if stack_depth is None else stack_depth
        if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 1:
            raise ValueError("stack_depth must be an integer >= 1")
        return resolved

    def _display_cohort_rows(
        self, limit: int | None = None, stack_depth: int | None = None
    ) -> list[dict[str, object]]:
        depth = self._display_stack_depth(stack_depth)
        rows = []
        for cohort in self._visible_cohorts(limit):
            row = cohort.to_row()
            row.pop("stack_frames_json")
            row["stack_key"] = cohort.display_stack(depth)
            rows.append(row)
        return rows

    def _display_point_rows(self, limit: int | None = None) -> list[dict[str, object]]:
        return [
            item.to_row(cohort.cohort_id)
            for cohort in self._visible_cohorts(limit)
            for item in cohort.points
        ]

    @staticmethod
    def _display_transition_rows(
        cohorts: Sequence[AllocationCohort],
        attribute: Literal["births", "free_requests", "free_completions"],
        stack_depth: int,
    ) -> list[dict[str, object]]:
        rows = []
        for cohort in cohorts:
            for transition in getattr(cohort, attribute):
                row = transition.to_row(cohort.cohort_id)
                row.pop("stack_frames_json")
                row["stack_key"] = transition.display_stack(stack_depth)
                rows.append(row)
        return rows


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
    attribution_status: MemoryAttributionStatus = field(
        default_factory=MemoryAttributionStatus
    )
    lifecycle_available: bool = False
    lifecycle_confidence: Literal["unavailable", "approximate", "exact"] = "unavailable"
    warnings: tuple[str, ...] = ()
    display_stack_depth: int = 2
    display_limit: int = 20

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

    def event_dicts(self) -> list[dict[str, object]]:
        return [
            item.to_dict(
                reference_label=_state_label(self.reference),
                candidate_label=_state_label(self.candidate),
            )
            for item in self.allocator_events
        ]

    def _visible_attribution_rows(
        self,
        values: Sequence[Any],
        limit: int | None,
    ) -> tuple[Any, ...]:
        resolved = _resolve_display_limit(self.display_limit, limit)
        return tuple(values[:resolved])

    def _display_allocation_stack_comparison_rows(
        self,
        *,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> list[dict[str, object]]:
        depth = _resolve_display_stack_depth(self.display_stack_depth, stack_depth)
        rows = [
            {"scope": "pool", **_display_stack_delta_row(item, depth)}
            for item in self._visible_attribution_rows(
                self.allocation_stack_comparisons, limit
            )
        ]
        rows.extend(
            {"scope": "observation", **_display_stack_delta_row(item, depth)}
            for item in self._visible_attribution_rows(
                self.allocation_stack_observation_comparisons, limit
            )
        )
        return rows

    def _display_event_rows(
        self,
        *,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> list[dict[str, object]]:
        depth = _resolve_display_stack_depth(self.display_stack_depth, stack_depth)
        reference_label = _state_label(self.reference)
        candidate_label = _state_label(self.candidate)
        return [
            _display_event_row(
                item,
                reference_label=reference_label,
                candidate_label=candidate_label,
                stack_depth=depth,
            )
            for item in self._visible_attribution_rows(self.allocator_events, limit)
        ]

    def _attribution_text_lines(
        self,
        *,
        indent: str,
        limit: int | None,
        stack_depth: int | None,
    ) -> list[str]:
        depth = _resolve_display_stack_depth(self.display_stack_depth, stack_depth)
        lines: list[str] = []
        for label, values in (
            ("allocation stack deltas", self.allocation_stack_comparisons),
            (
                "allocation stack deltas by stream",
                self.allocation_stack_observation_comparisons,
            ),
        ):
            if not values:
                continue
            visible = self._visible_attribution_rows(values, limit)
            lines.append(f"{indent}{label} (showing {len(visible)} of {len(values)}):")
            lines.extend(
                _stack_delta_text(
                    item,
                    indent=f"{indent}  ",
                    stack_depth=depth,
                )
                for item in visible
            )
        if self.allocator_events:
            visible = self._visible_attribution_rows(self.allocator_events, limit)
            lines.append(
                f"{indent}allocator events "
                f"(showing {len(visible)} of {len(self.allocator_events)}):"
            )
            lines.extend(
                f"{indent}  device[{item.device_index}] "
                f"{pool_id_label(item.pool_id) if item.pool_id is not None else 'pool[unknown]'} "
                f"{stream_label(item.stream)} {item.action}: "
                f"{format_bytes(item.size_bytes)} in {item.count} events "
                f"[{item.attribution_confidence}] at "
                f"{item.display_stack(depth)}"
                for item in visible
            )
        return lines

    def to_text(
        self,
        *,
        include_unchanged: bool = True,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> str:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        reference_label = _state_label(self.reference)
        candidate_label = _state_label(self.candidate)
        lines = [
            f"Memory comparison {reference_label!r} -> {candidate_label!r} "
            f"({_state_scope(self.reference, self.candidate)})"
        ]
        lines.append(f"  address lifecycle: {self.lifecycle_confidence}")
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
        lines.append("  device/pool/stream observations:")
        selected_observation_comparisons = [
            item
            for item in self.observation_comparisons
            if include_unchanged or item.changed
        ]
        if not selected_observation_comparisons:
            lines.append("    no changed device/pool/stream observations")
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
        lines.extend(
            self._attribution_text_lines(
                indent="  ",
                limit=limit,
                stack_depth=stack_depth,
            )
        )
        if self.allocation_lifetimes is not None:
            lines.extend(
                self.allocation_lifetimes.summary_lines(
                    indent="  ",
                    limit=limit,
                    stack_depth=stack_depth,
                )
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": self._COMPARISON_KIND,
            "reference": self.reference.descriptor(),
            "candidate": self.candidate.descriptor(),
            "lifecycle_confidence": self.lifecycle_confidence,
            "lifecycle_available": self.lifecycle_available,
            "attribution_status": self.attribution_status.to_dict(),
            "display_stack_depth": self.display_stack_depth,
            "display_limit": self.display_limit,
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
                item.to_dict() for item in self.allocation_stack_comparisons
            ],
            "allocation_stack_observation_comparisons": [
                item.to_dict() for item in self.allocation_stack_observation_comparisons
            ],
            "events": self.event_dicts(),
            "allocation_lifetimes": (
                self.allocation_lifetimes.to_dict()
                if self.allocation_lifetimes is not None
                else None
            ),
        }

    def to_html(
        self,
        *,
        include_unchanged: bool = True,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> str:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        stack_rows = self._display_allocation_stack_comparison_rows(
            limit=limit,
            stack_depth=stack_depth,
        )
        stack_total = len(self.allocation_stack_comparisons) + len(
            self.allocation_stack_observation_comparisons
        )
        event_rows = self._display_event_rows(
            limit=limit,
            stack_depth=stack_depth,
        )
        cohort_rows = (
            self.allocation_lifetimes._display_cohort_rows(
                limit=limit,
                stack_depth=stack_depth,
            )
            if self.allocation_lifetimes is not None
            else []
        )
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
                "No changed device/pool/stream observations",
            ),
        ]
        if stack_rows:
            sections.extend(
                [
                    "<h2>Allocation Stacks</h2>",
                    _render_table(
                        stack_rows,
                        "No stack deltas",
                        caption=_showing_caption(len(stack_rows), stack_total),
                    ),
                ]
            )
        if event_rows:
            sections.extend(
                [
                    "<h2>Allocator Events</h2>",
                    _render_table(
                        event_rows,
                        "No allocator events",
                        caption=_showing_caption(
                            len(event_rows), len(self.allocator_events)
                        ),
                    ),
                ]
            )
        if self.allocation_lifetimes is not None:
            sections.extend(
                [
                    "<h2>Allocation Cohorts</h2>",
                    _render_table(
                        cohort_rows,
                        "No allocation cohorts",
                        caption=_showing_caption(
                            len(cohort_rows),
                            len(self.allocation_lifetimes.cohorts),
                            noun="cohorts",
                        ),
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
        limit: int | None = None,
        stack_depth: int | None = None,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        root = _prepare_output(output_dir, overwrite=overwrite)
        paths = _write_common(
            root,
            text=self.to_text(
                include_unchanged=include_unchanged,
                limit=limit,
                stack_depth=stack_depth,
            ),
            payload=self.to_dict(),
            html=self.to_html(
                include_unchanged=include_unchanged,
                limit=limit,
                stack_depth=stack_depth,
            ),
            allocator_scopes=self.allocator_scope_comparison_rows(
                include_unchanged=include_unchanged
            ),
            pools=self.pool_comparison_rows(include_unchanged=include_unchanged),
            observations=self.observation_comparison_rows(
                include_unchanged=include_unchanged
            ),
        )
        if (
            self.allocation_stack_comparisons
            or self.allocation_stack_observation_comparisons
        ):
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
    return f"{value.probe_name}@snapshot-{value.snapshot_index}"


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
    display_stack_depth: int = 2
    display_limit: int = 20

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
            if include_unchanged or (item.delta is not None and item.delta.changed)
        ]

    def pool_rows(self, *, include_unchanged: bool = True) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.pool_entries
            if include_unchanged or (item.delta is not None and item.delta.changed)
        ]

    def observation_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.observation_entries
            if include_unchanged or (item.delta is not None and item.delta.changed)
        ]

    def to_text(
        self,
        *,
        include_unchanged: bool = True,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> str:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
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
            point_lines = []
            for item in allocator_scopes_by_point.get(point.index, []):
                if not include_unchanged and (
                    item.delta is None or not item.delta.changed
                ):
                    continue
                point_lines.extend(
                    _timeline_stats_text(
                        f"total[{item.scope}]",
                        item.stats,
                        item.delta,
                        indent="    ",
                    )
                )
            for item in pools_by_point.get(point.index, []):
                if not include_unchanged and (
                    item.delta is None or not item.delta.changed
                ):
                    continue
                point_lines.extend(
                    _timeline_stats_text(
                        item.key.label,
                        item.stats,
                        item.delta,
                        indent="    ",
                    )
                )
            for item in observations_by_point.get(point.index, []):
                if not include_unchanged and (
                    item.delta is None or not item.delta.changed
                ):
                    continue
                point_lines.extend(
                    _timeline_stats_text(
                        item.key.label,
                        item.stats,
                        item.delta,
                        indent="      ",
                    )
                )
            if point_lines:
                lines.append(f"  [{point.index}] {point.label}")
                lines.extend(point_lines)
        for comparison in self.point_comparisons:
            if not (
                comparison.allocation_stack_comparisons
                or comparison.allocation_stack_observation_comparisons
                or comparison.allocator_events
            ):
                continue
            lines.append(
                f"  attribution {_state_label(comparison.reference)!r} -> "
                f"{_state_label(comparison.candidate)!r}:"
            )
            lines.extend(
                comparison._attribution_text_lines(
                    indent="    ",
                    limit=limit,
                    stack_depth=stack_depth,
                )
            )
        if self.allocation_lifetimes is not None:
            lines.extend(
                self.allocation_lifetimes.summary_lines(
                    indent="  ",
                    limit=limit,
                    stack_depth=stack_depth,
                )
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "timeline",
            "run": self.run.descriptor(),
            "points": [point.descriptor() for point in self.run.points],
            "display_stack_depth": self.display_stack_depth,
            "display_limit": self.display_limit,
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

    def to_html(
        self,
        *,
        include_unchanged: bool = True,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> str:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        pool_entries = self.pool_rows(include_unchanged=include_unchanged)
        # Charts must plot the true per-point series; the include_unchanged
        # filter is a table concern and would bend or drop chart lines.
        chart_entries = self.pool_rows(include_unchanged=True)
        allocation_stack_rows = [
            {
                "reference": _state_label(item.reference),
                "candidate": _state_label(item.candidate),
                **row,
            }
            for item in self.point_comparisons
            for row in item._display_allocation_stack_comparison_rows(
                limit=limit,
                stack_depth=stack_depth,
            )
        ]
        event_rows = [
            row
            for item in self.point_comparisons
            for row in item._display_event_rows(
                limit=limit,
                stack_depth=stack_depth,
            )
        ]
        allocation_stack_total = sum(
            len(item.allocation_stack_comparisons)
            + len(item.allocation_stack_observation_comparisons)
            for item in self.point_comparisons
        )
        event_total = sum(len(item.allocator_events) for item in self.point_comparisons)
        cohort_rows = (
            self.allocation_lifetimes._display_cohort_rows(
                limit=limit,
                stack_depth=stack_depth,
            )
            if self.allocation_lifetimes is not None
            else []
        )
        cohort_point_rows = (
            self.allocation_lifetimes._display_point_rows(limit)
            if self.allocation_lifetimes is not None
            else []
        )
        sections = [
            "<h2>Allocator Totals</h2>",
            _render_table(
                self.allocator_scope_rows(include_unchanged=include_unchanged),
                "No memory points",
            ),
            "<h2>Allocated Memory</h2>",
            _timeline_svg(chart_entries, "state_allocated_bytes"),
            "<h2>Reserved Memory</h2>",
            _timeline_svg(chart_entries, "state_reserved_bytes"),
            "<h2>Active Memory</h2>",
            _timeline_svg(chart_entries, "state_active_bytes"),
            "<h2>Requested Memory</h2>",
            _timeline_svg(chart_entries, "state_requested_bytes"),
            "<h2>Pool Timeline</h2>",
            _render_table(pool_entries, "No memory points"),
            "<h2>Pool/Stream Timeline</h2>",
            _render_table(
                self.observation_rows(include_unchanged=include_unchanged),
                "No memory points",
            ),
        ]
        if allocation_stack_rows:
            sections.extend(
                [
                    "<h2>Allocation Stack Deltas</h2>",
                    _render_table(
                        allocation_stack_rows,
                        "No stack deltas",
                        caption=_showing_caption(
                            len(allocation_stack_rows), allocation_stack_total
                        ),
                    ),
                ]
            )
        if event_rows:
            sections.extend(
                [
                    "<h2>Allocator Events</h2>",
                    _render_table(
                        event_rows,
                        "No allocator events",
                        caption=_showing_caption(len(event_rows), event_total),
                    ),
                ]
            )
        if self.allocation_lifetimes is not None:
            sections.extend(
                [
                    "<h2>Allocation Cohorts</h2>",
                    _cohort_timeline_svg(cohort_point_rows),
                    _render_table(
                        cohort_rows,
                        "No allocation cohorts",
                        caption=_showing_caption(
                            len(cohort_rows),
                            len(self.allocation_lifetimes.cohorts),
                            noun="cohorts",
                        ),
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
        limit: int | None = None,
        stack_depth: int | None = None,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        root = _prepare_output(output_dir, overwrite=overwrite)
        paths = _write_common(
            root,
            text=self.to_text(
                include_unchanged=include_unchanged,
                limit=limit,
                stack_depth=stack_depth,
            ),
            payload=self.to_dict(),
            html=self.to_html(
                include_unchanged=include_unchanged,
                limit=limit,
                stack_depth=stack_depth,
            ),
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
    allocator_scope_decomposition: tuple[MemoryAllocatorScopePhaseDecomposition, ...]
    pool_decomposition: tuple[MemoryPoolPhaseDecomposition, ...]
    display_stack_depth: int = 2
    display_limit: int = 20

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
            item.to_row()
            for item in self.allocator_scope_decomposition
            if include_unchanged or item.changed
        ]

    def pool_decomposition_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.pool_decomposition
            if include_unchanged or item.changed
        ]

    def to_text(
        self,
        *,
        include_unchanged: bool = True,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> str:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        lines = [
            f"Phase comparison {self.baseline_name!r} vs {self.candidate_name!r}",
            "  end_gap = start_gap + candidate_change - baseline_change",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        lines.append("  allocator totals:")
        allocator_rows = self.allocator_scope_decomposition_rows(
            include_unchanged=include_unchanged
        )
        if not allocator_rows:
            lines.append("    no changed allocator totals")
        for row in allocator_rows:
            lines.append(
                "    "
                f"total[{row['scope']}] {row['metric']}: "
                f"end_gap {format_delta_bytes(int(row['end_gap_bytes']))} = "
                f"start_gap {format_delta_bytes(int(row['start_gap_bytes']))} + "
                f"candidate_change {format_delta_bytes(int(row['candidate_change_bytes']))} - "
                f"baseline_change {format_delta_bytes(int(row['baseline_change_bytes']))}"
            )
        lines.append("  matched pools:")
        pool_rows = self.pool_decomposition_rows(include_unchanged=include_unchanged)
        if not pool_rows:
            lines.append("    no changed matched pools")
        for row in pool_rows:
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
            ("start gap", self.start_gap),
            ("end gap", self.end_gap),
        ):
            attribution_lines = comparison._attribution_text_lines(
                indent="    ",
                limit=limit,
                stack_depth=stack_depth,
            )
            if attribution_lines:
                lines.append(f"  {name} attribution:")
                lines.extend(attribution_lines)
            if comparison.allocation_lifetimes is not None:
                lines.append(f"  {name}:")
                lines.extend(
                    comparison.allocation_lifetimes.summary_lines(
                        indent="    ",
                        limit=limit,
                        stack_depth=stack_depth,
                    )
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "phase-comparison",
            "baseline_name": self.baseline_name,
            "candidate_name": self.candidate_name,
            "display_stack_depth": self.display_stack_depth,
            "display_limit": self.display_limit,
            "warnings": list(self.warnings),
            "allocator_scope_decomposition": [
                item.to_dict() for item in self.allocator_scope_decomposition
            ],
            "pool_decomposition": [item.to_dict() for item in self.pool_decomposition],
            "baseline_change": self.baseline_change.to_dict(),
            "candidate_change": self.candidate_change.to_dict(),
            "start_gap": self.start_gap.to_dict(),
            "end_gap": self.end_gap.to_dict(),
        }

    def to_html(
        self,
        *,
        include_unchanged: bool = True,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> str:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
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
        for comparison_name, comparison in (
            ("Baseline Change", self.baseline_change),
            ("Candidate Change", self.candidate_change),
            ("Start Gap", self.start_gap),
            ("End Gap", self.end_gap),
        ):
            sections.extend(
                [
                    f"<h2>{comparison_name}: Allocator Totals</h2>",
                    _render_table(
                        comparison.allocator_scope_comparison_rows(
                            include_unchanged=include_unchanged
                        ),
                        "No selected allocator totals",
                    ),
                    f"<h2>{comparison_name}: Pools</h2>",
                    _render_table(
                        comparison.pool_comparison_rows(
                            include_unchanged=include_unchanged
                        ),
                        "No selected pools",
                    ),
                    f"<h2>{comparison_name}: Pool/Stream Observations</h2>",
                    _render_table(
                        comparison.observation_comparison_rows(
                            include_unchanged=include_unchanged
                        ),
                        "No selected device/pool/stream observations",
                    ),
                ]
            )
            stack_rows = comparison._display_allocation_stack_comparison_rows(
                limit=limit,
                stack_depth=stack_depth,
            )
            if stack_rows:
                sections.extend(
                    [
                        f"<h2>{comparison_name}: Allocation Stacks</h2>",
                        _render_table(
                            stack_rows,
                            "No stack deltas",
                            caption=_showing_caption(
                                len(stack_rows),
                                len(comparison.allocation_stack_comparisons)
                                + len(
                                    comparison.allocation_stack_observation_comparisons
                                ),
                            ),
                        ),
                    ]
                )
            event_rows = comparison._display_event_rows(
                limit=limit,
                stack_depth=stack_depth,
            )
            if event_rows:
                sections.extend(
                    [
                        f"<h2>{comparison_name}: Allocator Events</h2>",
                        _render_table(
                            event_rows,
                            "No allocator events",
                            caption=_showing_caption(
                                len(event_rows), len(comparison.allocator_events)
                            ),
                        ),
                    ]
                )
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
                        comparison.allocation_lifetimes._display_cohort_rows(
                            limit=limit,
                            stack_depth=stack_depth,
                        ),
                        "No allocation cohorts",
                        caption=_showing_caption(
                            min(
                                limit,
                                len(comparison.allocation_lifetimes.cohorts),
                            ),
                            len(comparison.allocation_lifetimes.cohorts),
                            noun="cohorts",
                        ),
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
        limit: int | None = None,
        stack_depth: int | None = None,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        root = _prepare_output(output_dir, overwrite=overwrite)
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
            text=self.to_text(
                include_unchanged=include_unchanged,
                limit=limit,
                stack_depth=stack_depth,
            ),
            payload=self.to_dict(),
            html=self.to_html(
                include_unchanged=include_unchanged,
                limit=limit,
                stack_depth=stack_depth,
            ),
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
        size_outcome_rows = [
            {"comparison": name, **row}
            for name, report in lifetime_reports
            for row in report.size_outcome_rows()
        ]
        birth_rows = [
            {"comparison": name, **row}
            for name, report in lifetime_reports
            for row in report.birth_rows()
        ]
        free_request_rows = [
            {"comparison": name, **row}
            for name, report in lifetime_reports
            for row in report.free_request_rows()
        ]
        free_completion_rows = [
            {"comparison": name, **row}
            for name, report in lifetime_reports
            for row in report.free_completion_rows()
        ]
        if cohort_rows:
            paths["cohorts"] = _write_csv(root / "cohorts.csv", cohort_rows)
            paths["cohort_points"] = _write_csv(
                root / "cohort_points.csv", cohort_point_rows
            )
            paths["size_histograms"] = _write_csv(
                root / "size_histograms.csv", size_rows
            )
            paths["size_outcomes"] = _write_csv(
                root / "size_outcomes.csv", size_outcome_rows
            )
        if birth_rows:
            paths["birth_stacks"] = _write_csv(root / "birth_stacks.csv", birth_rows)
        if free_request_rows:
            paths["free_request_stacks"] = _write_csv(
                root / "free_request_stacks.csv", free_request_rows
            )
        if free_completion_rows:
            paths["free_completion_stacks"] = _write_csv(
                root / "free_completion_stacks.csv", free_completion_rows
            )
        return paths


@dataclass(frozen=True)
class MemoryRunGroupSummary:
    """Per-rank point states and cross-rank extrema for one run group."""

    run_group: MemoryRunGroup
    rank_points: tuple[MemoryRankPointState, ...]
    point_aggregates: tuple[MemoryRankPointAggregate, ...]
    warnings: tuple[str, ...] = ()

    def to_text(self) -> str:
        lines = [
            f"Memory run group summary {self.run_group.name!r} "
            f"ranks={list(self.run_group.ranks)}",
            "  values are per rank; GPU memory is not summed across ranks",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        current_point: tuple[int, str] | None = None
        for item in self.point_aggregates:
            if item.metric not in _CORE_MEMORY_METRICS:
                continue
            point = (item.point_index, item.point_label)
            if point != current_point:
                lines.append(f"  [{point[0]}] {point[1]}")
                current_point = point
            lines.append(
                "    "
                f"total[{item.scope}] {item.metric}: "
                f"min {format_bytes(item.min_value)} on rank {item.min_rank}, "
                f"max {format_bytes(item.max_value)} on rank {item.max_rank}, "
                f"spread {format_bytes(item.spread_value)}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "run-group-summary",
            "aggregation": "per_rank_extrema_no_sum",
            "run_group": self.run_group.descriptor(),
            "warnings": list(self.warnings),
            "rank_points": [item.to_dict() for item in self.rank_points],
            "point_aggregates": [item.to_dict() for item in self.point_aggregates],
        }

    def to_html(self) -> str:
        aggregate_rows = [
            item.to_dict()
            for item in self.point_aggregates
            if item.metric in _CORE_MEMORY_METRICS
        ]
        rank_rows = []
        for item in self.rank_points:
            row = item.to_dict()
            rank_rows.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "rank",
                            "run_id",
                            "point_index",
                            "point_label",
                            "scope",
                        )
                    },
                    **{metric: row[metric] for metric in _CORE_MEMORY_METRICS},
                }
            )
        return _html_document(
            f"Memory run group summary {self.run_group.name}",
            (
                "<p>Values are per rank; GPU memory is not summed across ranks.</p>",
                "<h2>Cross-Rank Point Summary</h2>",
                _render_table(aggregate_rows, "No point summaries"),
                "<h2>Per-Rank Point States</h2>",
                _render_table(rank_rows, "No rank point states"),
            ),
            self.warnings,
        )

    def write(
        self,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        root = _prepare_output(output_dir, overwrite=overwrite)
        paths = _write_report_documents(
            root,
            text=self.to_text(),
            payload=self.to_dict(),
            html=self.to_html(),
        )
        paths["rank_points"] = _write_csv(
            root / "rank_points.csv",
            [item.to_dict() for item in self.rank_points],
        )
        paths["point_aggregates"] = _write_csv(
            root / "point_aggregates.csv",
            [item.to_dict() for item in self.point_aggregates],
        )
        return paths


@dataclass(frozen=True)
class MemoryRunGroupPhaseComparison:
    """Rank-paired four-point phase equations and cross-rank skew."""

    baseline_group: MemoryRunGroup
    candidate_group: MemoryRunGroup
    rank_comparisons: Mapping[int, MemoryPhaseComparison]
    rank_decomposition: tuple[MemoryRankPhaseDecomposition, ...]
    rank_pool_decomposition: tuple[MemoryRankPoolPhaseDecomposition, ...]
    phase_aggregates: tuple[MemoryRunGroupPhaseAggregate, ...]
    warnings: tuple[str, ...] = ()
    display_stack_depth: int = 2
    display_limit: int = 20

    def rank_decomposition_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_dict()
            for item in self.rank_decomposition
            if include_unchanged or item.changed
        ]

    def rank_pool_decomposition_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_dict()
            for item in self.rank_pool_decomposition
            if include_unchanged or item.changed
        ]

    def phase_aggregate_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_dict()
            for item in self.phase_aggregates
            if include_unchanged or item.changed
        ]

    def to_text(
        self,
        *,
        include_unchanged: bool = True,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> str:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        lines = [
            f"Group phase comparison {self.baseline_group.name!r} vs "
            f"{self.candidate_group.name!r} ranks={list(self.rank_comparisons)}",
            "  values are per rank; GPU memory is not summed across ranks",
            "  end_gap = start_gap + candidate_change - baseline_change",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        selected_rows = [
            row
            for row in self.phase_aggregate_rows(include_unchanged=include_unchanged)
            if row["metric"] in PHASE_METRICS
        ]
        if not selected_rows:
            lines.append("  no changed phase rows")
        for row in selected_rows:
            lines.append(
                "  "
                f"total[{row['scope']}] {row['metric']}: "
                f"end gap min {format_delta_bytes(int(row['end_gap_min_bytes']))} "
                f"on rank {row['end_gap_min_rank']}, max "
                f"{format_delta_bytes(int(row['end_gap_max_bytes']))} "
                f"on rank {row['end_gap_max_rank']} "
                f"(spread {format_bytes(int(row['end_gap_spread_bytes']))}); "
                f"change gap min "
                f"{format_delta_bytes(int(row['change_gap_min_bytes']))} "
                f"on rank {row['change_gap_min_rank']}, max "
                f"{format_delta_bytes(int(row['change_gap_max_bytes']))} "
                f"on rank {row['change_gap_max_rank']} "
                f"(spread {format_bytes(int(row['change_gap_spread_bytes']))})"
            )
        for rank, phase in self.rank_comparisons.items():
            for name, comparison in _phase_components(phase):
                attribution_lines = comparison._attribution_text_lines(
                    indent="      ",
                    limit=limit,
                    stack_depth=stack_depth,
                )
                if not attribution_lines:
                    continue
                lines.append(f"  rank {rank} {name} attribution:")
                lines.extend(attribution_lines)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "kind": "run-group-phase-comparison",
            "aggregation": "per_rank_extrema_no_sum",
            "baseline_group": self.baseline_group.descriptor(),
            "candidate_group": self.candidate_group.descriptor(),
            "display_stack_depth": self.display_stack_depth,
            "display_limit": self.display_limit,
            "warnings": list(self.warnings),
            "rank_decomposition": [item.to_dict() for item in self.rank_decomposition],
            "rank_pool_decomposition": [
                item.to_dict() for item in self.rank_pool_decomposition
            ],
            "phase_aggregates": [item.to_dict() for item in self.phase_aggregates],
            "rank_comparisons": {
                str(rank): comparison.to_dict()
                for rank, comparison in self.rank_comparisons.items()
            },
        }

    def to_html(
        self,
        *,
        include_unchanged: bool = True,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> str:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        stack_rows = []
        event_rows = []
        stack_total = 0
        event_total = 0
        for rank, phase in self.rank_comparisons.items():
            for name, comparison in _phase_components(phase):
                stack_total += len(comparison.allocation_stack_comparisons)
                stack_total += len(comparison.allocation_stack_observation_comparisons)
                event_total += len(comparison.allocator_events)
                stack_rows.extend(
                    {
                        "rank": rank,
                        "comparison": name,
                        **row,
                    }
                    for row in comparison._display_allocation_stack_comparison_rows(
                        limit=limit,
                        stack_depth=stack_depth,
                    )
                )
                event_rows.extend(
                    {
                        "rank": rank,
                        "comparison": name,
                        **row,
                    }
                    for row in comparison._display_event_rows(
                        limit=limit,
                        stack_depth=stack_depth,
                    )
                )
        sections = [
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
            "<h2>Per-Rank Pool Phase Equations</h2>",
            _render_table(
                self.rank_pool_decomposition_rows(include_unchanged=include_unchanged),
                "No rank pool phase rows",
            ),
        ]
        if stack_rows:
            sections.extend(
                [
                    "<h2>Per-Rank Allocation Stacks</h2>",
                    _render_table(
                        stack_rows,
                        "No stack deltas",
                        caption=_showing_caption(len(stack_rows), stack_total),
                    ),
                ]
            )
        if event_rows:
            sections.extend(
                [
                    "<h2>Per-Rank Allocator Events</h2>",
                    _render_table(
                        event_rows,
                        "No allocator events",
                        caption=_showing_caption(len(event_rows), event_total),
                    ),
                ]
            )
        return _html_document(
            f"Group phase comparison {self.baseline_group.name} vs "
            f"{self.candidate_group.name}",
            sections,
            self.warnings,
        )

    def write(
        self,
        output_dir: str | Path,
        *,
        include_unchanged: bool = True,
        limit: int | None = None,
        stack_depth: int | None = None,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        root = _prepare_output(output_dir, overwrite=overwrite)
        paths = _write_report_documents(
            root,
            text=self.to_text(
                include_unchanged=include_unchanged,
                limit=limit,
                stack_depth=stack_depth,
            ),
            payload=self.to_dict(),
            html=self.to_html(
                include_unchanged=include_unchanged,
                limit=limit,
                stack_depth=stack_depth,
            ),
        )
        paths["rank_decomposition"] = _write_csv(
            root / "rank_decomposition.csv",
            self.rank_decomposition_rows(include_unchanged=include_unchanged),
        )
        paths["rank_pool_decomposition"] = _write_csv(
            root / "rank_pool_decomposition.csv",
            self.rank_pool_decomposition_rows(include_unchanged=include_unchanged),
        )
        paths["phase_aggregates"] = _write_csv(
            root / "phase_aggregates.csv",
            self.phase_aggregate_rows(include_unchanged=include_unchanged),
        )
        stack_rows = [
            {"rank": rank, "comparison": name, **row}
            for rank, phase in self.rank_comparisons.items()
            for name, comparison in _phase_components(phase)
            for row in comparison.allocation_stack_comparison_rows()
        ]
        event_rows = [
            {"rank": rank, "comparison": name, **row}
            for rank, phase in self.rank_comparisons.items()
            for name, comparison in _phase_components(phase)
            for row in comparison.event_rows()
        ]
        if stack_rows:
            paths["allocation_stack_comparisons"] = _write_csv(
                root / "allocation_stack_comparisons.csv",
                stack_rows,
            )
        if event_rows:
            paths["events"] = _write_csv(root / "events.csv", event_rows)
        return paths


def _phase_components(
    phase: MemoryPhaseComparison,
) -> tuple[tuple[str, MemoryPointComparison], ...]:
    return (
        ("baseline_change", phase.baseline_change),
        ("candidate_change", phase.candidate_change),
        ("start_gap", phase.start_gap),
        ("end_gap", phase.end_gap),
    )


def _resolve_display_limit(default: int, override: int | None) -> int:
    value = default if override is None else override
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("limit must be an integer >= 1")
    return value


def _resolve_display_stack_depth(default: int, override: int | None) -> int:
    value = default if override is None else override
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("stack_depth must be an integer >= 1")
    return value


def _resolve_display_options(
    *,
    default_limit: int,
    default_stack_depth: int,
    limit: int | None,
    stack_depth: int | None,
) -> tuple[int, int]:
    return (
        _resolve_display_limit(default_limit, limit),
        _resolve_display_stack_depth(default_stack_depth, stack_depth),
    )


def _display_stack_delta_row(
    item: AllocationStackDelta, stack_depth: int
) -> dict[str, object]:
    row = item.to_row()
    row.pop("stack_frames_json")
    row["stack_key"] = item.display_stack(stack_depth)
    return row


def _display_event_row(
    item: AllocatorEventSummary,
    *,
    reference_label: str,
    candidate_label: str,
    stack_depth: int,
) -> dict[str, object]:
    row = item.to_row(
        reference_label=reference_label,
        candidate_label=candidate_label,
    )
    row.pop("stack_frames_json")
    row["stack_key"] = item.display_stack(stack_depth)
    return row


def _pool_text(item: MemoryPoolComparison) -> list[str]:
    reference = item.reference_key.label if item.reference_key else "<none>"
    candidate = item.candidate_key.label if item.candidate_key else "<none>"
    return [
        f"    {reference} -> {candidate} [{item.match}]",
        *_comparison_stats_text(
            item.reference, item.candidate, item.delta, item.lifecycle, indent="      "
        ),
    ]


def _observation_text(item: MemoryObservationComparison) -> list[str]:
    reference = item.reference_key.label if item.reference_key else "<none>"
    candidate = item.candidate_key.label if item.candidate_key else "<none>"
    return [
        f"    {reference} -> {candidate} [{item.match}]",
        *_comparison_stats_text(
            item.reference, item.candidate, item.delta, item.lifecycle, indent="      "
        ),
    ]


def _scope_text(item: MemoryAllocatorScopeComparison) -> list[str]:
    return [
        f"    total[{item.scope}]",
        *_comparison_stats_text(
            item.reference, item.candidate, item.delta, None, indent="      "
        ),
    ]


def _comparison_stats_text(
    reference: Any,
    candidate: Any,
    delta: Any,
    lifecycle: Any | None,
    *,
    indent: str,
) -> list[str]:
    lines = [
        f"{indent}allocated: "
        f"{format_comparison(reference.allocated_bytes, candidate.allocated_bytes, delta.allocated_bytes)}, "
        f"reserved: "
        f"{format_comparison(reference.reserved_bytes, candidate.reserved_bytes, delta.reserved_bytes)}",
        f"{indent}active: "
        f"{format_comparison(reference.active_bytes, candidate.active_bytes, delta.active_bytes)}, "
        f"requested: "
        f"{format_comparison(reference.requested_bytes, candidate.requested_bytes, delta.requested_bytes)}",
    ]
    diagnostics = []
    for label, name, is_bytes in (
        ("inactive", "inactive_bytes", True),
        ("awaiting free", "awaiting_free_bytes", True),
        ("fragmentation", "internal_fragmentation_bytes", True),
        ("segments", "segment_count", False),
        ("blocks", "block_count", False),
        ("inactive blocks", "inactive_block_count", False),
        ("largest inactive block", "largest_inactive_block_bytes", True),
        ("expandable segments", "expandable_segment_count", False),
        ("expandable reserved", "expandable_reserved_bytes", True),
        ("expandable inactive", "expandable_inactive_bytes", True),
    ):
        change = int(getattr(delta, name))
        if not change:
            continue
        before = int(getattr(reference, name))
        after = int(getattr(candidate, name))
        value = (
            format_comparison(before, after, change)
            if is_bytes
            else f"{before} -> {after} (delta {change:+d})"
        )
        diagnostics.append(f"{label}={value}")
    if diagnostics:
        lines.append(f"{indent}diagnostics: " + ", ".join(diagnostics))
    if lifecycle is not None and lifecycle.changed:
        lifecycle_values = ", ".join(
            f"{name.removesuffix('_bytes').replace('_', ' ')}="
            f"{format_bytes(int(value))}"
            for name, value in lifecycle.to_dict().items()
            if value
        )
        lines.append(f"{indent}lifecycle: {lifecycle_values}")
    return lines


def _stack_delta_text(
    item: AllocationStackDelta,
    *,
    indent: str,
    stack_depth: int,
) -> str:
    reference = item.reference_key.label
    candidate = item.candidate_key.label
    pool = reference if reference == candidate else f"{reference} -> {candidate}"
    stream = f" {stream_label(item.stream)}" if item.stream is not None else ""
    return (
        f"{indent}{pool}{stream} "
        f"size={format_comparison(item.reference_size_bytes, item.candidate_size_bytes, item.delta_size_bytes)}, "
        f"requested={format_comparison(item.reference_requested_bytes, item.candidate_requested_bytes, item.delta_requested_bytes)}, "
        f"count={item.reference_count} -> {item.candidate_count} "
        f"(delta {item.delta_count:+d}) at "
        f"{item.display_stack(stack_depth)}"
    )


def _timeline_stats_text(
    label: str,
    stats: MemoryStats,
    delta: MemoryStatsDelta | None,
    *,
    indent: str,
) -> list[str]:
    def render_bytes(name: str) -> str:
        current = int(getattr(stats, name))
        if delta is None:
            return format_bytes(current)
        return (
            f"{format_bytes(current)} "
            f"(delta {format_delta_bytes(int(getattr(delta, name)))})"
        )

    lines = [
        f"{indent}{label} "
        f"allocated={render_bytes('allocated_bytes')}, "
        f"reserved={render_bytes('reserved_bytes')}, "
        f"active={render_bytes('active_bytes')}, "
        f"requested={render_bytes('requested_bytes')}"
    ]
    diagnostics = []
    for diagnostic_label, name, is_bytes in (
        ("inactive", "inactive_bytes", True),
        ("awaiting free", "awaiting_free_bytes", True),
        ("internal fragmentation", "internal_fragmentation_bytes", True),
        ("segments", "segment_count", False),
        ("blocks", "block_count", False),
        ("inactive blocks", "inactive_block_count", False),
        ("largest inactive block", "largest_inactive_block_bytes", True),
        ("expandable segments", "expandable_segment_count", False),
        ("expandable reserved", "expandable_reserved_bytes", True),
        ("expandable inactive", "expandable_inactive_bytes", True),
    ):
        current = int(getattr(stats, name))
        if delta is None:
            if not current:
                continue
            rendered = format_bytes(current) if is_bytes else str(current)
        else:
            change = int(getattr(delta, name))
            if not change:
                continue
            rendered = (
                f"{format_bytes(current)} (delta {format_delta_bytes(change)})"
                if is_bytes
                else f"{current} (delta {change:+d})"
            )
        diagnostics.append(f"{diagnostic_label}={rendered}")
    if diagnostics:
        lines.append(f"{indent}  diagnostics: " + ", ".join(diagnostics))
    return lines


def _prepare_output(
    output_dir: str | Path,
    *,
    overwrite: bool,
) -> Path:
    return prepare_output_dir(output_dir, overwrite=overwrite)


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
    atomic_write_text(text_path, text + "\n")
    atomic_write_json(json_path, payload)
    atomic_write_text(html_path, html)
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
    return atomic_write_csv(
        path,
        rows,
        empty_fieldnames=("row_type",),
        extrasaction="ignore",
    )


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
    caption {{ color: #4b5563; margin-bottom: 8px; text-align: left; }}
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


def _showing_caption(shown: int, total: int, *, noun: str = "rows") -> str:
    return f"Showing {shown} of {total} {noun}."


def _render_table(
    rows: Sequence[Mapping[str, object]],
    empty: str,
    *,
    caption: str | None = None,
) -> str:
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
    caption_html = f"<caption>{escape(caption)}</caption>" if caption else ""
    return (
        f"<table>{caption_html}<thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


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
    pools = sorted({str(row.get("key", "")) for row in rows})
    lines = []
    for color_index, pool in enumerate(pools):
        points = []
        for row in rows:
            if str(row.get("key", "")) != pool:
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
            "key": row.get("cohort_id", ""),
            "state_active_bytes": row.get("active_bytes", 0),
        }
        for row in rows
    ]
    return _timeline_svg(chart_rows, "state_active_bytes")
