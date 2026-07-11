"""Structured text, JSON, CSV, and HTML reports for memory debugging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from .._reporting import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    prepare_output_dir,
)
from ._pool_identity import DEFAULT_POOL_ID
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
    MemoryDeviceComparison,
    MemoryDevicePhaseDecomposition,
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
        MemoryRankDevicePhaseDecomposition,
        MemoryRankDevicePointState,
        MemoryRankPhaseDecomposition,
        MemoryRankPointAggregate,
        MemoryRankPointState,
        MemoryRankPoolPhaseDecomposition,
        MemoryRunGroup,
        MemoryRunGroupPhaseAggregate,
    )
    from .timeline import (
        MemoryAllocatorScopeTimelineEntry,
        MemoryDeviceTimelineEntry,
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
                f"{item.pool_label} "
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
        """Render the cohort report as text.

        ``limit`` and ``stack_depth`` are presentation-only overrides of the
        stored display defaults and raise ValueError when not integers >= 1.
        """

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
        """Return the complete JSON payload; display limits never truncate it."""

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
        """Render the cohort report as HTML using the text display rules."""

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
        """Write text, JSON, HTML, and CSV artifacts into ``output_dir``.

        Returns a mapping of artifact kind to path and refuses a non-empty
        directory unless ``overwrite=True``. Display limits shape text and
        HTML only; JSON and CSV rows are never truncated.
        """

        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        root, paths = _write_report_documents(
            output_dir,
            overwrite=overwrite,
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
    device_comparisons: tuple[MemoryDeviceComparison, ...] = ()
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

    def _report_tree(self) -> tuple[_ReportTreeNode, ...]:
        return _build_comparison_tree(
            self.device_comparisons,
            self.pool_comparisons,
            self.observation_comparisons,
        )

    def pool_comparison_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in _report_tree_payloads(
                self._report_tree(),
                kind="pool",
                include_unchanged=include_unchanged,
            )
        ]

    def observation_comparison_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in _report_tree_payloads(
                self._report_tree(),
                kind="stream",
                include_unchanged=include_unchanged,
            )
        ]

    def device_comparison_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in _report_tree_payloads(
                self._report_tree(),
                kind="device",
                include_unchanged=include_unchanged,
            )
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
                f"{item.pool_label} "
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
        depth: TreeDepth | None = None,
    ) -> str:
        """Render the comparison as text.

        ``include_unchanged=False`` drops unchanged device/pool/stream rows.
        ``depth`` truncates the rendered tree at "device", "pool", or
        "stream" (ValueError otherwise); ``limit`` and ``stack_depth``
        override the stored display defaults and raise ValueError when not
        integers >= 1.
        """

        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        tree_depth = _resolve_tree_depth(depth)
        reference_label = _state_label(self.reference)
        candidate_label = _state_label(self.candidate)
        lines = [
            f"Memory comparison {reference_label!r} -> {candidate_label!r} "
            f"({_state_scope(self.reference, self.candidate)})"
        ]
        lines.append(f"  address lifecycle: {self.lifecycle_confidence}")
        if any(
            item.reference is not None or item.candidate is not None
            for item in self.device_comparisons
        ):
            lines.append(f"  {_CUDA_SCOPE_NOTE}")
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        lines.append("")
        lines.extend(
            _comparison_tree_lines(
                self.device_comparisons,
                self.pool_comparisons,
                self.observation_comparisons,
                include_unchanged=include_unchanged,
                depth=tree_depth,
            )
        )

        trailer: list[str] = []
        if (
            self.reference_stack_coverage is not None
            and self.candidate_stack_coverage is not None
        ):
            trailer.append(
                "  allocation stack coverage: "
                f"{self.reference_stack_coverage.ratio:.1%} -> "
                f"{self.candidate_stack_coverage.ratio:.1%}"
            )
        trailer.extend(
            self._attribution_text_lines(
                indent="  ",
                limit=limit,
                stack_depth=stack_depth,
            )
        )
        if self.allocation_lifetimes is not None:
            trailer.extend(
                self.allocation_lifetimes.summary_lines(
                    indent="  ",
                    limit=limit,
                    stack_depth=stack_depth,
                )
            )
        if trailer:
            lines.append("")
            lines.extend(trailer)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON payload; display options never filter it."""

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
            "device_comparisons": [item.to_dict() for item in self.device_comparisons],
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
        depth: TreeDepth | None = None,
    ) -> str:
        """Render the comparison as HTML using the text display rules."""

        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        tree_depth = _resolve_tree_depth(depth)
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
        sections = ["<h2>Devices</h2>"]
        if any(
            item.reference is not None or item.candidate is not None
            for item in self.device_comparisons
        ):
            sections.append(f"<p>{escape(_CUDA_SCOPE_NOTE)}</p>")
        sections.extend(
            [
                _render_report_tree_html(
                    self._report_tree(),
                    include_unchanged=include_unchanged,
                    depth=tree_depth,
                    empty=("No devices" if include_unchanged else "No changed devices"),
                ),
                "<h2>Allocator Scopes</h2>",
                _render_table(
                    self.allocator_scope_comparison_rows(
                        include_unchanged=include_unchanged
                    ),
                    "No changed allocator scopes",
                ),
            ]
        )
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
        depth: TreeDepth | None = None,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Write text, JSON, HTML, and CSV artifacts into ``output_dir``.

        Returns a mapping of artifact kind to path and refuses a non-empty
        directory unless ``overwrite=True``. ``include_unchanged`` also
        filters CSV rows; ``limit``/``stack_depth``/``depth`` shape text and
        HTML only, and JSON is never truncated.
        """

        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        tree_depth = _resolve_tree_depth(depth)
        root, paths = _write_common(
            output_dir,
            overwrite=overwrite,
            text=self.to_text(
                include_unchanged=include_unchanged,
                limit=limit,
                stack_depth=stack_depth,
                depth=tree_depth,
            ),
            payload=self.to_dict(),
            html=self.to_html(
                include_unchanged=include_unchanged,
                limit=limit,
                stack_depth=stack_depth,
                depth=tree_depth,
            ),
            allocator_scopes=self.allocator_scope_comparison_rows(
                include_unchanged=include_unchanged
            ),
            pools=self.pool_comparison_rows(include_unchanged=include_unchanged),
            observations=self.observation_comparison_rows(
                include_unchanged=include_unchanged
            ),
        )
        if self.device_comparisons:
            paths["devices"] = _write_csv(
                root / "devices.csv",
                self.device_comparison_rows(include_unchanged=include_unchanged),
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
    device_entries: tuple[MemoryDeviceTimelineEntry, ...] = ()
    allocation_lifetimes: MemoryAllocationLifetimeAnalysis | None = None
    display_stack_depth: int = 2
    display_limit: int = 20

    @property
    def warnings(self) -> tuple[str, ...]:
        """Point warnings prefixed with their point index and label, plus
        deduplicated comparison and lifetime warnings."""

        point_warning_values = {
            warning for point in self.run.points for warning in point.warnings
        }
        values = [
            f"point [{point.index}] {point.label}: {warning}"
            for point in self.run.points
            for warning in point.warnings
        ]
        values.extend(
            warning
            for comparison in self.point_comparisons
            for warning in comparison.warnings
            if warning not in point_warning_values
        )
        if self.allocation_lifetimes is not None:
            values.extend(
                warning
                for warning in self.allocation_lifetimes.warnings
                if warning not in point_warning_values
            )
        return tuple(dict.fromkeys(values))

    def allocator_scope_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.allocator_scope_entries
            if include_unchanged or (item.delta is not None and item.delta.changed)
        ]

    def _report_trees(self) -> tuple[tuple[_ReportTreeNode, ...], ...]:
        pools_by_point: dict[int, list[MemoryPoolTimelineEntry]] = {}
        observations_by_point: dict[int, list[MemoryObservationTimelineEntry]] = {}
        devices_by_point: dict[int, list[MemoryDeviceTimelineEntry]] = {}
        for item in self.pool_entries:
            pools_by_point.setdefault(item.point_index, []).append(item)
        for item in self.observation_entries:
            observations_by_point.setdefault(item.point_index, []).append(item)
        for item in self.device_entries:
            devices_by_point.setdefault(item.point_index, []).append(item)
        return tuple(
            _build_timeline_point_tree(
                devices_by_point.get(point.index, ()),
                pools_by_point.get(point.index, ()),
                observations_by_point.get(point.index, ()),
            )
            for point in self.run.points
        )

    def _tree_rows(
        self, kind: str, *, include_unchanged: bool
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for tree in self._report_trees()
            for item in _report_tree_payloads(
                tree,
                kind=kind,
                include_unchanged=include_unchanged,
            )
        ]

    def pool_rows(self, *, include_unchanged: bool = True) -> list[dict[str, object]]:
        return self._tree_rows("pool", include_unchanged=include_unchanged)

    def observation_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return self._tree_rows("stream", include_unchanged=include_unchanged)

    def device_rows(self, *, include_unchanged: bool = True) -> list[dict[str, object]]:
        return self._tree_rows("device", include_unchanged=include_unchanged)

    def to_text(
        self,
        *,
        include_unchanged: bool = True,
        limit: int | None = None,
        stack_depth: int | None = None,
        depth: TreeDepth | None = None,
    ) -> str:
        """Render every point as text.

        ``include_unchanged=False`` drops unchanged rows, ``depth`` truncates
        the rendered tree at "device", "pool", or "stream" (ValueError
        otherwise), and ``limit``/``stack_depth`` override the stored display
        defaults (ValueError when not integers >= 1).
        """

        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        tree_depth = _resolve_tree_depth(depth)
        lines = [f"CUDA allocator memory timeline {self.run.name!r}"]
        if any(item.sample is not None for item in self.device_entries):
            lines.append(f"  {_CUDA_SCOPE_NOTE}")
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        pools_by_point: dict[int, list[MemoryPoolTimelineEntry]] = {}
        observations_by_point: dict[int, list[MemoryObservationTimelineEntry]] = {}
        devices_by_point: dict[int, list[MemoryDeviceTimelineEntry]] = {}
        for item in self.pool_entries:
            pools_by_point.setdefault(item.point_index, []).append(item)
        for item in self.observation_entries:
            observations_by_point.setdefault(item.point_index, []).append(item)
        for item in self.device_entries:
            devices_by_point.setdefault(item.point_index, []).append(item)
        for point in self.run.points:
            point_lines = _timeline_point_tree_lines(
                devices_by_point.get(point.index, []),
                pools_by_point.get(point.index, []),
                observations_by_point.get(point.index, []),
                include_unchanged=include_unchanged,
                depth=tree_depth,
                indent="  ",
            )
            if point_lines:
                lines.append("")
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
        """Return the complete JSON payload; display options never filter it."""

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
            "device_entries": [item.to_dict() for item in self.device_entries],
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
        depth: TreeDepth | None = None,
    ) -> str:
        """Render the timeline as HTML using the text display rules."""

        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        tree_depth = _resolve_tree_depth(depth)
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
        sections = ["<h2>Devices</h2>"]
        if any(item.sample is not None for item in self.device_entries):
            sections.append(f"<p>{escape(_CUDA_SCOPE_NOTE)}</p>")
        rendered_points = 0
        for point, tree in zip(self.run.points, self._report_trees()):
            if not _prune_report_tree(
                tree,
                include_unchanged=include_unchanged,
                depth=tree_depth,
            ):
                continue
            rendered_points += 1
            sections.extend(
                [
                    f"<h3>[{point.index}] {escape(point.label)}</h3>",
                    _render_report_tree_html(
                        tree,
                        include_unchanged=include_unchanged,
                        depth=tree_depth,
                        empty="No devices",
                    ),
                ]
            )
        if rendered_points == 0:
            sections.append(
                "<p>No devices</p>"
                if include_unchanged
                else "<p>No changed devices</p>"
            )
        sections.extend(
            [
                "<h2>Allocated Memory</h2>",
                _timeline_svg(chart_entries, "state_allocated_bytes"),
                "<h2>Reserved Memory</h2>",
                _timeline_svg(chart_entries, "state_reserved_bytes"),
                "<h2>Active Memory</h2>",
                _timeline_svg(chart_entries, "state_active_bytes"),
                "<h2>Requested Memory</h2>",
                _timeline_svg(chart_entries, "state_requested_bytes"),
                "<h2>Allocator Scopes</h2>",
                _render_table(
                    self.allocator_scope_rows(include_unchanged=include_unchanged),
                    "No memory points",
                ),
            ]
        )
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
        depth: TreeDepth | None = None,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Write text, JSON, HTML, and CSV artifacts into ``output_dir``.

        Returns a mapping of artifact kind to path and refuses a non-empty
        directory unless ``overwrite=True``. ``include_unchanged`` also
        filters CSV rows; JSON is never truncated by display options.
        """

        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        tree_depth = _resolve_tree_depth(depth)
        root, paths = _write_common(
            output_dir,
            overwrite=overwrite,
            text=self.to_text(
                include_unchanged=include_unchanged,
                limit=limit,
                stack_depth=stack_depth,
                depth=tree_depth,
            ),
            payload=self.to_dict(),
            html=self.to_html(
                include_unchanged=include_unchanged,
                limit=limit,
                stack_depth=stack_depth,
                depth=tree_depth,
            ),
            allocator_scopes=self.allocator_scope_rows(
                include_unchanged=include_unchanged
            ),
            pools=self.pool_rows(include_unchanged=include_unchanged),
            observations=self.observation_rows(include_unchanged=include_unchanged),
        )
        if self.device_entries:
            paths["devices"] = _write_csv(
                root / "devices.csv",
                self.device_rows(include_unchanged=include_unchanged),
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
    device_decomposition: tuple[MemoryDevicePhaseDecomposition, ...]
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

    def device_decomposition_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_row()
            for item in self.device_decomposition
            if include_unchanged or item.changed
        ]

    def to_text(
        self,
        *,
        include_unchanged: bool = True,
        limit: int | None = None,
        stack_depth: int | None = None,
    ) -> str:
        """Render the four-point phase equations as text.

        ``include_unchanged=False`` drops unchanged rows; ``limit`` and
        ``stack_depth`` override the stored display defaults and raise
        ValueError when not integers >= 1.
        """

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
        lines.append("  device-wide CUDA Runtime totals:")
        device_rows = self.device_decomposition_rows(
            include_unchanged=include_unchanged
        )
        if not device_rows:
            lines.append("    no complete changed device equations")
        for row in device_rows:
            lines.append(
                "    "
                f"device[{row['baseline_device_index']}] -> "
                f"device[{row['candidate_device_index']}] {row['metric']}: "
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
        """Return the complete JSON payload; display options never filter it."""

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
            "device_decomposition": [
                item.to_dict() for item in self.device_decomposition
            ],
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
        """Render the phase comparison as HTML using the text display rules."""

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
            "<h2>Device Decomposition</h2>",
            _render_table(
                self.device_decomposition_rows(include_unchanged=include_unchanged),
                "No complete device decomposition rows",
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
        """Write text, JSON, HTML, and CSV artifacts into ``output_dir``.

        Returns a mapping of artifact kind to path and refuses a non-empty
        directory unless ``overwrite=True``. ``include_unchanged`` also
        filters CSV rows; JSON is never truncated by display options.
        """

        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
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
        device_comparison_rows = [
            {"comparison": name, **row}
            for name, comparison in phase_comparisons
            for row in comparison.device_comparison_rows(
                include_unchanged=include_unchanged
            )
        ]
        root, paths = _write_common(
            output_dir,
            overwrite=overwrite,
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
        if device_comparison_rows:
            paths["devices"] = _write_csv(
                root / "devices.csv",
                device_comparison_rows,
            )
        if self.device_decomposition:
            paths["device_decomposition"] = _write_csv(
                root / "device_decomposition.csv",
                self.device_decomposition_rows(include_unchanged=include_unchanged),
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
    rank_devices: tuple[MemoryRankDevicePointState, ...]
    point_aggregates: tuple[MemoryRankPointAggregate, ...]
    warnings: tuple[str, ...] = ()

    def to_text(self) -> str:
        """Render per-point cross-rank extrema as text, never summing GPUs."""

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
        if self.rank_devices:
            lines.append(
                "  device-wide CUDA Runtime states are per rank/device and "
                "are not aggregated across ranks"
            )
            for item in self.rank_devices:
                lines.append(
                    f"    rank {item.rank} [{item.point_index}] {item.point_label} "
                    f"device[{item.device_index}]: CUDA used "
                    f"{format_bytes(item.sample.used_bytes)} of "
                    f"{format_bytes(item.sample.total_bytes)}, allocator reserved "
                    f"{format_bytes(item.allocator_reserved_bytes)}, "
                    f"residual "
                    f"{format_bytes(item.cuda_allocator_residual_bytes)}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON payload with every metric row."""

        return {
            "schema": REPORT_SCHEMA,
            "kind": "run-group-summary",
            "aggregation": "per_rank_extrema_no_sum",
            "run_group": self.run_group.descriptor(),
            "warnings": list(self.warnings),
            "rank_points": [item.to_dict() for item in self.rank_points],
            "rank_devices": [item.to_dict() for item in self.rank_devices],
            "point_aggregates": [item.to_dict() for item in self.point_aggregates],
        }

    def to_html(self) -> str:
        """Render cross-rank and per-rank state tables as HTML."""

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
                "<h2>Per-Rank Device States</h2>",
                _render_table(
                    [item.to_dict() for item in self.rank_devices],
                    "No device memory samples",
                ),
            ),
            self.warnings,
        )

    def write(
        self,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Write text, JSON, HTML, and CSV artifacts into ``output_dir``.

        Returns a mapping of artifact kind to path and refuses a non-empty
        directory unless ``overwrite=True``.
        """

        root, paths = _write_report_documents(
            output_dir,
            overwrite=overwrite,
            text=self.to_text(),
            payload=self.to_dict(),
            html=self.to_html(),
        )
        paths["rank_points"] = _write_csv(
            root / "rank_points.csv",
            [item.to_dict() for item in self.rank_points],
        )
        if self.rank_devices:
            paths["rank_devices"] = _write_csv(
                root / "rank_devices.csv",
                [item.to_dict() for item in self.rank_devices],
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
    rank_device_decomposition: tuple[MemoryRankDevicePhaseDecomposition, ...]
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

    def rank_device_decomposition_rows(
        self, *, include_unchanged: bool = True
    ) -> list[dict[str, object]]:
        return [
            item.to_dict()
            for item in self.rank_device_decomposition
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
        """Render cross-rank phase skew and per-rank attribution as text.

        ``include_unchanged=False`` drops unchanged rows; ``limit`` and
        ``stack_depth`` override the stored display defaults and raise
        ValueError when not integers >= 1.
        """

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
        device_rows = self.rank_device_decomposition_rows(
            include_unchanged=include_unchanged
        )
        if device_rows:
            lines.append(
                "  device-wide CUDA Runtime equations are per rank/device "
                "and are not aggregated across ranks"
            )
            for row in device_rows:
                lines.append(
                    "  "
                    f"rank {row['rank']} device[{row['baseline_device_index']}] -> device[{row['candidate_device_index']}] "
                    f"{row['metric']}: end gap "
                    f"{format_delta_bytes(int(row['end_gap_bytes']))}; "
                    f"change gap "
                    f"{format_delta_bytes(int(row['change_gap_bytes']))}"
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
        """Return the complete JSON payload; display options never filter it."""

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
            "rank_device_decomposition": [
                item.to_dict() for item in self.rank_device_decomposition
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
        """Render the group comparison as HTML using the text display rules."""

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
            "<h2>Per-Rank Device Phase Equations</h2>",
            _render_table(
                self.rank_device_decomposition_rows(
                    include_unchanged=include_unchanged
                ),
                "No complete rank device phase rows",
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
        """Write text, JSON, HTML, and CSV artifacts into ``output_dir``.

        Returns a mapping of artifact kind to path and refuses a non-empty
        directory unless ``overwrite=True``. ``include_unchanged`` also
        filters CSV rows; JSON is never truncated by display options.
        """

        limit, stack_depth = _resolve_display_options(
            default_limit=self.display_limit,
            default_stack_depth=self.display_stack_depth,
            limit=limit,
            stack_depth=stack_depth,
        )
        root, paths = _write_report_documents(
            output_dir,
            overwrite=overwrite,
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
        if self.rank_device_decomposition:
            paths["rank_device_decomposition"] = _write_csv(
                root / "rank_device_decomposition.csv",
                self.rank_device_decomposition_rows(
                    include_unchanged=include_unchanged
                ),
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


TreeDepth = Literal["device", "pool", "stream"]
_TREE_DEPTHS: tuple[TreeDepth, ...] = ("device", "pool", "stream")
_CUDA_SCOPE_NOTE = (
    "CUDA scope: device-wide, includes other processes; residual = CUDA used - "
    "allocator reserved. CUDA and allocator measurements are consecutive, not atomic."
)
_RENDERED_MATCH_TAGS = frozenset(
    {"mapped", "pool_mapping", "reference_only", "candidate_only"}
)
_DIAGNOSTIC_METRICS: tuple[tuple[str, str, bool], ...] = (
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
)


def _resolve_tree_depth(value: TreeDepth | None) -> TreeDepth:
    resolved = "stream" if value is None else value
    if resolved not in _TREE_DEPTHS:
        raise ValueError("depth must be one of 'device', 'pool', or 'stream'")
    return cast(TreeDepth, resolved)


def _match_tag(match: str) -> str:
    return f" [{match}]" if match in _RENDERED_MATCH_TAGS else ""


def _tree_pool_label(pool_id: Any) -> str:
    scope = "default" if tuple(pool_id) == DEFAULT_POOL_ID else "private"
    return f"{pool_id_label(pool_id)} ({scope})"


def _format_delta_count(value: int) -> str:
    return f"{value:+d}" if value else "0"


def _lifecycle_line(lifecycle: Any | None, *, indent: str) -> list[str]:
    if lifecycle is None or not lifecycle.changed:
        return []
    lifecycle_values = ", ".join(
        f"{name.removesuffix('_bytes').replace('_', ' ')}={format_bytes(int(value))}"
        for name, value in lifecycle.to_dict().items()
        if value
    )
    return [f"{indent}lifecycle: {lifecycle_values}"]


def _comparison_stat_block(
    reference: Any,
    candidate: Any,
    delta: Any,
    lifecycle: Any | None,
    *,
    indent: str,
) -> list[str]:
    lines = [
        f"{indent}reserved: "
        f"{format_comparison(reference.reserved_bytes, candidate.reserved_bytes, delta.reserved_bytes)}",
        f"{indent}allocated: "
        f"{format_comparison(reference.allocated_bytes, candidate.allocated_bytes, delta.allocated_bytes)}, "
        f"active: "
        f"{format_comparison(reference.active_bytes, candidate.active_bytes, delta.active_bytes)}, "
        f"requested: "
        f"{format_comparison(reference.requested_bytes, candidate.requested_bytes, delta.requested_bytes)}",
    ]
    diagnostics = []
    for label, name, is_bytes in _DIAGNOSTIC_METRICS:
        change = int(getattr(delta, name))
        if not change:
            continue
        before = int(getattr(reference, name))
        after = int(getattr(candidate, name))
        value = (
            format_comparison(before, after, change)
            if is_bytes
            else f"{before} -> {after} ({_format_delta_count(change)})"
        )
        diagnostics.append(f"{label}={value}")
    if diagnostics:
        lines.append(f"{indent}diagnostics: " + ", ".join(diagnostics))
    lines.extend(_lifecycle_line(lifecycle, indent=indent))
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
        f"({_format_delta_count(item.delta_count)}) at "
        f"{item.display_stack(stack_depth)}"
    )


@dataclass
class _ReportTreeNode:
    """Private report tree shared by comparisons and timelines."""

    kind: str
    identity: tuple[object, ...]
    label: str
    details: tuple[str, ...] = ()
    payload: Any | None = None
    changed_self: bool = False
    children: list["_ReportTreeNode"] = field(default_factory=list)

    @property
    def changed_subtree(self) -> bool:
        return self.changed_self or any(
            child.changed_subtree for child in self.children
        )


def _tree_kind_visible(kind: str, depth: TreeDepth) -> bool:
    if kind == "stream":
        return depth == "stream"
    if kind == "pool":
        return depth in ("pool", "stream")
    return True


def _prune_report_tree(
    nodes: Sequence[_ReportTreeNode],
    *,
    include_unchanged: bool,
    depth: TreeDepth,
) -> tuple[_ReportTreeNode, ...]:
    """Prune once, bottom-up, while retaining changed descendants' parents."""

    def prune(node: _ReportTreeNode) -> _ReportTreeNode | None:
        if not _tree_kind_visible(node.kind, depth):
            return None
        children = [
            selected
            for child in node.children
            if (selected := prune(child)) is not None
        ]
        # Children hidden by the depth cutoff must still contribute their
        # changed signal, or a node whose only changes live below the cutoff
        # would be pruned as unchanged.
        changed_self = node.changed_self or any(
            child.changed_subtree
            for child in node.children
            if not _tree_kind_visible(child.kind, depth)
        )
        if not include_unchanged and not changed_self and not children:
            return None
        return _ReportTreeNode(
            kind=node.kind,
            identity=node.identity,
            label=node.label,
            details=node.details,
            payload=node.payload,
            changed_self=changed_self,
            children=children,
        )

    return tuple(selected for node in nodes if (selected := prune(node)) is not None)


def _report_tree_payloads(
    nodes: Sequence[_ReportTreeNode],
    *,
    kind: str,
    include_unchanged: bool,
) -> tuple[Any, ...]:
    selected = _prune_report_tree(
        nodes,
        include_unchanged=include_unchanged,
        depth="stream",
    )
    payloads: list[Any] = []

    def visit(node: _ReportTreeNode) -> None:
        if node.kind == kind and node.payload is not None:
            payloads.append(node.payload)
        for child in node.children:
            visit(child)

    for node in selected:
        visit(node)
    return tuple(payloads)


def _render_report_tree_text(
    nodes: Sequence[_ReportTreeNode],
    *,
    include_unchanged: bool,
    depth: TreeDepth,
    indent: str = "  ",
    empty: str,
) -> list[str]:
    selected = _prune_report_tree(
        nodes,
        include_unchanged=include_unchanged,
        depth=depth,
    )
    if not selected:
        return [f"{indent}{empty}"]
    lines: list[str] = []

    def render(node: _ReportTreeNode, level: int) -> None:
        prefix = indent + "  " * level
        lines.append(f"{prefix}{node.label}")
        lines.extend(f"{prefix}  {detail}" for detail in node.details)
        for child in node.children:
            render(child, level + 1)

    for node in selected:
        render(node, 0)
    return lines


def _render_report_tree_html(
    nodes: Sequence[_ReportTreeNode],
    *,
    include_unchanged: bool,
    depth: TreeDepth,
    empty: str,
) -> str:
    selected = _prune_report_tree(
        nodes,
        include_unchanged=include_unchanged,
        depth=depth,
    )
    if not selected:
        return f"<p>{escape(empty)}</p>"

    def render(node: _ReportTreeNode) -> str:
        details = "".join(
            f'<div class="tree-detail">{escape(detail)}</div>'
            for detail in node.details
        )
        children = (
            '<ul class="memory-tree">'
            + "".join(render(child) for child in node.children)
            + "</ul>"
            if node.children
            else ""
        )
        return (
            f'<li data-kind="{escape(node.kind)}">'
            f'<div class="tree-label">{escape(node.label)}</div>'
            f"{details}{children}</li>"
        )

    return (
        '<ul class="memory-tree">'
        + "".join(render(node) for node in selected)
        + "</ul>"
    )


def _pair_value_line(
    label: str,
    reference: int | None,
    candidate: int | None,
) -> str:
    if reference is None:
        return f"{label}: n/a -> {format_bytes(cast(int, candidate))}"
    if candidate is None:
        return f"{label}: {format_bytes(reference)} -> n/a"
    return f"{label}: {format_comparison(reference, candidate, candidate - reference)}"


def _device_pair_label(row: MemoryDeviceComparison) -> str:
    reference = row.reference_device_index
    candidate = row.candidate_device_index
    if reference is None:
        return f"device[{candidate}] [candidate_only]"
    if candidate is None:
        return f"device[{reference}] [reference_only]"
    label = f"device[{reference}]"
    if reference != candidate:
        label += f" -> device[{candidate}]"
    return label + _match_tag(row.match)


def _comparison_sample_changed(row: MemoryDeviceComparison) -> bool:
    if (row.reference is None) != (row.candidate is None):
        return True
    return bool(row.delta_used_bytes or row.delta_free_bytes or row.delta_total_bytes)


def _comparison_pool_node(
    row: MemoryPoolComparison,
    observations: Sequence[MemoryObservationComparison],
) -> _ReportTreeNode:
    reference_key = row.reference_key
    candidate_key = row.candidate_key
    if (
        reference_key is not None
        and candidate_key is not None
        and reference_key.pool_id != candidate_key.pool_id
    ):
        label = (
            f"{_tree_pool_label(reference_key.pool_id)} -> "
            f"{_tree_pool_label(candidate_key.pool_id)}"
        )
    else:
        key = reference_key if reference_key is not None else candidate_key
        label = _tree_pool_label(cast(Any, key).pool_id)
    node = _ReportTreeNode(
        kind="pool",
        identity=(reference_key, candidate_key),
        label=label + _match_tag(row.match),
        details=tuple(
            _comparison_stat_block(
                row.reference,
                row.candidate,
                row.delta,
                row.lifecycle,
                indent="",
            )
        ),
        payload=row,
        changed_self=row.changed,
    )
    for observation in observations:
        key = observation.reference_key or observation.candidate_key
        node.children.append(
            _ReportTreeNode(
                kind="stream",
                identity=(observation.reference_key, observation.candidate_key),
                label=stream_label(cast(Any, key).stream)
                + _match_tag(observation.match),
                details=tuple(
                    _comparison_stat_block(
                        observation.reference,
                        observation.candidate,
                        observation.delta,
                        observation.lifecycle,
                        indent="",
                    )
                ),
                payload=observation,
                changed_self=observation.changed,
            )
        )
    return node


def _build_comparison_tree(
    device_rows: Sequence[MemoryDeviceComparison],
    pool_rows: Sequence[MemoryPoolComparison],
    observation_rows: Sequence[MemoryObservationComparison],
) -> tuple[_ReportTreeNode, ...]:
    observations_by_reference: dict[Any, list[MemoryObservationComparison]] = {}
    observations_by_candidate: dict[Any, list[MemoryObservationComparison]] = {}
    for observation in observation_rows:
        if observation.reference_key is not None:
            observations_by_reference.setdefault(
                observation.reference_key.pool_key, []
            ).append(observation)
        if observation.candidate_key is not None:
            observations_by_candidate.setdefault(
                observation.candidate_key.pool_key, []
            ).append(observation)

    pool_nodes: list[tuple[MemoryPoolComparison, _ReportTreeNode]] = []
    for pool in pool_rows:
        # Independent comparisons produce one-sided observation rows, so a
        # matched pool needs the union of both indexes, not just the
        # reference side.
        observations: list[MemoryObservationComparison] = []
        seen: set[int] = set()
        if pool.reference_key is not None:
            for observation in observations_by_reference.get(pool.reference_key, []):
                seen.add(id(observation))
                observations.append(observation)
        if pool.candidate_key is not None:
            observations.extend(
                observation
                for observation in observations_by_candidate.get(pool.candidate_key, [])
                if id(observation) not in seen
            )
        pool_nodes.append((pool, _comparison_pool_node(pool, observations)))

    roots: list[
        tuple[MemoryDeviceComparison | None, _ReportTreeNode, _ReportTreeNode]
    ] = []
    for row in device_rows:
        root = _ReportTreeNode(
            kind="device",
            identity=(row.reference_device_index, row.candidate_device_index),
            label=_device_pair_label(row),
            payload=row,
            changed_self=_comparison_sample_changed(row),
        )
        allocator = _ReportTreeNode(
            kind="allocator",
            identity=("allocator", *root.identity),
            label="allocator:",
            details=tuple(
                _comparison_stat_block(
                    row.reference_allocator,
                    row.candidate_allocator,
                    row.allocator_delta,
                    None,
                    indent="",
                )
            ),
            changed_self=row.allocator_delta.changed,
        )
        sampled = row.reference is not None or row.candidate is not None
        if sampled:
            total = _ReportTreeNode(
                kind="cuda_total",
                identity=("cuda_total", *root.identity),
                label=_pair_value_line(
                    "CUDA total",
                    row.reference.total_bytes if row.reference else None,
                    row.candidate.total_bytes if row.candidate else None,
                ),
                changed_self=(
                    (row.reference is None) != (row.candidate is None)
                    or bool(row.delta_total_bytes)
                ),
            )
            residual = _ReportTreeNode(
                kind="residual",
                identity=("residual", *root.identity),
                label=_pair_value_line(
                    "residual",
                    row.reference_cuda_allocator_residual_bytes,
                    row.candidate_cuda_allocator_residual_bytes,
                ),
                changed_self=(
                    (row.reference is None) != (row.candidate is None)
                    or bool(row.delta_cuda_allocator_residual_bytes)
                ),
            )
            used = _ReportTreeNode(
                kind="cuda_used",
                identity=("cuda_used", *root.identity),
                label=_pair_value_line(
                    "CUDA used",
                    row.reference.used_bytes if row.reference else None,
                    row.candidate.used_bytes if row.candidate else None,
                ),
                changed_self=(
                    (row.reference is None) != (row.candidate is None)
                    or bool(row.delta_used_bytes)
                ),
                children=[residual, allocator],
            )
            root.children.extend((total, used))
        else:
            root.children.append(allocator)
        roots.append((row, root, allocator))

    for pool, pool_node in pool_nodes:
        reference_device = (
            pool.reference_key.device_index if pool.reference_key is not None else None
        )
        candidate_device = (
            pool.candidate_key.device_index if pool.candidate_key is not None else None
        )
        parent = next(
            (
                allocator
                for _row, root, allocator in roots
                if (
                    (reference_device is None or root.identity[0] == reference_device)
                    and (
                        candidate_device is None or root.identity[1] == candidate_device
                    )
                )
            ),
            None,
        )
        if parent is None:
            if reference_device is None:
                label = f"device[{candidate_device}] [candidate_only]"
            elif candidate_device is None:
                label = f"device[{reference_device}] [reference_only]"
            elif reference_device == candidate_device:
                label = f"device[{reference_device}]"
            else:
                label = f"device[{reference_device}] -> device[{candidate_device}]"
            root = _ReportTreeNode(
                kind="device",
                identity=(reference_device, candidate_device),
                label=label,
            )
            parent = _ReportTreeNode(
                kind="allocator",
                identity=("allocator", *root.identity),
                label="allocator:",
            )
            root.children.append(parent)
            roots.append((None, root, parent))
        parent.children.append(pool_node)

    for row, _root, allocator in roots:
        if row is not None:
            continue
        reference_allocator = MemoryStats.combine(
            child.payload.reference for child in allocator.children
        )
        candidate_allocator = MemoryStats.combine(
            child.payload.candidate for child in allocator.children
        )
        delta = MemoryStatsDelta.between(
            reference_allocator,
            candidate_allocator,
        )
        allocator.details = tuple(
            _comparison_stat_block(
                reference_allocator,
                candidate_allocator,
                delta,
                None,
                indent="",
            )
        )
        allocator.changed_self = delta.changed

    return tuple(root for _row, root, _allocator in roots)


def _timeline_value_line(label: str, current: int, delta: int | None) -> str:
    if delta is None:
        return f"{label}: {format_bytes(current)}"
    return f"{label}: {format_bytes(current)} ({format_delta_bytes(delta)})"


def _timeline_stat_block(
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
            f"{format_bytes(current)} ({format_delta_bytes(int(getattr(delta, name)))})"
        )

    lines = [
        f"{indent}reserved: {render_bytes('reserved_bytes')}",
        f"{indent}allocated: {render_bytes('allocated_bytes')}, "
        f"active: {render_bytes('active_bytes')}, "
        f"requested: {render_bytes('requested_bytes')}",
    ]
    diagnostics = []
    for diagnostic_label, name, is_bytes in _DIAGNOSTIC_METRICS:
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
                f"{format_bytes(current)} ({format_delta_bytes(change)})"
                if is_bytes
                else f"{current} ({_format_delta_count(change)})"
            )
        diagnostics.append(f"{diagnostic_label}={rendered}")
    if diagnostics:
        lines.append(f"{indent}diagnostics: " + ", ".join(diagnostics))
    return lines


def _timeline_entry_changed(item: Any) -> bool:
    return item.delta is not None and item.delta.changed


def _timeline_device_changed(item: MemoryDeviceTimelineEntry) -> bool:
    return bool(
        item.delta_used_bytes or item.delta_free_bytes or item.delta_total_bytes
    )


def _build_timeline_point_tree(
    device_entries: Sequence[MemoryDeviceTimelineEntry],
    pool_entries: Sequence[MemoryPoolTimelineEntry],
    observation_entries: Sequence[MemoryObservationTimelineEntry],
) -> tuple[_ReportTreeNode, ...]:
    observations_by_pool: dict[Any, list[MemoryObservationTimelineEntry]] = {}
    for observation in observation_entries:
        observations_by_pool.setdefault(observation.key.pool_key, []).append(
            observation
        )
    pools_by_device: dict[int, list[_ReportTreeNode]] = {}
    for pool in pool_entries:
        pool_node = _ReportTreeNode(
            kind="pool",
            identity=(pool.point_index, pool.key),
            label=_tree_pool_label(pool.key.pool_id),
            details=tuple(_timeline_stat_block(pool.stats, pool.delta, indent="")),
            payload=pool,
            changed_self=_timeline_entry_changed(pool),
        )
        for observation in observations_by_pool.get(pool.key, []):
            pool_node.children.append(
                _ReportTreeNode(
                    kind="stream",
                    identity=(observation.point_index, observation.key),
                    label=stream_label(observation.key.stream),
                    details=tuple(
                        _timeline_stat_block(
                            observation.stats,
                            observation.delta,
                            indent="",
                        )
                    ),
                    payload=observation,
                    changed_self=_timeline_entry_changed(observation),
                )
            )
        pools_by_device.setdefault(pool.key.device_index, []).append(pool_node)

    roots: list[_ReportTreeNode] = []
    for entry in device_entries:
        root = _ReportTreeNode(
            kind="device",
            identity=(entry.point_index, entry.device_index),
            label=f"device[{entry.device_index}]",
            payload=entry,
            changed_self=_timeline_device_changed(entry),
        )
        allocator = _ReportTreeNode(
            kind="allocator",
            identity=(entry.point_index, entry.device_index, "allocator"),
            label="allocator:",
            details=tuple(_timeline_stat_block(entry.stats, entry.delta, indent="")),
            changed_self=(entry.delta is not None and entry.delta.changed),
            children=pools_by_device.pop(entry.device_index, []),
        )
        if entry.sample is not None:
            total = _ReportTreeNode(
                kind="cuda_total",
                identity=(entry.point_index, entry.device_index, "cuda_total"),
                label=_timeline_value_line(
                    "CUDA total",
                    entry.sample.total_bytes,
                    entry.delta_total_bytes,
                ),
                changed_self=bool(entry.delta_total_bytes),
            )
            residual = _ReportTreeNode(
                kind="residual",
                identity=(entry.point_index, entry.device_index, "residual"),
                label=_timeline_value_line(
                    "residual",
                    cast(int, entry.cuda_allocator_residual_bytes),
                    entry.delta_cuda_allocator_residual_bytes,
                ),
                changed_self=bool(entry.delta_cuda_allocator_residual_bytes),
            )
            used = _ReportTreeNode(
                kind="cuda_used",
                identity=(entry.point_index, entry.device_index, "cuda_used"),
                label=_timeline_value_line(
                    "CUDA used", entry.sample.used_bytes, entry.delta_used_bytes
                ),
                changed_self=bool(entry.delta_used_bytes),
                children=[residual, allocator],
            )
            root.children.extend((total, used))
        else:
            root.children.append(allocator)
        roots.append(root)

    for device, pools in sorted(pools_by_device.items()):
        allocator = _ReportTreeNode(
            kind="allocator",
            identity=(pool_entries[0].point_index, device, "allocator"),
            label="allocator:",
            changed_self=any(pool.changed_subtree for pool in pools),
            children=pools,
        )
        roots.append(
            _ReportTreeNode(
                kind="device",
                identity=(pool_entries[0].point_index, device),
                label=f"device[{device}]",
                children=[allocator],
            )
        )
    return tuple(roots)


def _comparison_tree_lines(
    device_rows: Sequence[MemoryDeviceComparison],
    pool_rows: Sequence[MemoryPoolComparison],
    observation_rows: Sequence[MemoryObservationComparison],
    *,
    include_unchanged: bool,
    depth: TreeDepth,
) -> list[str]:
    return _render_report_tree_text(
        _build_comparison_tree(device_rows, pool_rows, observation_rows),
        include_unchanged=include_unchanged,
        depth=depth,
        empty="no devices" if include_unchanged else "no changed devices",
    )


def _timeline_point_tree_lines(
    device_entries: Sequence[MemoryDeviceTimelineEntry],
    pool_entries: Sequence[MemoryPoolTimelineEntry],
    observation_entries: Sequence[MemoryObservationTimelineEntry],
    *,
    include_unchanged: bool,
    depth: TreeDepth,
    indent: str,
) -> list[str]:
    tree = _build_timeline_point_tree(
        device_entries,
        pool_entries,
        observation_entries,
    )
    if not _prune_report_tree(
        tree,
        include_unchanged=include_unchanged,
        depth=depth,
    ):
        return []
    return _render_report_tree_text(
        tree,
        include_unchanged=include_unchanged,
        depth=depth,
        indent=indent,
        empty="no devices",
    )


def _write_report_documents(
    output_dir: str | Path,
    *,
    overwrite: bool,
    text: str,
    payload: Mapping[str, object],
    html: str,
) -> tuple[Path, dict[str, Path]]:
    """Prepare the output directory only after every document argument has
    been rendered, so a rendering failure cannot destroy a previous report."""

    root = prepare_output_dir(output_dir, overwrite=overwrite)
    text_path = root / "report.txt"
    json_path = root / "report.json"
    html_path = root / "report.html"
    atomic_write_text(text_path, text + "\n")
    atomic_write_json(json_path, payload)
    atomic_write_text(html_path, html)
    return root, {"text": text_path, "json": json_path, "html": html_path}


def _write_common(
    output_dir: str | Path,
    *,
    overwrite: bool,
    text: str,
    payload: Mapping[str, object],
    html: str,
    allocator_scopes: Sequence[Mapping[str, object]],
    pools: Sequence[Mapping[str, object]],
    observations: Sequence[Mapping[str, object]],
) -> tuple[Path, dict[str, Path]]:
    root, paths = _write_report_documents(
        output_dir,
        overwrite=overwrite,
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
    return root, paths


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
    .memory-tree {{ list-style: none; margin: 8px 0 0; padding-left: 22px; border-left: 1px solid #d1d5db; }}
    .memory-tree > li {{ margin: 8px 0; }}
    .tree-label {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 600; }}
    .tree-detail {{ margin: 3px 0 0 14px; color: #374151; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
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
