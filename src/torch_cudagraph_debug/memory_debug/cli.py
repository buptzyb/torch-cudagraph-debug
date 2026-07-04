"""Command-line interface for offline CUDA allocator memory analysis."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .attribution import MemoryAttributionOptions, MemoryLifetimeOptions
from .comparison import compare_phases, compare_points
from .recording import MemoryRun
from .errors import MemoryDebugError
from .run_groups import MemoryRunGroup, compare_run_group_phases
from .allocator_snapshot import format_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcgd-memory",
        description="Analyze torch-cudagraph-debug memory run bundles.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    summary = commands.add_parser("summary", help="Summarize one memory run")
    summary.add_argument("bundle")

    timeline = commands.add_parser(
        "timeline", help="Report every point in one memory run"
    )
    timeline.add_argument("bundle")
    _add_attribution_options(timeline)
    lifetime_analysis = commands.add_parser(
        "allocation-lifetimes",
        help="Trace active allocation cohorts across one memory run",
    )
    lifetime_analysis.add_argument("bundle")
    lifetime_selection = lifetime_analysis.add_mutually_exclusive_group()
    lifetime_selection.add_argument(
        "--at",
        help="Only trace allocation instances active at this point",
    )
    lifetime_selection.add_argument(
        "--born-between",
        nargs=2,
        metavar=("START", "END"),
        help="Trace allocation instances born in the marker interval (START, END]",
    )
    lifetime_analysis.add_argument(
        "--through",
        help="Stop tracing at this point instead of the final point",
    )
    lifetime_analysis.add_argument("--output", type=Path, required=True)
    lifetime_analysis.add_argument(
        "--no-events",
        action="store_false",
        dest="events",
        help="Skip allocator event pairing and use snapshots only",
    )
    lifetime_analysis.add_argument(
        "--on-missing",
        choices=("warn", "error"),
        default="warn",
    )
    lifetime_analysis.add_argument("--stack-depth", type=int, default=4)
    lifetime_analysis.add_argument("--limit", type=int, default=20)
    lifetime_analysis.set_defaults(events=True)

    compare = commands.add_parser(
        "compare-points",
        help="Compare one reference point with one candidate point",
    )
    compare.add_argument(
        "reference_bundle", help="Bundle containing the reference point"
    )
    compare.add_argument(
        "candidate_bundle",
        nargs="?",
        help="Bundle containing the candidate point; defaults to the reference bundle",
    )
    compare.add_argument("--reference-point", required=True)
    compare.add_argument("--candidate-point", required=True)
    compare.add_argument(
        "--pool-map",
        action="append",
        default=[],
        metavar="REFERENCE=CANDIDATE",
        help="Explicit private-pool mapping, for example 0,1=0,2",
    )
    _add_attribution_options(compare)

    phases = commands.add_parser(
        "compare-phases",
        help="Decompose baseline and candidate phase changes using four points",
    )
    phases.add_argument("baseline_bundle")
    phases.add_argument("candidate_bundle")
    phases.add_argument("--baseline-start", required=True)
    phases.add_argument("--baseline-end", required=True)
    phases.add_argument("--candidate-start", required=True)
    phases.add_argument("--candidate-end", required=True)
    phases.add_argument(
        "--pool-map",
        action="append",
        default=[],
        metavar="BASELINE=CANDIDATE",
        help="Explicit private-pool mapping, for example 0,1=0,2",
    )
    _add_attribution_options(phases)

    group_summary = commands.add_parser(
        "summarize-run-group",
        help="Summarize compatible per-rank memory bundles without summing GPUs",
    )
    group_summary.add_argument("group_dir")
    group_summary.add_argument("--output", type=Path, required=True)

    group_phases = commands.add_parser(
        "compare-run-group-phases",
        help="Compare four-point phase memory rank by rank across two run groups",
    )
    group_phases.add_argument("baseline_group")
    group_phases.add_argument("candidate_group")
    group_phases.add_argument("--baseline-start", required=True)
    group_phases.add_argument("--baseline-end", required=True)
    group_phases.add_argument("--candidate-start", required=True)
    group_phases.add_argument("--candidate-end", required=True)
    _add_attribution_options(group_phases)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "summary":
            print(_summary_text(_load_run(args.bundle)))
            return 0
        if args.command == "timeline":
            options = _attribution_from_args(args)
            result = _load_run(args.bundle).timeline(attribution=options)
        elif args.command == "allocation-lifetimes":
            run = _load_run(args.bundle)
            result = run.lifetimes(
                args.at,
                born_between=(
                    tuple(args.born_between) if args.born_between is not None else None
                ),
                through=args.through,
                options=MemoryLifetimeOptions(
                    events=args.events,
                    on_missing=args.on_missing,
                    stack_depth=args.stack_depth,
                    limit=args.limit,
                ),
            )
        elif args.command == "compare-points":
            options = _attribution_from_args(args)
            reference = _load_run(args.reference_bundle)
            candidate = _load_run(args.candidate_bundle or args.reference_bundle)
            pool_mapping = _parse_pool_mappings(args.pool_map)
            if reference.run_id == candidate.run_id:
                if pool_mapping:
                    raise ValueError("--pool-map is only valid across independent runs")
                result = reference.compare(
                    args.reference_point,
                    args.candidate_point,
                    attribution=options,
                )
            else:
                result = compare_points(
                    reference.point(args.reference_point),
                    candidate.point(args.candidate_point),
                    pool_mapping=pool_mapping,
                    attribution=options,
                )
        elif args.command == "compare-phases":
            options = _attribution_from_args(args)
            baseline = _load_run(args.baseline_bundle)
            candidate = _load_run(args.candidate_bundle)
            result = compare_phases(
                baseline.between(args.baseline_start, args.baseline_end),
                candidate.between(args.candidate_start, args.candidate_end),
                pool_mapping=_parse_pool_mappings(args.pool_map),
                attribution=options,
            )
        elif args.command == "summarize-run-group":
            result = MemoryRunGroup.load(args.group_dir).summary()
        elif args.command == "compare-run-group-phases":
            options = _attribution_from_args(args)
            baseline_group = MemoryRunGroup.load(args.baseline_group)
            candidate_group = MemoryRunGroup.load(args.candidate_group)
            result = compare_run_group_phases(
                baseline_group,
                candidate_group,
                baseline_start=args.baseline_start,
                baseline_end=args.baseline_end,
                candidate_start=args.candidate_start,
                candidate_end=args.candidate_end,
                attribution=options,
            )
        else:
            raise ValueError(f"unknown command {args.command!r}")

        include_unchanged = not getattr(args, "only_changed", False)
        if args.command in {"allocation-lifetimes", "summarize-run-group"}:
            paths = result.write(args.output)
            print(result.to_text())
        else:
            paths = result.write(
                args.output,
                include_unchanged=include_unchanged,
            )
            print(result.to_text(include_unchanged=include_unchanged))
        for kind, path in paths.items():
            print(f"{kind}: {path.resolve()}")
        return 0
    except (
        MemoryDebugError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        IndexError,
    ) as exc:
        print(f"tcgd-memory: error: {exc}", file=sys.stderr)
        return 2


def _summary_text(run: MemoryRun) -> str:
    lines = [
        f"Memory run {run.name!r}",
        f"  complete={run.complete} points={len(run.points)} "
        f"rank={run.rank} world_size={run.world_size}",
    ]
    for point in run.points:
        total = point.allocator_scope_stats["all"]
        lines.append(
            f"  [{point.index}] {point.label}: "
            f"observations={len(point.observations)} pools={len(point.pool_stats)} "
            f"allocated={format_bytes(total.allocated_bytes)} "
            f"reserved={format_bytes(total.reserved_bytes)} "
            f"active={format_bytes(total.active_bytes)} "
            f"requested={format_bytes(total.requested_bytes)}"
        )
    return "\n".join(lines)


def _load_run(bundle: str) -> MemoryRun:
    return MemoryRun.load(bundle, cache_snapshots=False)


def _add_attribution_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stacks", action="store_true")
    parser.add_argument("--events", action="store_true")
    parser.add_argument("--lifetimes", action="store_true")
    parser.add_argument(
        "--on-missing",
        choices=("warn", "error"),
        default="warn",
    )
    parser.add_argument("--stack-depth", type=int, default=2)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="Omit unchanged allocator, pool, and pool/stream rows from text, HTML, and CSV",
    )


def _attribution_from_args(args: argparse.Namespace) -> MemoryAttributionOptions:
    return MemoryAttributionOptions(
        stacks=args.stacks,
        events=args.events,
        lifetimes=args.lifetimes,
        on_missing=args.on_missing,
        stack_depth=args.stack_depth,
        limit=args.limit,
    )


def _parse_pool_mappings(
    values: Sequence[str],
) -> Mapping[tuple[Any, ...], tuple[Any, ...]]:
    result: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for value in values:
        if value.count("=") != 1:
            raise ValueError(
                f"invalid pool mapping {value!r}; expected REFERENCE=CANDIDATE"
            )
        reference_text, candidate_text = value.split("=", 1)
        reference = _parse_pool_id(reference_text)
        candidate = _parse_pool_id(candidate_text)
        if reference in result:
            raise ValueError(f"pool {reference_text!r} is mapped more than once")
        result[reference] = candidate
    return result


def _parse_pool_id(value: str) -> tuple[Any, ...]:
    if not value:
        raise ValueError("pool ID must be non-empty")
    parts = value.split(",")
    parsed = []
    for part in parts:
        text = part.strip()
        if not text:
            raise ValueError(f"invalid pool ID {value!r}")
        try:
            parsed.append(int(text))
        except ValueError:
            parsed.append(text)
    return tuple(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
