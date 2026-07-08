"""Command-line interface for offline CUDA allocator memory analysis."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from ._pool_identity import MemoryPoolKey
from .allocator_snapshot import format_bytes
from .attribution import (
    MemoryAttributionOptions,
    MemoryDisplayOptions,
    MemoryLifetimeOptions,
)
from .comparison import compare_phases, compare_points
from .errors import MemoryDebugError
from .recording import MemoryLifetimeSelection, MemoryRun
from .run_groups import MemoryRunGroup, compare_run_group_phases


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
        "--active-at",
        help="Only trace allocation generations not reusable at this point",
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
    _add_output_options(lifetime_analysis)
    lifetime_analysis.add_argument("--stack-depth", type=int, default=4)
    lifetime_analysis.add_argument("--limit", type=int, default=20)

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
        help="Explicit private-pool mapping, for example 0:0,1=0:0,2",
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
        help="Explicit private-pool mapping, for example 0:0,1=0:0,2",
    )
    _add_attribution_options(phases)

    group_summary = commands.add_parser(
        "group-summary",
        help="Summarize compatible per-rank memory bundles without summing GPUs",
    )
    group_summary.add_argument("group_dir")
    _add_output_options(group_summary)

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
    group_phases.add_argument(
        "--pool-map",
        action="append",
        default=[],
        metavar="RANK@BASELINE=CANDIDATE",
        help="Per-rank private-pool mapping, for example 0@0:0,1=0:0,2",
    )
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
            if args.born_between is not None:
                selection = MemoryLifetimeSelection.born_between(*args.born_between)
            elif args.active_at is not None:
                selection = MemoryLifetimeSelection.active_at(args.active_at)
            else:
                selection = MemoryLifetimeSelection.all()
            result = run.lifetimes(
                selection,
                through=args.through,
                options=MemoryLifetimeOptions(
                    display=MemoryDisplayOptions(
                        stack_depth=args.stack_depth,
                        limit=args.limit,
                    ),
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
        elif args.command == "group-summary":
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
                pool_mappings=_parse_rank_pool_mappings(args.pool_map),
                attribution=options,
            )
        else:
            raise ValueError(f"unknown command {args.command!r}")

        include_unchanged = not getattr(args, "only_changed", False)
        print(
            result.to_text()
            if args.command in {"allocation-lifetimes", "group-summary"}
            else result.to_text(include_unchanged=include_unchanged)
        )
        if args.output is not None:
            if args.command in {"allocation-lifetimes", "group-summary"}:
                paths = result.write(args.output, overwrite=args.overwrite)
            else:
                paths = result.write(
                    args.output,
                    include_unchanged=include_unchanged,
                    overwrite=args.overwrite,
                )
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


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace report artifacts in an existing non-empty output directory",
    )


def _add_attribution_options(parser: argparse.ArgumentParser) -> None:
    _add_output_options(parser)
    parser.add_argument("--stacks", action="store_true")
    parser.add_argument("--events", action="store_true")
    parser.add_argument("--lifetimes", action="store_true")
    parser.add_argument("--stack-depth", type=int, default=2)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="Omit unchanged allocator, pool, and device/pool/stream rows from text, HTML, and CSV",
    )


def _attribution_from_args(args: argparse.Namespace) -> MemoryAttributionOptions:
    return MemoryAttributionOptions(
        stacks=args.stacks,
        events=args.events,
        lifetimes=args.lifetimes,
        display=MemoryDisplayOptions(
            stack_depth=args.stack_depth,
            limit=args.limit,
        ),
    )


def _parse_pool_mappings(
    values: Sequence[str],
) -> Mapping[MemoryPoolKey, MemoryPoolKey]:
    result: dict[MemoryPoolKey, MemoryPoolKey] = {}
    for value in values:
        if value.count("=") != 1:
            raise ValueError(
                f"invalid pool mapping {value!r}; expected REFERENCE=CANDIDATE"
            )
        reference_text, candidate_text = value.split("=", 1)
        reference = _parse_pool_key(reference_text)
        candidate = _parse_pool_key(candidate_text)
        if reference in result:
            raise ValueError(f"pool {reference_text!r} is mapped more than once")
        result[reference] = candidate
    return result


def _parse_rank_pool_mappings(
    values: Sequence[str],
) -> Mapping[int, Mapping[MemoryPoolKey, MemoryPoolKey]]:
    result: dict[int, dict[MemoryPoolKey, MemoryPoolKey]] = {}
    for value in values:
        rank_text, separator, mapping_text = value.partition("@")
        if not separator or not rank_text or not mapping_text:
            raise ValueError(
                f"invalid rank pool mapping {value!r}; "
                "expected RANK@REFERENCE=CANDIDATE"
            )
        try:
            rank = int(rank_text)
        except ValueError as exc:
            raise ValueError(f"invalid rank in pool mapping {value!r}") from exc
        if rank < 0:
            raise ValueError("pool mapping rank must be non-negative")
        parsed = _parse_pool_mappings((mapping_text,))
        rank_mappings = result.setdefault(rank, {})
        for reference, candidate in parsed.items():
            if reference in rank_mappings:
                raise ValueError(
                    f"pool {reference.label} is mapped more than once on rank {rank}"
                )
            rank_mappings[reference] = candidate
    return result


def _parse_pool_key(value: str) -> MemoryPoolKey:
    if value.count(":") != 1:
        raise ValueError(f"invalid pool key {value!r}; expected DEVICE:POOL0,POOL1")
    device_text, pool_text = value.split(":", 1)
    try:
        device_index = int(device_text)
        pool_parts = tuple(int(part) for part in pool_text.split(","))
    except ValueError as exc:
        raise ValueError(f"invalid pool key {value!r}") from exc
    return MemoryPoolKey(device_index, pool_parts)
