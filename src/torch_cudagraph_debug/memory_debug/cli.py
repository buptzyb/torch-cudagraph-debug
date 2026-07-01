"""Command-line interface for offline CUDA allocator memory analysis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any

from .core import (
    AttributionOptions,
    MemoryRun,
    compare_phases,
    compare_points,
)
from .errors import MemoryDebugError
from .groups import MemoryRunGroup, compare_group_phases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcgd-memory",
        description="Analyze torch-cudagraph-debug memory run bundles.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    timeline = commands.add_parser(
        "timeline", help="Report every point in one memory run"
    )
    timeline.add_argument("bundle")
    _add_common_options(timeline)
    lifetimes = commands.add_parser(
        "lifetimes", help="Trace active allocation cohorts across one memory run"
    )
    lifetimes.add_argument("bundle")
    lifetime_selection = lifetimes.add_mutually_exclusive_group()
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
    lifetimes.add_argument(
        "--through",
        help="Stop tracing at this point instead of the final point",
    )
    _add_common_options(lifetimes)
    lifetimes.add_argument(
        "--no-events",
        action="store_false",
        dest="events",
        help="Skip allocator event pairing and use snapshots only",
    )
    lifetimes.set_defaults(events=True, stack_depth=4)

    compare = commands.add_parser(
        "compare", help="Compare two ordered points in one memory run"
    )
    compare.add_argument("bundle")
    compare.add_argument("--before", required=True)
    compare.add_argument("--after", required=True)
    _add_common_options(compare)

    runs = commands.add_parser(
        "compare-runs",
        help="Compare one point from each of two independent runs",
    )
    runs.add_argument("before_bundle")
    runs.add_argument("after_bundle")
    runs.add_argument("--before", required=True)
    runs.add_argument("--after", required=True)
    runs.add_argument(
        "--pool-map",
        action="append",
        default=[],
        metavar="BEFORE=AFTER",
        help="Explicit private-pool mapping, for example 0,1=0,2",
    )
    _add_common_options(runs)

    phases = commands.add_parser(
        "compare-phases",
        help="Decompose a baseline/candidate difference using four points",
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
    _add_common_options(phases)

    group_summary = commands.add_parser(
        "summarize-group",
        help="Summarize compatible per-rank memory bundles without summing GPUs",
    )
    group_summary.add_argument("group_dir")
    group_summary.add_argument("--output", required=True)

    group_phases = commands.add_parser(
        "compare-group-phases",
        help="Compare four-point phase memory rank by rank across two groups",
    )
    group_phases.add_argument("baseline_group")
    group_phases.add_argument("candidate_group")
    group_phases.add_argument("--baseline-start", required=True)
    group_phases.add_argument("--baseline-end", required=True)
    group_phases.add_argument("--candidate-start", required=True)
    group_phases.add_argument("--candidate-end", required=True)
    _add_common_options(group_phases)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    options = AttributionOptions(
        stacks=getattr(args, "stacks", False),
        events=getattr(args, "events", False),
        lifetimes=getattr(args, "lifetimes", False),
        on_missing=getattr(args, "on_missing", "warn"),
        stack_depth=getattr(args, "stack_depth", 2),
        limit=getattr(args, "limit", 20),
    )
    include_unchanged = not getattr(args, "only_changed", False)

    try:
        if args.command == "timeline":
            result = _load_run(args.bundle).timeline(attribution=options)
        elif args.command == "lifetimes":
            run = _load_run(args.bundle)
            result = run.lifetimes(
                args.at,
                born_between=(
                    tuple(args.born_between) if args.born_between is not None else None
                ),
                through=args.through,
                attribution=options,
            )
        elif args.command == "compare":
            run = _load_run(args.bundle)
            result = run.compare(
                args.before,
                args.after,
                attribution=options,
            )
        elif args.command == "compare-runs":
            before_run = _load_run(args.before_bundle)
            after_run = _load_run(args.after_bundle)
            result = compare_points(
                before_run.point(args.before),
                after_run.point(args.after),
                pool_mapping=_parse_pool_mappings(args.pool_map),
                attribution=options,
            )
        elif args.command == "compare-phases":
            baseline = _load_run(args.baseline_bundle)
            candidate = _load_run(args.candidate_bundle)
            result = compare_phases(
                baseline.between(args.baseline_start, args.baseline_end),
                candidate.between(args.candidate_start, args.candidate_end),
                pool_mapping=_parse_pool_mappings(args.pool_map),
                attribution=options,
            )
        elif args.command == "summarize-group":
            result = MemoryRunGroup.load(args.group_dir).summary()
        else:
            baseline_group = MemoryRunGroup.load(args.baseline_group)
            candidate_group = MemoryRunGroup.load(args.candidate_group)
            result = compare_group_phases(
                baseline_group,
                candidate_group,
                baseline_start=args.baseline_start,
                baseline_end=args.baseline_end,
                candidate_start=args.candidate_start,
                candidate_end=args.candidate_end,
                attribution=options,
            )
    except (MemoryDebugError, KeyError, ValueError, IndexError) as exc:
        parser.error(str(exc))

    result.write(
        args.output,
        include_unchanged=include_unchanged,
    )
    print(result.to_text(include_unchanged=include_unchanged))
    return 0


def _load_run(bundle: str) -> MemoryRun:
    return MemoryRun.load(bundle, cache_snapshots=False)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True)
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
        help="Omit unchanged pool and pool/stream rows from text and CSV",
    )


def _parse_pool_mappings(
    values: Sequence[str],
) -> Mapping[tuple[Any, ...], tuple[Any, ...]]:
    result: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for value in values:
        if value.count("=") != 1:
            raise ValueError(f"invalid pool mapping {value!r}; expected BEFORE=AFTER")
        before_text, after_text = value.split("=", 1)
        before = _parse_pool_id(before_text)
        after = _parse_pool_id(after_text)
        if before in result:
            raise ValueError(f"pool {before_text!r} is mapped more than once")
        result[before] = after
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
