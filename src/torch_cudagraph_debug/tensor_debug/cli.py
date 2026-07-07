"""Command-line interface for persisted tensor run analysis."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .comparison import (
    TensorComparisonOptions,
    compare_point_series,
    compare_points,
    compare_runs,
)
from .errors import TensorDebugError
from .recording import TensorRun
from .run_groups import TensorRunGroup, compare_run_groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcgd-tensor",
        description="Inspect and compare torch-cudagraph-debug tensor bundles.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    summary = commands.add_parser("summary", help="Summarize one tensor run")
    summary.add_argument("bundle")

    group_summary = commands.add_parser(
        "group-summary",
        help="Summarize direct child rank bundles as one tensor run group",
    )
    group_summary.add_argument("group")
    _add_output_options(group_summary)

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
    _add_compare_options(compare)

    run_compare = commands.add_parser(
        "compare-runs",
        help="Compare same-labeled points from two tensor runs",
    )
    run_compare.add_argument("reference_bundle")
    run_compare.add_argument("candidate_bundle")
    run_compare.add_argument(
        "--point-map",
        action="append",
        default=[],
        metavar="REFERENCE=CANDIDATE",
        help="Explicit point-label mapping; repeat for multiple points",
    )
    _add_compare_options(run_compare)

    group_compare = commands.add_parser(
        "compare-run-groups",
        help="Compare matching ranks from two tensor run groups",
    )
    group_compare.add_argument("reference_group")
    group_compare.add_argument("candidate_group")
    group_compare.add_argument(
        "--point-map",
        action="append",
        default=[],
        metavar="REFERENCE=CANDIDATE",
        help="Explicit point-label mapping; repeat for multiple points",
    )
    _add_compare_options(group_compare)

    series = commands.add_parser(
        "compare-point-series",
        help="Compare one reference point against every candidate point",
    )
    series.add_argument(
        "reference_bundle", help="Bundle containing the reference point"
    )
    series.add_argument(
        "candidate_bundle",
        nargs="?",
        help="Candidate run bundle; defaults to the reference bundle",
    )
    series.add_argument("--reference-point", required=True)
    _add_compare_options(series)
    return parser


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace report artifacts in an existing non-empty output directory",
    )


def _add_compare_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("allclose", "exact"), default="allclose")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--equal-nan", action="store_true")
    parser.add_argument(
        "--promote-dtypes",
        action="store_true",
        help="Promote differing numeric dtypes before value comparison",
    )
    parser.add_argument(
        "--ignore-layout",
        action="store_true",
        help="Compare values without requiring equal source strides",
    )
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="Omit matching observations from text, HTML, and CSV output",
    )
    parser.add_argument("--limit", type=int, default=20)
    _add_output_options(parser)


def _options(args: argparse.Namespace) -> TensorComparisonOptions:
    return TensorComparisonOptions(
        mode=args.mode,
        rtol=args.rtol,
        atol=args.atol,
        equal_nan=args.equal_nan,
        dtype_policy="promote" if args.promote_dtypes else "strict",
        layout_policy="ignore" if args.ignore_layout else "strict",
        limit=args.limit,
    )


def _point_mapping(values: Sequence[str]) -> dict[str, str] | None:
    if not values:
        return None
    result: dict[str, str] = {}
    for value in values:
        reference, separator, candidate = value.partition("=")
        if not separator or not reference or not candidate:
            raise ValueError(
                f"invalid --point-map {value!r}; expected REFERENCE=CANDIDATE"
            )
        if reference in result:
            raise ValueError(f"duplicate reference point mapping {reference!r}")
        result[reference] = candidate
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "summary":
            run = TensorRun.load(args.bundle, cache_tensors=False)
            print(_summary_text(run))
            return 0

        if args.command == "group-summary":
            result = TensorRunGroup.load(args.group, cache_tensors=False).summary()
            print(result.to_text())
            if args.output is not None:
                _print_paths(result.write(args.output, overwrite=args.overwrite))
            return 0

        if args.command == "compare-run-groups":
            reference_group = TensorRunGroup.load(
                args.reference_group, cache_tensors=False
            )
            candidate_group = TensorRunGroup.load(
                args.candidate_group, cache_tensors=False
            )
            result = compare_run_groups(
                reference_group,
                candidate_group,
                point_mapping=_point_mapping(args.point_map),
                options=_options(args),
            )
        else:
            reference = TensorRun.load(args.reference_bundle, cache_tensors=False)
            candidate = TensorRun.load(
                args.candidate_bundle or args.reference_bundle,
                cache_tensors=False,
            )
            options = _options(args)
            if args.command == "compare-points":
                result = compare_points(
                    reference.point(args.reference_point),
                    candidate.point(args.candidate_point),
                    options=options,
                )
            elif args.command == "compare-runs":
                result = compare_runs(
                    reference,
                    candidate,
                    point_mapping=_point_mapping(args.point_map),
                    options=options,
                )
            elif args.command == "compare-point-series":
                result = compare_point_series(
                    reference.point(args.reference_point),
                    candidate,
                    options=options,
                )
            else:  # pragma: no cover
                parser.error(f"unknown command {args.command!r}")

        include_unchanged = not args.only_changed
        print(result.to_text(include_unchanged=include_unchanged))
        if args.output is not None:
            _print_paths(
                result.write(
                    args.output,
                    include_unchanged=include_unchanged,
                    overwrite=args.overwrite,
                )
            )
        return 0 if result.ok else 1
    except (TensorDebugError, OSError, ValueError, KeyError) as exc:
        print(f"tcgd-tensor: error: {exc}", file=sys.stderr)
        return 2


def _print_paths(paths: dict[str, Path]) -> None:
    for kind, path in paths.items():
        print(f"{kind}: {path}")


def _summary_text(run: TensorRun) -> str:
    lines = [
        f"Tensor run {run.name!r}",
        f"  execution={run.execution} complete={run.complete} "
        f"points={len(run.points)} default_payload={run.default_payload}",
    ]
    for point in run.points:
        full = sum(item.payload == "full" for item in point.observations)
        summary = len(point.observations) - full
        replay_text = (
            str(point.replay_index) if point.replay_index is not None else "eager"
        )
        lines.append(
            f"  [{point.index}] {point.label}: "
            f"observations={len(point.observations)} full={full} "
            f"summary={summary} replay={replay_text}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
