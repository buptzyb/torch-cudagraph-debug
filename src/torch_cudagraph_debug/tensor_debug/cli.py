"""Command-line interface for persisted tensor run analysis."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .comparison import (
    TensorCompareOptions,
    compare_points,
    compare_runs,
    compare_series,
)
from .errors import TensorDebugError
from .runs import TensorRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcgd-tensor",
        description="Inspect and compare torch-cudagraph-debug tensor bundles.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    summary = commands.add_parser("summary", help="Summarize one tensor run")
    summary.add_argument("bundle")

    compare = commands.add_parser(
        "compare",
        help="Compare one reference point with one candidate point",
    )
    compare.add_argument("reference_bundle")
    compare.add_argument("candidate_bundle")
    compare.add_argument("--reference-point", required=True)
    compare.add_argument("--candidate-point", required=True)
    _add_compare_options(compare)

    run_compare = commands.add_parser(
        "compare-runs",
        help="Compare same-labeled points from two tensor runs",
    )
    run_compare.add_argument("reference_bundle")
    run_compare.add_argument("candidate_bundle")
    _add_compare_options(run_compare)

    series = commands.add_parser(
        "compare-series",
        help="Compare one reference point against every candidate point",
    )
    series.add_argument("reference_bundle")
    series.add_argument("candidate_bundle")
    series.add_argument("--reference-point", required=True)
    _add_compare_options(series)
    return parser


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
        "--only-changed",
        action="store_true",
        help="Omit matching observations from text, HTML, and CSV output",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path)


def _options(args: argparse.Namespace) -> TensorCompareOptions:
    return TensorCompareOptions(
        mode=args.mode,
        rtol=args.rtol,
        atol=args.atol,
        equal_nan=args.equal_nan,
        dtype_policy="promote" if args.promote_dtypes else "strict",
        limit=args.limit,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "summary":
            run = TensorRun.load(args.bundle, cache_tensors=False)
            print(_summary_text(run))
            return 0

        reference = TensorRun.load(args.reference_bundle, cache_tensors=False)
        candidate = TensorRun.load(args.candidate_bundle, cache_tensors=False)
        options = _options(args)
        if args.command == "compare":
            report = compare_points(
                reference.point(args.reference_point),
                candidate.point(args.candidate_point),
                options=options,
            )
        elif args.command == "compare-runs":
            report = compare_runs(reference, candidate, options=options)
        elif args.command == "compare-series":
            report = compare_series(
                reference.point(args.reference_point),
                candidate,
                options=options,
            )
        else:  # pragma: no cover - argparse guarantees a known command
            parser.error(f"unknown command {args.command!r}")

        include_matches = not args.only_changed
        print(report.to_text(include_matches=include_matches))
        if args.output is not None:
            paths = report.write(args.output, include_matches=include_matches)
            for kind, path in paths.items():
                print(f"{kind}: {path}")
        return 0 if report.ok else 1
    except (TensorDebugError, OSError, ValueError, KeyError) as exc:
        print(f"tcgd-tensor: error: {exc}", file=sys.stderr)
        return 2


def _summary_text(run: TensorRun) -> str:
    lines = [
        f"Tensor run {run.name!r}",
        f"  execution={run.execution} complete={run.complete} "
        f"points={len(run.points)} default_payload={run.default_payload}",
    ]
    for point in run.points:
        full = sum(item.payload == "full" for item in point.observations)
        summary = len(point.observations) - full
        replay_indices = sorted(
            {
                item.replay_index
                for item in point.observations
                if item.replay_index is not None
            }
        )
        replay_text = (
            ",".join(str(item) for item in replay_indices)
            if replay_indices
            else "eager"
        )
        lines.append(
            f"  [{point.index}] {point.label}: "
            f"observations={len(point.observations)} full={full} "
            f"summary={summary} replay={replay_text}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
