"""Compare every CUDA Graph replay with one eager reference point.

Most observations use summary-only storage. The output remains full so the
second replay still produces a definite numerical mismatch; earlier
summary-only changes remain explicitly inconclusive under allclose.

Run with:
  python examples/tensor_debug/recorder/replay_series.py \
    --output-dir /tmp/tcgd-tensor-series
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from torch_cudagraph_debug.tensor_debug import (
    TensorRecorder,
    TensorRun,
    compare_point_series,
)


def observed_forward(
    inputs: torch.Tensor,
    recorder: TensorRecorder,
) -> torch.Tensor:
    inputs = recorder.observe("input", inputs)
    hidden = recorder.observe("hidden", inputs * 2)
    return recorder.observe("output", hidden + 3, payload="full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    eager_bundle = output_dir / "eager-summary.tcgd-tensor"
    graph_bundle = output_dir / "graph-series.tcgd-tensor"
    report_dir = output_dir / "series-report"
    reference_x = torch.arange(4, dtype=torch.float32, device="cuda")
    static_x = reference_x.clone()
    replay_stream = torch.cuda.current_stream()

    with TensorRecorder(
        execution="eager",
        name="eager-summary",
        bundle_dir=eager_bundle,
        payload="summary",
    ) as eager_recorder:
        with eager_recorder.record_point("forward", synchronize=replay_stream):
            observed_forward(reference_x, eager_recorder)

    graph_recorder = TensorRecorder(
        execution="cuda_graph",
        name="graph-series",
        bundle_dir=graph_bundle,
        payload="summary",
    )
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            observed_forward(static_x, graph_recorder)

        with graph_recorder.record_point("replay-1", synchronize=replay_stream):
            graph.replay()

        static_x.add_(1)
        with graph_recorder.record_point("replay-2", synchronize=replay_stream):
            graph.replay()
        graph_recorder.finish()
    finally:
        if "graph" in locals():
            del graph
        graph_recorder.close(synchronize=replay_stream)

    eager = TensorRun.load(eager_bundle)
    candidate = TensorRun.load(graph_bundle)
    print(f"reference bundle: {eager_bundle}")
    print(f"candidate bundle: {graph_bundle}")
    if args.record_only:
        return

    series = compare_point_series(eager["forward"], candidate)
    assert series.point_comparisons[0].status == "match"
    assert series.point_comparisons[1].status == "mismatch"
    paths = series.write(report_dir, include_unchanged=False)

    print(series.to_text(include_unchanged=False))
    print(f"HTML report: {paths['html']}")


if __name__ == "__main__":
    main()
