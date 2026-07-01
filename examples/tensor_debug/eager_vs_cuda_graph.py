"""Record matching eager and CUDA Graph runs, then compare their bundles.

Run with:
  python examples/tensor_debug/eager_vs_cuda_graph.py \
    --output-dir /tmp/tcgd-tensor-eager-vs-cg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from torch_cudagraph_debug.tensor_debug import (
    TensorRecorder,
    TensorRun,
    compare_points,
)


def observed_forward(
    inputs: torch.Tensor,
    recorder: TensorRecorder,
) -> torch.Tensor:
    inputs = recorder.observe("input", inputs)
    hidden = recorder.observe("hidden.add", inputs + 1)
    hidden = recorder.observe("hidden.square", hidden.square())
    return recorder.observe("output", hidden.sum(dim=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eager_bundle = args.output_dir / "eager.tcgd-tensor"
    graph_bundle = args.output_dir / "cuda-graph.tcgd-tensor"
    report_dir = args.output_dir / "comparison"
    static_x = torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4)
    replay_stream = torch.cuda.current_stream()

    with TensorRecorder(
        execution="eager",
        name="eager",
        bundle_dir=eager_bundle,
        run_metadata={"scenario": "eager-reference"},
    ) as eager_recorder:
        with eager_recorder.point("forward", synchronize=replay_stream):
            eager_output = observed_forward(static_x, eager_recorder)
    assert eager_output is not None

    with TensorRecorder(
        execution="cuda_graph",
        name="cuda-graph",
        bundle_dir=graph_bundle,
        run_metadata={"scenario": "cuda-graph-candidate"},
    ) as graph_recorder:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = observed_forward(static_x, graph_recorder)

        with graph_recorder.point("replay-1", synchronize=replay_stream):
            graph.replay()
        assert graph_output is not None

    eager = TensorRun.load(eager_bundle)
    candidate = TensorRun.load(graph_bundle)
    comparison = compare_points(eager["forward"], candidate["replay-1"])
    comparison.assert_ok()
    paths = comparison.write(report_dir)

    print(comparison.to_text())
    print(f"eager bundle: {eager_bundle}")
    print(f"CUDA Graph bundle: {graph_bundle}")
    print(f"HTML report: {paths['html']}")


if __name__ == "__main__":
    main()
