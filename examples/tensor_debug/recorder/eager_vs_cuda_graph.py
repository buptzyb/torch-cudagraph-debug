"""Record matching eager and CUDA Graph runs, then compare their bundles.

Run with:
  python examples/tensor_debug/recorder/eager_vs_cuda_graph.py \
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
    compare_runs,
)


def observed_forward(
    inputs: torch.Tensor,
    recorder: TensorRecorder,
) -> torch.Tensor:
    inputs = recorder.observe(inputs, name="input")
    hidden = recorder.observe(inputs + 1, name="hidden.add")
    hidden = recorder.observe(hidden.square(), name="hidden.square")
    return recorder.observe(hidden.sum(dim=0), name="output")


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
    eager_bundle = output_dir / "eager.tcgd-tensor"
    graph_bundle = output_dir / "cuda-graph.tcgd-tensor"
    report_dir = output_dir / "run-comparison"
    static_x = torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4)
    replay_stream = torch.cuda.current_stream()

    with TensorRecorder(
        execution="eager",
        name="eager",
        bundle_dir=eager_bundle,
        run_metadata={"scenario": "eager-reference"},
        rank=0,
        group_id="eager-example",
        world_size=1,
    ) as eager_recorder:
        with eager_recorder.record_point("forward", synchronize=replay_stream):
            eager_output = observed_forward(static_x, eager_recorder)
    assert eager_output is not None

    graph_recorder = TensorRecorder(
        execution="cuda_graph",
        name="cuda-graph",
        bundle_dir=graph_bundle,
        run_metadata={"scenario": "cuda-graph-candidate"},
        rank=0,
        group_id="cuda-graph-example",
        world_size=1,
    )
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = observed_forward(static_x, graph_recorder)

        with graph_recorder.record_point("forward", synchronize=replay_stream):
            graph.replay()
        assert graph_output is not None
        graph_recorder.finish()
    finally:
        if "graph" in locals():
            del graph
        graph_recorder.close(synchronize=replay_stream)

    eager = TensorRun.load(eager_bundle)
    candidate = TensorRun.load(graph_bundle)
    point_comparison = compare_points(eager["forward"], candidate["forward"])
    point_comparison.assert_ok()
    run_comparison = compare_runs(eager, candidate)
    assert run_comparison.ok, run_comparison.to_text()

    print(f"eager bundle: {eager_bundle}")
    print(f"CUDA Graph bundle: {graph_bundle}")
    if args.record_only:
        return

    paths = run_comparison.write(report_dir)
    print(run_comparison.to_text())
    print(f"HTML report: {paths['html']}")


if __name__ == "__main__":
    main()
