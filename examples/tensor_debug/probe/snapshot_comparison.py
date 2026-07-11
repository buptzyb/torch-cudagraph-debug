"""Compare eager and CUDA Graph values without creating run bundles.

Run with: python examples/tensor_debug/probe/snapshot_comparison.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import (
    RecordAction,
    TensorProbe,
    compare_snapshots,
)


def observed_forward(inputs: torch.Tensor, probe: TensorProbe) -> torch.Tensor:
    hidden = probe((inputs + 1).square(), name="hidden.square")
    return hidden.sum()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    inputs = torch.arange(8, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream()

    eager_probe = TensorProbe(
        "eager-forward",
        [RecordAction()],
        when="always",
    )
    try:
        eager_output = observed_forward(inputs, eager_probe)
        eager_snapshot = eager_probe.snapshot(synchronize=stream)
    finally:
        eager_probe.close()

    graph: torch.cuda.CUDAGraph | None = None
    graph_probe = TensorProbe("cuda-graph-forward", [RecordAction()])
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = observed_forward(inputs, graph_probe)

        graph.replay()
        graph_snapshot = graph_probe.snapshot(synchronize=stream)

        comparison = compare_snapshots(eager_snapshot, graph_snapshot)
        comparison.assert_ok()
        assert eager_snapshot.replay_index == 0
        assert graph_snapshot.replay_index == 1
        torch.testing.assert_close(eager_output, graph_output)

        print(comparison.to_text())
    finally:
        if graph is not None:
            del graph
        graph_probe.close()


if __name__ == "__main__":
    main()
