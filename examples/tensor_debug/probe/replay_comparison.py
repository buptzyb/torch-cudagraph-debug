"""Compare two graph replays recorded by one multi-invocation probe.

Run with: python examples/tensor_debug/probe/replay_comparison.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import RecordAction, TensorProbe


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    static_x = torch.arange(4, device="cuda", dtype=torch.float32)
    graph: torch.cuda.CUDAGraph | None = None
    probe = TensorProbe("replay.hidden", [RecordAction()])
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            first = probe(static_x + 1)
            second = probe(first.square())
            output = second.sum()

        replay_stream = torch.cuda.current_stream()
        graph.replay()
        reference = probe.snapshot(synchronize=replay_stream)

        static_x.add_(1)
        graph.replay()
        candidate = probe.snapshot(synchronize=replay_stream)

        comparison = probe.compare(reference, candidate)
        assert comparison.status == "mismatch"
        assert comparison.first_issue is not None
        assert reference.replay_index == 1
        assert candidate.replay_index == 2
        assert [item.invocation_index for item in candidate.observations] == [
            0,
            1,
        ]
        print(comparison.to_text(include_unchanged=False))
        assert output is not None
    finally:
        if graph is not None:
            del graph
        probe.close()


if __name__ == "__main__":
    main()
