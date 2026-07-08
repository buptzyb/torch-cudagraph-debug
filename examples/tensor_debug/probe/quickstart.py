"""Record named tensors from CUDA Graph replay.

Run with: python examples/tensor_debug/probe/quickstart.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import RecordAction, TensorProbe


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    static_x = torch.arange(8, device="cuda", dtype=torch.float32)
    first_expected = torch.arange(8, dtype=torch.float32) * 2
    expected = (first_expected, torch.relu(first_expected - 5))
    graph: torch.cuda.CUDAGraph | None = None
    probe = TensorProbe("quickstart.hidden", [RecordAction()])
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # One probe can record multiple intermediate tensors in one graph.
            first_source = static_x * 2
            first_hidden = probe(first_source, name="after_scale")
            second_source = torch.relu(first_hidden - 5)
            second_hidden = probe(second_source, name="after_relu")
            output = second_hidden.square()

        replay_stream = torch.cuda.current_stream()
        graph.replay()

        # The query waits only for the stream that launched this replay.
        snapshot = probe.snapshot(synchronize=replay_stream)
        assert [
            (item.name, item.invocation_index, item.order)
            for item in snapshot.observations
        ] == [("after_scale", 0, 0), ("after_relu", 0, 1)]
        for observation, expected_tensor in zip(snapshot.observations, expected):
            tensor = observation.tensor()
            torch.testing.assert_close(tensor, expected_tensor)
            print(
                f"replay={snapshot.replay_index} "
                f"name={observation.name} invocation={observation.invocation_index} "
                f"order={observation.order}: {tensor}"
            )
        assert first_hidden is first_source
        assert second_hidden is second_source
        torch.testing.assert_close(output.cpu(), expected[1].square())
    finally:
        if graph is not None:
            del graph
        probe.close()


if __name__ == "__main__":
    main()
