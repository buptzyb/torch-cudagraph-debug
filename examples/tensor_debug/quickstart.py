"""Record one tensor from CUDA Graph replay.

Run with: python examples/tensor_debug/quickstart.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import RecordTensor, TensorProbe


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    static_x = torch.arange(8, device="cuda", dtype=torch.float32)
    first_expected = torch.arange(8, dtype=torch.float32) * 2
    expected = (first_expected, torch.relu(first_expected - 5))
    probe = TensorProbe("quickstart.hidden", [RecordTensor()])
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # One probe can record multiple intermediate tensors in one graph.
            first_source = static_x * 2
            first_hidden = probe(first_source)
            second_source = torch.relu(first_hidden - 5)
            second_hidden = probe(second_source)
            output = second_hidden.square()

        replay_stream = torch.cuda.current_stream()
        graph.replay()

        # The query waits only for the stream that launched this replay.
        snapshots = probe.snapshots(synchronize=replay_stream)
        assert [item.invocation_index for item in snapshots] == [0, 1]
        for snapshot, expected_tensor in zip(snapshots, expected):
            torch.testing.assert_close(snapshot.tensor, expected_tensor)
            print(
                f"replay={snapshot.replay_index} "
                f"invocation={snapshot.invocation_index}: {snapshot.tensor}"
            )
        assert first_hidden is first_source
        assert second_hidden is second_source
        torch.testing.assert_close(output.cpu(), expected[1].square())
    finally:
        probe.close()


if __name__ == "__main__":
    main()
