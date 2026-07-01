"""Use one probe at several call sites in the same graph capture.

Run with: python examples/tensor_debug/multiple_invocations.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import (
    CompareTensor,
    RecordTensor,
    TensorProbe,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    static_x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = tuple(
        torch.arange(4, dtype=torch.float32) + offset for offset in (1, 2, 3)
    )
    probe = TensorProbe(
        "repeated.hidden",
        [RecordTensor(), CompareTensor(expected, rtol=0.0, atol=0.0)],
    )
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            first = probe(static_x + 1)
            second = probe(static_x + 2)
            third = probe(static_x + 3)
            output = first + second + third

        replay_stream = torch.cuda.current_stream()
        for _ in range(2):
            graph.replay()

        probe.assert_ok(synchronize=replay_stream)
        snapshots = probe.snapshots(synchronize=False)
        assert len(snapshots) == 3
        assert [snapshot.replay_index for snapshot in snapshots] == [2, 2, 2]
        for snapshot in snapshots:
            expected_tensor = expected[snapshot.invocation_index]
            torch.testing.assert_close(
                snapshot.tensor,
                expected_tensor,
                rtol=0.0,
                atol=0.0,
            )
            print(
                f"invocation={snapshot.invocation_index} "
                f"replay={snapshot.replay_index} "
                f"value={snapshot.tensor.tolist()}"
            )
        assert output is not None
    finally:
        probe.close()


if __name__ == "__main__":
    main()
