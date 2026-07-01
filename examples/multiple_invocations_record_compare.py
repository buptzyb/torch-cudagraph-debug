"""Record repeated probe calls and compare them in Python after replay."""

from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import (
    TensorProbe,
    RecordTensor,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = [torch.arange(4, dtype=torch.float32) + offset for offset in (1, 2, 3)]
    probe = TensorProbe(
        "repeated.hidden",
        actions=[RecordTensor()],
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y0 = probe(x + 1)
        y1 = probe(x + 2)
        y2 = probe(x + 3)
        _ = y0 + y1 + y2

    snapshots_by_replay: list[tuple[int, int, torch.Tensor]] = []
    for replay_index in range(1, 3):
        graph.replay()
        torch.cuda.synchronize()
        for snapshot in probe.snapshots():
            snapshots_by_replay.append(
                (replay_index, snapshot.invocation_index, snapshot.tensor.clone())
            )

    for snapshot in probe.snapshots():
        print(
            f"replay={snapshot.replay_index} "
            f"invocation={snapshot.invocation_index} "
            f"value={snapshot.tensor.tolist()}"
        )
        torch.testing.assert_close(
            snapshot.tensor,
            expected[snapshot.invocation_index],
            rtol=0.0,
            atol=0.0,
        )
    assert len(snapshots_by_replay) == 6
    for _, invocation_index, tensor in snapshots_by_replay:
        torch.testing.assert_close(
            tensor,
            expected[invocation_index],
            rtol=0.0,
            atol=0.0,
        )
    probe.close()


if __name__ == "__main__":
    main()
