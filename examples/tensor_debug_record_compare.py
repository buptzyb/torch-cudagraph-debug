import torch

from torch_cudagraph_debug.tensor_debug import (
    TensorProbe,
    CompareTensor,
    RecordTensor,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")

    x = torch.ones(4, device="cuda")
    expected = torch.full((4,), 3.0, device="cpu")
    probe = TensorProbe(
        "mid",
        actions=[
            RecordTensor(),
            CompareTensor([expected], rtol=1e-5, atol=1e-8),
        ],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = probe(x + 2)

    g.replay()
    torch.cuda.synchronize()

    probe.assert_ok()
    snapshots = probe.snapshots()
    assert len(snapshots) == 1
    assert snapshots[0].replay_index == 0
    assert torch.equal(snapshots[0].tensor, expected)
    assert y is not None
    probe.close()


if __name__ == "__main__":
    main()
