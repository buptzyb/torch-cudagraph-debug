import torch

from torch_cudagraph_debug.tensor_debug import TensorProbe, PrintTensor


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")

    x = torch.ones(8, device="cuda")
    probe = TensorProbe("mid", actions=[PrintTensor(max_items=8)])

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = probe(x * 2)

    g.replay()
    torch.cuda.synchronize()
    probe.close()

    assert y is not None


if __name__ == "__main__":
    main()
