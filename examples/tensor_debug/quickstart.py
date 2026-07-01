"""Print one tensor from CUDA Graph replay.

Run with: python examples/tensor_debug/quickstart.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import PrintTensor, TensorProbe


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    static_x = torch.arange(8, device="cuda", dtype=torch.float32)
    probe = TensorProbe("quickstart.hidden", [PrintTensor(max_items=8)])
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hidden = probe(static_x * 2)

        graph.replay()
        torch.cuda.synchronize()

        assert probe.status().ok
        assert hidden.data_ptr() != static_x.data_ptr()
        print(f"status: {probe.status()}")
    finally:
        probe.close()


if __name__ == "__main__":
    main()
