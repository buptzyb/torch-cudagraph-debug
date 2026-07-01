"""Choose eager/capture behavior and a non-contiguous input policy.

Run with: python examples/tensor_debug/probe_modes.py
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

    base = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    non_contiguous = base.t()
    expected_view = non_contiguous.detach().cpu().contiguous()

    capture_only = TensorProbe("mode.capture-only", [RecordTensor()])
    eager = TensorProbe(
        "mode.always",
        [
            RecordTensor(),
            CompareTensor(base.detach().cpu(), rtol=0.0, atol=0.0),
        ],
        when="always",
    )
    copying = TensorProbe(
        "mode.non-contiguous-copy",
        [
            RecordTensor(),
            CompareTensor(expected_view, rtol=0.0, atol=0.0),
        ],
        non_contiguous="copy",
    )
    rejecting = TensorProbe(
        "mode.non-contiguous-error",
        [RecordTensor()],
        when="always",
    )
    try:
        assert capture_only(non_contiguous) is non_contiguous
        torch.cuda.synchronize()
        assert capture_only.snapshots() == []
        print("capture-only probe: eager call was a transparent no-op")

        assert eager(base) is base
        torch.cuda.synchronize()
        eager.assert_ok()
        assert len(eager.snapshots()) == 1
        print("always probe: eager call produced one snapshot")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            copied_output = copying(non_contiguous)
        graph.replay()
        torch.cuda.synchronize()
        copying.assert_ok()
        assert copied_output is non_contiguous
        assert torch.equal(copying.snapshots()[0].tensor, expected_view)
        print("copy policy: non-contiguous graph input was recorded")

        try:
            rejecting(non_contiguous)
        except RuntimeError as exc:
            print(f"expected default-policy error: {exc}")
        else:
            raise AssertionError("default non-contiguous policy should reject the view")
    finally:
        capture_only.close()
        eager.close()
        copying.close()
        rejecting.close()


if __name__ == "__main__":
    main()
