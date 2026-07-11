"""Choose eager/capture behavior and a non-contiguous input policy.

Run with: python examples/tensor_debug/probe/capture_modes.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import (
    CheckAction,
    RecordAction,
    TensorProbe,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    base = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    non_contiguous = base.t()
    expected_view = non_contiguous.detach().cpu().contiguous()

    capture_only = TensorProbe("mode.capture-only", [RecordAction()])
    eager = TensorProbe(
        "mode.always",
        [RecordAction()],
        when="always",
    )
    copying = TensorProbe(
        "mode.non-contiguous-copy",
        [
            RecordAction(),
            CheckAction(expected_view, rtol=0.0, atol=0.0),
        ],
        non_contiguous="copy",
    )
    rejecting = TensorProbe(
        "mode.non-contiguous-error",
        [RecordAction()],
        when="always",
    )
    graph: torch.cuda.CUDAGraph | None = None
    try:
        assert capture_only(non_contiguous) is non_contiguous
        print("capture-only probe: eager call was a transparent no-op")

        shifted = base + 1
        latest_base = base + 2
        assert eager(base, name="base") is base
        assert eager(shifted, name="shifted") is shifted
        assert eager(latest_base, name="base") is latest_base
        eager_snapshot = eager.snapshot(synchronize=torch.cuda.current_stream())
        assert [item.name for item in eager_snapshot.observations] == [
            "base",
            "shifted",
        ]
        assert torch.equal(
            eager_snapshot.observation("base").tensor(),
            latest_base.detach().cpu(),
        )
        # Re-sampling "base" in place left an audit trail: the snapshot
        # discloses how many earlier values the latest sample superseded.
        assert dict(eager_snapshot.eager_overwrite_counts) == {"base": 1}
        print(
            "always probe: eager names kept independent latest-value slots "
            f"(overwrites: {dict(eager_snapshot.eager_overwrite_counts)})"
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            copied_output = copying(non_contiguous)
        replay_stream = torch.cuda.current_stream()
        graph.replay()
        copying.assert_check_ok(synchronize=replay_stream)
        assert copied_output is non_contiguous
        assert torch.equal(
            copying.snapshot(synchronize=False).tensor(),
            expected_view,
        )
        print("copy policy: non-contiguous graph input was recorded")

        try:
            rejecting(non_contiguous)
        except RuntimeError as exc:
            print(f"expected default-policy error: {exc}")
        else:
            raise AssertionError("default non-contiguous policy should reject the view")
    finally:
        if graph is not None:
            del graph
        capture_only.close()
        eager.close()
        copying.close()
        rejecting.close()


if __name__ == "__main__":
    main()
