"""Record tensors and handle both successful and failed checks.

Run with: python examples/tensor_debug/record_and_check.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import (
    CheckAction,
    RecordAction,
    TensorCheckError,
    TensorProbe,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    static_x = torch.ones(4, device="cuda")
    expected = torch.full((4,), 3.0, device="cpu")
    matching_probe = TensorProbe(
        "check.matching",
        [RecordAction(), CheckAction(expected, rtol=0.0, atol=0.0)],
    )
    mismatching_probe = TensorProbe(
        "check.mismatching",
        [CheckAction(torch.zeros(4), rtol=0.0, atol=0.0)],
    )
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            matching_probe(static_x + 2)
            mismatching_probe(static_x + 2)

        replay_stream = torch.cuda.current_stream()
        graph.replay()

        matching_probe.assert_check_ok(synchronize=replay_stream)
        snapshot = matching_probe.snapshot(synchronize=False)
        assert len(snapshot.observations) == 1
        assert torch.equal(snapshot.tensor(), expected)
        print(f"matching snapshot: {snapshot.tensor().tolist()}")

        status = mismatching_probe.check_status(synchronize=False)
        assert not status.ok
        try:
            mismatching_probe.assert_check_ok(synchronize=False)
        except TensorCheckError as exc:
            print(f"expected mismatch: {exc}")
        else:
            raise AssertionError("the mismatching probe should fail")

        matching_probe.clear_snapshot(synchronize=False)
        cleared = matching_probe.snapshot(synchronize=False)
        assert len(cleared.observations) == 1
        assert torch.equal(cleared.tensor(), torch.zeros_like(expected))
        print("cleared snapshot storage: retained slot now contains zeros")
    finally:
        matching_probe.close()
        mismatching_probe.close()


if __name__ == "__main__":
    main()
