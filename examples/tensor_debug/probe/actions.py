"""Record, print, and check one hidden tensor across graph replays.

Run with: python examples/tensor_debug/probe/actions.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import (
    CheckAction,
    PrintAction,
    RecordAction,
    TensorCheckError,
    TensorProbe,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    static_x = torch.ones(4, device="cuda")
    expected = torch.full((4,), 3.0, device="cpu")
    graph: torch.cuda.CUDAGraph | None = None
    probe = TensorProbe(
        "actions.hidden",
        [
            RecordAction(),
            PrintAction(max_items=4),
            CheckAction(expected, rtol=0.0, atol=0.0),
        ],
    )
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            probe(static_x + 2)

        replay_stream = torch.cuda.current_stream()
        graph.replay()

        probe.assert_check_ok(synchronize=replay_stream)
        matching = probe.snapshot(synchronize=False)
        assert matching.replay_index == 1
        assert torch.equal(matching.tensor(), expected)
        print(f"matching snapshot: {matching.tensor().tolist()}")

        # A real debugging session often changes static input between replays.
        static_x.fill_(2)
        graph.replay()

        status = probe.check_status(synchronize=replay_stream)
        assert not status.ok
        try:
            probe.assert_check_ok(synchronize=False)
        except TensorCheckError as exc:
            print(f"expected mismatch: {exc}")
        else:
            raise AssertionError("the changed replay should fail its check")

        changed = probe.snapshot(synchronize=False)
        assert changed.replay_index == 2
        assert torch.equal(changed.tensor(), torch.full((4,), 4.0))
        print(f"changed snapshot: {changed.tensor().tolist()}")

    finally:
        if graph is not None:
            del graph
        probe.close()


if __name__ == "__main__":
    main()
