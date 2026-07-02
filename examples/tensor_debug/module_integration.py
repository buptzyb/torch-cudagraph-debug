"""Insert a configurable probe into an ordinary torch.nn.Module.

Run with: python examples/tensor_debug/module_integration.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import (
    CheckAction,
    RecordAction,
    TensorProbe,
)


class InstrumentedBlock(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_probe: TensorProbe | None = None

    def debug_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        return torch.nn.functional.gelu(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.debug_hidden(hidden_states)
        if self.hidden_probe is not None:
            hidden_states = self.hidden_probe(hidden_states)
        return residual + self.fc2(hidden_states)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    torch.manual_seed(1234)
    block = InstrumentedBlock(hidden_size=16).cuda().eval()
    block.requires_grad_(False)
    static_input = torch.randn(2, 16, device="cuda")

    with torch.no_grad():
        for _ in range(3):
            block(static_input)
        expected_hidden = block.debug_hidden(static_input).cpu()

    probe = TensorProbe(
        "block.hidden_after_fc1",
        [
            RecordAction(),
            CheckAction(expected_hidden, rtol=1e-5, atol=1e-6),
        ],
    )
    block.hidden_probe = probe
    try:
        with torch.no_grad():
            block(static_input)

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            output = block(static_input)

        replay_stream = torch.cuda.current_stream()
        graph.replay()

        probe.assert_check_ok(synchronize=replay_stream)
        snapshot = probe.snapshot(synchronize=False)
        assert len(snapshot.observations) == 1
        torch.testing.assert_close(
            snapshot.tensor(),
            expected_hidden,
            rtol=1e-5,
            atol=1e-6,
        )
        assert output is not None
        print(
            f"recorded {snapshot.probe_name} "
            f"with shape {snapshot.observation().shape}"
        )
    finally:
        probe.close()


if __name__ == "__main__":
    main()
