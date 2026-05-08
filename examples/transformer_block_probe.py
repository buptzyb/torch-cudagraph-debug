from __future__ import annotations

from dataclasses import dataclass

import torch

from torch_cudagraph_debug.tensor_debug import (
    CudaGraphTensorProbe,
    TensorCompare,
    TensorPrint,
    TensorRecord,
)


@dataclass(frozen=True)
class HiddenProbeConfig:
    print_hidden: bool = False
    record_hidden: bool = True
    compare_hidden: bool = True
    expected_hidden: torch.Tensor | None = None


class FeedForwardBlock(torch.nn.Module):
    """Small transformer-block-shaped module with a probe on an internal hidden tensor."""

    def __init__(
        self,
        hidden_size: int,
        *,
        layer_index: int,
        probe_config: HiddenProbeConfig,
    ) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = torch.nn.Linear(hidden_size * 2, hidden_size)

        expected = probe_config.expected_hidden
        if expected is None:
            expected = torch.empty(0)

        self.hidden_probe = CudaGraphTensorProbe(
            f"block.{layer_index}.hidden_after_fc1",
            actions=[
                TensorPrint(max_items=8, every=1, enabled=probe_config.print_hidden),
                TensorRecord(enabled=probe_config.record_hidden),
                TensorCompare(
                    [expected],
                    rtol=1e-5,
                    atol=1e-6,
                    enabled=probe_config.compare_hidden,
                ),
            ],
            mode="capture",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.hidden_probe(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states

    def close_debug_probes(self) -> None:
        self.hidden_probe.close()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    hidden_size = 16
    static_input = torch.randn(2, hidden_size, device=device)

    disabled_probe_config = HiddenProbeConfig(
        record_hidden=False,
        compare_hidden=False,
    )
    warmup_block = FeedForwardBlock(
        hidden_size,
        layer_index=0,
        probe_config=disabled_probe_config,
    ).to(device)
    warmup_block.eval()
    warmup_block.requires_grad_(False)

    with torch.no_grad():
        for _ in range(3):
            warmup_block(static_input)
    torch.cuda.synchronize()
    assert warmup_block.hidden_probe.records() == []

    with torch.no_grad():
        expected_hidden = torch.nn.functional.gelu(
            warmup_block.fc1(warmup_block.norm(static_input))
        ).detach().cpu()

    probe_config = HiddenProbeConfig(expected_hidden=expected_hidden)
    block = FeedForwardBlock(
        hidden_size,
        layer_index=0,
        probe_config=probe_config,
    ).to(device)
    block.load_state_dict(warmup_block.state_dict())
    block.eval()
    block.requires_grad_(False)

    with torch.no_grad():
        for _ in range(3):
            block(static_input)
    torch.cuda.synchronize()
    assert block.hidden_probe.records() == []

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        output = block(static_input)

    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()

    block.hidden_probe.assert_ok()
    records = block.hidden_probe.records()
    assert len(records) == 1
    assert torch.allclose(records[0].tensor, expected_hidden, rtol=1e-5, atol=1e-6)
    assert output is not None

    warmup_block.close_debug_probes()
    block.close_debug_probes()


if __name__ == "__main__":
    main()
