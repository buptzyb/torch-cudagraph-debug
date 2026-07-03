"""Probe forward activations and activation or parameter gradients.

Run with: python examples/tensor_debug/probe/gradients.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import RecordAction, TensorProbe


class DebugMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        activation_probe: TensorProbe,
        activation_grad_probe: TensorProbe,
    ) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 3, bias=False)
        self.fc2 = torch.nn.Linear(3, 2, bias=False)
        self.activation_probe = activation_probe
        self.activation_grad_probe = activation_grad_probe

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation_probe(self.fc1(inputs), name="activation.value")
        self.activation_grad_probe.watch_grad(
            hidden, name="activation.grad", strict=True
        )
        return self.fc2(torch.relu(hidden)).sum()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    torch.manual_seed(1234)
    static_x = torch.randn(2, 4, device="cuda")
    probes = {
        name: TensorProbe(name, [RecordAction()])
        for name in (
            "activation.value",
            "activation.grad",
            "fc1.weight.grad.hook",
            "fc1.weight.grad.final",
        )
    }
    model = DebugMLP(
        activation_probe=probes["activation.value"],
        activation_grad_probe=probes["activation.grad"],
    ).cuda()
    weight_hook = probes["fc1.weight.grad.hook"].watch_grad(
        model.fc1.weight,
        name="fc1.weight.grad.hook",
        strict=True,
    )
    assert weight_hook is not None

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                loss = model(static_x)
                loss.backward()
        del loss
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(capture_stream)
        current_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            model.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph):
                captured_loss = model(static_x)
                captured_loss.backward()
                grad = model.fc1.weight.grad
                if grad is None:
                    raise RuntimeError("fc1.weight.grad should exist after backward")
                probes["fc1.weight.grad.final"](grad, name="fc1.weight.grad.final")
        torch.cuda.current_stream().wait_stream(capture_stream)

        replay_stream = torch.cuda.current_stream()
        graph.replay()
        replay_stream.synchronize()

        for probe in probes.values():
            snapshot = probe.snapshot(synchronize=False)
            assert snapshot.observations, f"{probe.name} did not record a replay value"
            observation = snapshot.observation()
            print(
                f"{observation.name}: replay={snapshot.replay_index} "
                f"shape={observation.shape}"
            )
    finally:
        weight_hook.remove()
        for probe in probes.values():
            probe.close()


if __name__ == "__main__":
    main()
