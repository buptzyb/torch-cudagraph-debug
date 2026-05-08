from __future__ import annotations

import torch

from torch_cudagraph_debug.tensor_debug import CudaGraphTensorProbe, TensorRecord


class DebugMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        activation_value_probe: CudaGraphTensorProbe,
        activation_grad_probe: CudaGraphTensorProbe,
    ) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 3, bias=False)
        self.fc2 = torch.nn.Linear(3, 2, bias=False)
        self.activation_value_probe = activation_value_probe
        self.activation_grad_probe = activation_grad_probe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)

        # Pattern 1: probe a forward activation value.
        hidden = self.activation_value_probe(hidden)

        # Pattern 2: probe the activation gradient when backward reaches hidden.
        hidden = self.activation_grad_probe.attach_grad(hidden)

        return self.fc2(torch.relu(hidden)).sum()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    static_x = torch.randn(2, 4, device=device)

    activation_value_probe = CudaGraphTensorProbe(
        "activation.value",
        [TensorRecord()],
    )
    activation_grad_probe = CudaGraphTensorProbe(
        "activation.grad",
        [TensorRecord()],
    )
    weight_grad_probe = CudaGraphTensorProbe(
        "fc1.weight.grad.hook",
        [TensorRecord()],
    )
    final_weight_grad_probe = CudaGraphTensorProbe(
        "fc1.weight.grad.final",
        [TensorRecord()],
    )

    model = DebugMLP(
        activation_value_probe=activation_value_probe,
        activation_grad_probe=activation_grad_probe,
    ).to(device)

    # Pattern 3: probe a parameter gradient when autograd produces it.
    weight_grad_probe.attach_grad(model.fc1.weight)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            loss = model(static_x)
            loss.backward()
    del loss
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    assert activation_value_probe.records() == []
    assert activation_grad_probe.records() == []
    assert weight_grad_probe.records() == []
    assert final_weight_grad_probe.records() == []

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        model.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph):
            loss = model(static_x)
            loss.backward()

            # Pattern 4: probe the final parameter .grad buffer after backward.
            if model.fc1.weight.grad is None:
                raise RuntimeError("fc1.weight.grad should exist after backward")
            final_weight_grad_probe(model.fc1.weight.grad)
    torch.cuda.current_stream().wait_stream(capture_stream)

    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()

    probes = [
        activation_value_probe,
        activation_grad_probe,
        weight_grad_probe,
        final_weight_grad_probe,
    ]
    for probe in probes:
        records = probe.records()
        assert records, f"{probe.name} did not record any graph replay snapshots"
        last = records[-1]
        print(f"{last.probe_name}: replay={last.replay_index} shape={last.shape}")

    for probe in probes:
        probe.close()


if __name__ == "__main__":
    main()
