"""Record activations and gradients from one CUDA Graph training step.

Run with:
  python examples/tensor_debug/recorder/forward_backward.py \
    --output-dir /tmp/tcgd-tensor-forward-backward
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from torch_cudagraph_debug.tensor_debug import TensorRecorder, TensorRun


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    output_dir = parse_args().output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    bundle_dir = output_dir / "forward-backward.tcgd-tensor"

    torch.manual_seed(7)
    model = torch.nn.Linear(4, 3, bias=False).cuda()
    static_x = torch.randn(2, 4, device="cuda")
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            warmup_hidden = model(static_x)
            warmup_hidden.square().mean().backward()
    del warmup_hidden
    replay_stream = torch.cuda.current_stream()
    replay_stream.wait_stream(capture_stream)
    replay_stream.synchronize()

    recorder = TensorRecorder(
        execution="cuda_graph",
        name="forward-backward",
        bundle_dir=bundle_dir,
        non_contiguous="copy",
        run_metadata={"scenario": "captured-training-step"},
    )
    gradient_hook = None
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            model.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph):
                hidden = recorder.observe(model(static_x), name="activation")
                gradient_hook = recorder.watch_grad(
                    hidden,
                    name="activation.grad",
                    strict=True,
                )
                loss = recorder.observe(hidden.square().mean(), name="loss")
                loss.backward()
                weight_grad = model.weight.grad
                if weight_grad is None:
                    raise RuntimeError("model.weight.grad is unavailable")
                recorder.observe(weight_grad, name="weight.grad")
        replay_stream.wait_stream(capture_stream)

        with recorder.record_point("train_step", synchronize=replay_stream):
            graph.replay()

        partial = recorder.preview()
        assert not partial.complete
        assert partial["train_step"].observation("activation.grad").shape == (2, 3)
        recorder.finish()
    finally:
        if "graph" in locals():
            del graph
        if gradient_hook is not None:
            gradient_hook.remove()
        recorder.close(synchronize=replay_stream)

    loaded = TensorRun.load(bundle_dir)
    point = loaded["train_step"]
    assert loaded.complete
    assert point.observation("activation").shape == (2, 3)
    assert point.observation("activation.grad").shape == (2, 3)
    assert point.observation("weight.grad").shape == (3, 4)

    print(
        "recorded train_step: "
        f"observations={len(point.observations)} replay={point.replay_index}"
    )
    print(f"bundle: {bundle_dir}")


if __name__ == "__main__":
    main()
