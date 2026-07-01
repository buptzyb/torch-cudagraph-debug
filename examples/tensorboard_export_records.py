from __future__ import annotations

import argparse
from pathlib import Path

import torch

from torch_cudagraph_debug.tensor_debug import (
    TensorProbe,
    RecordTensor,
    TensorSnapshot,
)
from torch_cudagraph_debug.tensor_debug.postprocess import (
    export_snapshots_to_tensorboard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CUDA Graph tensor debug records to TensorBoard.",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("runs/tcgd-tensorboard"),
        help="TensorBoard log directory.",
    )
    return parser.parse_args()


def create_writer(logdir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "This example requires TensorBoard. Install it with `pip install tensorboard`."
        ) from exc

    return SummaryWriter(str(logdir))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")

    args = parse_args()
    writer = create_writer(args.logdir)

    x = torch.arange(8, device="cuda", dtype=torch.float32)
    probe = TensorProbe(
        "example.activation",
        [RecordTensor()],
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = probe(x * 2)

    records: list[TensorSnapshot] = []
    for replay_index in range(1, 5):
        graph.replay()
        torch.cuda.synchronize()
        for snapshot in probe.snapshots():
            records.append(
                TensorSnapshot(
                    probe_name=snapshot.probe_name,
                    replay_index=replay_index,
                    invocation_index=snapshot.invocation_index,
                    shape=snapshot.shape,
                    dtype=snapshot.dtype,
                    device=snapshot.device,
                    tensor=snapshot.tensor.clone(),
                )
            )

    export_snapshots_to_tensorboard(
        writer,
        records,
        tag_prefix="helper/",
        write_histograms=True,
    )

    for snapshot in records:
        tensor = snapshot.tensor.float()
        flat = tensor.reshape(-1)
        step = snapshot.replay_index
        tag = f"manual/{snapshot.probe_name}"

        writer.add_scalar(f"{tag}/abs_max", flat.abs().max().item(), step)
        writer.add_scalar(f"{tag}/positive_count", (flat > 0).sum().item(), step)
        writer.add_histogram(f"{tag}/first_half", flat[: flat.numel() // 2], step)

    writer.flush()
    writer.close()
    probe.close()

    assert y is not None
    print(f"TensorBoard logs written to {args.logdir}")


if __name__ == "__main__":
    main()
