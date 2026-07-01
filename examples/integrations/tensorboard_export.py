"""Export replay snapshots to TensorBoard after synchronization.

Run with: python examples/integrations/tensorboard_export.py --logdir /tmp/tcgd-tb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from torch_cudagraph_debug.tensor_debug import RecordTensor, TensorProbe, TensorSnapshot
from torch_cudagraph_debug.tensor_debug.postprocess import (
    export_snapshots_to_tensorboard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("install TensorBoard with `pip install tensorboard`") from exc

    args = parse_args()
    static_x = torch.arange(8, device="cuda", dtype=torch.float32)
    probe = TensorProbe("tensorboard.activation", [RecordTensor()])
    writer = SummaryWriter(str(args.logdir))
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = probe(static_x * 2)

        records: list[TensorSnapshot] = []
        replay_stream = torch.cuda.current_stream()
        for _ in range(3):
            graph.replay()
            # Each query waits for the replay stream and returns an independent
            # CPU snapshot with the probe's current graph replay index.
            records.extend(probe.snapshots(synchronize=replay_stream))

        export_snapshots_to_tensorboard(
            writer,
            records,
            tag_prefix="tcgd/",
            write_histograms=True,
        )
        writer.flush()
        assert [snapshot.replay_index for snapshot in records] == [1, 2, 3]
        assert output is not None
        print(f"TensorBoard logs: {args.logdir.resolve()}")
    finally:
        writer.close()
        probe.close()


if __name__ == "__main__":
    main()
