"""Record and compare rank-local eager and CUDA Graph tensor bundles.

Run with:
  torchrun --standalone --nproc-per-node=2 \
    examples/tensor_debug/recorder/distributed_run_groups.py \
    --output-dir /tmp/tcgd-tensor-groups
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from torch_cudagraph_debug.tensor_debug import (
    TensorRecorder,
    TensorRunGroup,
    compare_run_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-only", action="store_true")
    return parser.parse_args()


def observed_forward(
    inputs: torch.Tensor,
    recorder: TensorRecorder,
) -> torch.Tensor:
    hidden = recorder.observe(inputs + 1, name="hidden")
    return recorder.observe(hidden.square(), name="output")


def record_eager(
    *,
    bundle_dir: Path,
    inputs: torch.Tensor,
    rank: int,
    world_size: int,
) -> None:
    with TensorRecorder(
        execution="eager",
        name="eager",
        bundle_dir=bundle_dir,
        rank=rank,
        group_id="tensor-eager-example",
        world_size=world_size,
        run_metadata={"scenario": "eager-reference"},
    ) as recorder:
        with recorder.record_point("forward"):
            observed_forward(inputs, recorder)


def record_cuda_graph(
    *,
    bundle_dir: Path,
    inputs: torch.Tensor,
    rank: int,
    world_size: int,
) -> None:
    recorder = TensorRecorder(
        execution="cuda_graph",
        name="cuda-graph",
        bundle_dir=bundle_dir,
        rank=rank,
        group_id="tensor-cuda-graph-example",
        world_size=world_size,
        run_metadata={"scenario": "cuda-graph-candidate"},
    )
    graph = torch.cuda.CUDAGraph()
    replay_stream = torch.cuda.current_stream()
    try:
        with torch.cuda.graph(graph):
            observed_forward(inputs, recorder)
        with recorder.record_point("forward", synchronize=replay_stream):
            graph.replay()
        recorder.finish()
    finally:
        del graph
        recorder.close(synchronize=replay_stream)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    output_dir = args.output_dir.resolve()

    try:
        output_ok = torch.ones((), dtype=torch.int32, device="cuda")
        if rank == 0:
            if output_dir.exists():
                output_ok.zero_()
            else:
                output_dir.mkdir(parents=True)
        dist.broadcast(output_ok, src=0)
        if output_ok.item() == 0:
            raise FileExistsError(f"output directory already exists: {output_dir}")

        inputs = (
            torch.arange(
                8,
                dtype=torch.float32,
                device="cuda",
            ).reshape(2, 4)
            + rank
        )
        record_eager(
            bundle_dir=output_dir / "eager" / f"rank-{rank:05d}.tcgd-tensor",
            inputs=inputs,
            rank=rank,
            world_size=world_size,
        )
        record_cuda_graph(
            bundle_dir=(output_dir / "cuda-graph" / f"rank-{rank:05d}.tcgd-tensor"),
            inputs=inputs,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier(device_ids=[local_rank])

        if rank == 0 and not args.record_only:
            eager = TensorRunGroup.load(output_dir / "eager")
            cuda_graph = TensorRunGroup.load(output_dir / "cuda-graph")
            eager.summary().write(output_dir / "eager-summary")
            comparison = compare_run_groups(eager, cuda_graph)
            comparison.write(output_dir / "group-comparison")
            comparison_ok = torch.tensor(
                int(comparison.ok),
                dtype=torch.int32,
                device="cuda",
            )
            if not comparison.ok:
                print(comparison.to_text())
        else:
            comparison_ok = torch.ones((), dtype=torch.int32, device="cuda")
        dist.broadcast(comparison_ok, src=0)
        if comparison_ok.item() != 1:
            raise RuntimeError("eager and CUDA Graph tensor groups differ")
        if rank == 0:
            print(f"eager group: {output_dir / 'eager'}")
            print(f"CUDA Graph group: {output_dir / 'cuda-graph'}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
