"""Record rank-local bundles and compare two multi-rank groups.

Run with:
  torchrun --standalone --nproc-per-node=2 \
    examples/memory_debug/recorder/distributed_run_groups.py --output-dir /tmp/tcgd-groups
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch
import torch.distributed as dist

from torch_cudagraph_debug.memory_debug import (
    MemoryRecorder,
    MemoryRunGroup,
    compare_run_group_phases,
)

MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-only", action="store_true")
    return parser.parse_args()


def _record_rank_run(
    *,
    name: str,
    group_id: str,
    bundle_dir: Path,
    rank: int,
    world_size: int,
    use_graph_pool: bool,
) -> None:
    recorder = MemoryRecorder(
        name=name,
        rank=rank,
        group_id=group_id,
        world_size=world_size,
        bundle_dir=bundle_dir,
        run_metadata={"scenario": name, "example": "distributed-run-groups"},
    )
    static_state = torch.empty(4 * MIB, dtype=torch.uint8, device="cuda")
    recorder.record_point("phase_start", metadata={"rank": rank})

    allocation_bytes = (8 + rank) * MIB
    if use_graph_pool:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
            phase_state = torch.empty(
                allocation_bytes,
                dtype=torch.uint8,
                device="cuda",
            )
            phase_state.fill_(1)
        graph.replay()
    else:
        phase_state = torch.empty(
            allocation_bytes,
            dtype=torch.uint8,
            device="cuda",
        )
        phase_state.fill_(1)

    recorder.record_point("phase_end", metadata={"rank": rank})
    recorder.finish()
    assert static_state.numel() == 4 * MIB
    assert phase_state.numel() == allocation_bytes


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
        output_ok = torch.ones(1, dtype=torch.int32, device="cuda")
        if rank == 0:
            if output_dir.exists():
                output_ok.zero_()
            else:
                output_dir.mkdir(parents=True)
        dist.broadcast(output_ok, src=0)
        if output_ok.item() == 0:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        dist.barrier(device_ids=[local_rank])

        torch.cuda.memory._record_memory_history(enabled=None)
        torch.cuda.empty_cache()
        _record_rank_run(
            name="baseline",
            group_id="distributed-example-baseline",
            bundle_dir=output_dir / "baseline" / f"rank-{rank:05d}.tcgd-memory",
            rank=rank,
            world_size=world_size,
            use_graph_pool=False,
        )

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier(device_ids=[local_rank])
        _record_rank_run(
            name="candidate",
            group_id="distributed-example-candidate",
            bundle_dir=output_dir / "candidate" / f"rank-{rank:05d}.tcgd-memory",
            rank=rank,
            world_size=world_size,
            use_graph_pool=True,
        )
        dist.barrier(device_ids=[local_rank])

        if rank == 0 and not args.record_only:
            baseline = MemoryRunGroup.load(output_dir / "baseline")
            candidate = MemoryRunGroup.load(output_dir / "candidate")
            baseline_summary = baseline.summary()
            candidate_summary = candidate.summary()
            phase = compare_run_group_phases(
                baseline,
                candidate,
                baseline_start="phase_start",
                baseline_end="phase_end",
                candidate_start="phase_start",
                candidate_end="phase_end",
            )
            baseline_summary.write(output_dir / "baseline-summary")
            candidate_summary.write(output_dir / "candidate-summary")
            phase.write(output_dir / "group-phase")

            assert baseline.ranks == tuple(range(world_size))
            assert candidate.ranks == tuple(range(world_size))
            assert all(
                item.decomposition.components.identity_holds
                for item in phase.rank_decomposition
            )
            print(baseline_summary.to_text())
            print()
            print(phase.to_text(include_unchanged=False))
        elif rank == 0:
            print(f"baseline group: {output_dir / 'baseline'}")
            print(f"candidate group: {output_dir / 'candidate'}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
