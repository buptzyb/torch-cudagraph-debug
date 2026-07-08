"""Trace allocations active at a point and allocations born in an interval.

Run with:
  python examples/memory_debug/recorder/allocation_lifetimes.py --output-dir /tmp/tcgd-life
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from torch_cudagraph_debug.memory_debug import (
    MemoryDisplayOptions,
    MemoryLifetimeOptions,
    MemoryLifetimeSelection,
    MemoryRecorder,
    MemoryRun,
)

MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    bundle_dir = output_dir / "lifetimes.tcgd-memory"

    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=100_000,
        clear_history=True,
    )
    try:
        warmup_state = torch.empty(64 * MIB, dtype=torch.uint8, device="cuda")
        persistent_state = torch.empty(32 * MIB, dtype=torch.uint8, device="cuda")

        recorder = MemoryRecorder(
            name="allocation-lifetimes",
            rank=0,
            bundle_dir=bundle_dir,
        )
        recorder.record_point("anchor")
        recorder.record_point("before_transient")

        transient_state = torch.empty(16 * MIB, dtype=torch.uint8, device="cuda")
        del transient_state
        gc.collect()
        torch.cuda.synchronize()
        recorder.record_point("after_transient")

        del warmup_state
        gc.collect()
        torch.cuda.synchronize()
        recorder.record_point("after_cleanup")
        recorder.finish()

        if args.record_only:
            print(f"bundle: {bundle_dir}")
            return

        run = MemoryRun.load(bundle_dir, cache_snapshots=False)
        options = MemoryLifetimeOptions(
            display=MemoryDisplayOptions(stack_depth=4, limit=20),
        )
        active = run.lifetimes(
            MemoryLifetimeSelection.active_at("anchor"),
            through="after_cleanup",
            options=options,
        )
        born = run.lifetimes(
            MemoryLifetimeSelection.born_between("before_transient", "after_transient"),
            through="after_cleanup",
            options=options,
        )
        active_paths = active.write(output_dir / "active-at-anchor")
        born_paths = born.write(output_dir / "born-in-interval")

        assert active.total_instance_bytes >= 96 * MIB
        assert (
            sum(item.owner_active_at_end_bytes for item in active.cohorts) >= 32 * MIB
        )
        assert born.total_instance_bytes >= 16 * MIB
        assert sum(item.born_bytes for item in born.cohorts) >= 16 * MIB
        assert sum(item.free_requested_bytes for item in active.cohorts) >= 64 * MIB
        assert sum(item.free_completed_bytes for item in active.cohorts) >= 64 * MIB
        assert sum(item.free_requested_bytes for item in born.cohorts) >= (16 * MIB)
        assert sum(item.free_completed_bytes for item in born.cohorts) >= (16 * MIB)
        assert active_paths["json"].is_file()
        assert active_paths["size_outcomes"].is_file()
        assert active_paths["free_request_stacks"].is_file()
        assert active_paths["free_completion_stacks"].is_file()
        assert born_paths["birth_stacks"].is_file()
        assert persistent_state.numel() == 32 * MIB

        print(active.to_text())
        print()
        print(born.to_text())
        print(f"reports: {output_dir}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
