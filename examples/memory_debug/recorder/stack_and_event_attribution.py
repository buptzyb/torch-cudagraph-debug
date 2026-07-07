"""Attribute allocator growth to stacks, events, and lifetimes.

Run with:
  python examples/memory_debug/recorder/stack_and_event_attribution.py \
    --output-dir /tmp/tcgd-stack-events
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryRecorder,
    MemoryRun,
)

MIB = 1024 * 1024


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
    bundle_dir = output_dir / "full-history.tcgd-memory"

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
        recorder = MemoryRecorder(
            name="stack-and-event-attribution",
            rank=0,
            bundle_dir=bundle_dir,
        )
        recorder.record_point("before_work")

        persistent = torch.empty(32 * MIB, dtype=torch.uint8, device="cuda")
        transient = torch.empty(16 * MIB, dtype=torch.uint8, device="cuda")
        del transient
        gc.collect()
        torch.cuda.synchronize()
        recorder.record_point("after_work")

        del persistent
        gc.collect()
        torch.cuda.synchronize()
        recorder.record_point("after_cleanup")
        recorder.finish()

        run = MemoryRun.load(bundle_dir, cache_snapshots=False)
        # These are text/HTML defaults. Python objects, JSON, and CSV retain
        # complete stacks and every attribution row.
        options = MemoryAttributionOptions(
            stacks=True,
            events=True,
            lifetimes=True,
            on_missing="error",
            stack_depth=4,
            limit=20,
        )

        comparison = run.compare(
            "before_work",
            "after_work",
            attribution=options,
        )
        assert comparison.candidate_stack_coverage is not None
        assert comparison.candidate_stack_coverage.attributed_bytes >= 32 * MIB
        assert comparison.allocation_stack_comparisons
        largest_stack_delta = max(
            comparison.allocation_stack_comparisons,
            key=lambda item: item.delta_size_bytes,
        )
        # stack_frames is the lossless programmatic form used by JSON and CSV.
        assert largest_stack_delta.stack_frames
        assert comparison.events_available is True
        assert comparison.events_complete is True
        assert comparison.allocator_events
        assert comparison.allocation_lifetimes is not None
        assert comparison.allocation_lifetimes.history_complete is True
        assert (
            sum(
                item.event_exact_birth_bytes
                for item in comparison.allocation_lifetimes.cohorts
            )
            >= 48 * MIB
        )
        comparison_paths = comparison.write(output_dir / "comparison-report")
        assert comparison_paths["events"].is_file()
        assert comparison_paths["cohorts"].is_file()

        timeline = run.timeline(attribution=options)
        assert len(timeline.point_comparisons) == 2
        assert all(
            item.events_available and item.events_complete
            for item in timeline.point_comparisons
        )
        assert timeline.allocation_lifetimes is not None
        assert timeline.allocation_lifetimes.history_complete is True
        assert (
            sum(
                item.event_exact_free_completed_bytes
                for item in timeline.allocation_lifetimes.cohorts
            )
            >= 48 * MIB
        )
        assert (
            sum(
                item.event_exact_free_requested_bytes
                for item in timeline.allocation_lifetimes.cohorts
            )
            >= 48 * MIB
        )
        timeline_paths = timeline.write(
            output_dir / "timeline-report",
            include_unchanged=False,
        )
        assert timeline_paths["events"].is_file()
        assert timeline_paths["cohorts"].is_file()

        print("=== Stack and event attribution ===")
        print(comparison.to_text(include_unchanged=False))
        print("largest allocation-stack delta frames:")
        for frame in largest_stack_delta.stack_frames:
            print(f"  {frame['filename']}:{frame['line']}:{frame['name']}")
        print()
        print("=== Timeline with embedded lifetimes ===")
        print(timeline.to_text(include_unchanged=False))
        print(f"bundle: {bundle_dir}")
        print(f"HTML report: {timeline_paths['html']}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
