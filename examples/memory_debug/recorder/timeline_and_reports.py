"""Persist a run, reload it, and export a timeline in every report format.

Run with:
  python examples/memory_debug/recorder/timeline_and_reports.py --output-dir /tmp/tcgd-timeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryDisplayOptions,
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
    bundle_dir = output_dir / "timeline.tcgd-memory"

    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(
        enabled="state",
        context="state",
        stacks="python",
        clear_history=True,
    )
    try:
        with MemoryRecorder(
            name="timeline",
            rank=0,
            bundle_dir=bundle_dir,
            run_metadata={"example": "timeline-and-reports"},
        ) as recorder:
            recorder.record_point("start")
            static_state = torch.empty(8 * MIB, dtype=torch.uint8, device="cuda")
            recorder.record_point("after_static_alloc")

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                graph_state = torch.empty(12 * MIB, dtype=torch.uint8, device="cuda")
                graph_state.fill_(1)
                recorder.record_point("during_capture")

            recorder.record_point("after_capture")
            graph.replay()
            recorder.record_point("after_replay")
            assert not recorder.preview().complete

        loaded = MemoryRun.load(bundle_dir, cache_snapshots=False)
        assert loaded.complete
        assert tuple(point.label for point in loaded.points) == (
            "start",
            "after_static_alloc",
            "during_capture",
            "after_capture",
            "after_replay",
        )
        timeline = loaded.timeline(
            attribution=MemoryAttributionOptions(
                stacks=True,
                display=MemoryDisplayOptions(stack_depth=4),
            )
        )
        paths = timeline.write(
            output_dir / "timeline-report",
            include_unchanged=False,
        )
        for name in (
            "text",
            "json",
            "html",
            "allocator_scopes",
            "pools",
            "observations",
        ):
            assert paths[name].is_file(), name
        assert static_state.numel() == 8 * MIB
        assert graph_state.numel() == 12 * MIB

        print(timeline.to_text(include_unchanged=False))
        print(f"bundle: {bundle_dir}")
        print(f"HTML report: {paths['html']}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
