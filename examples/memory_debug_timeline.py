"""Persist a CUDA allocator run and export its timeline charts."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from torch_cudagraph_debug.memory_debug import (
    AttributionOptions,
    MemoryRecorder,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("memory_debug_timeline.py requires CUDA")

    output_dir = Path("memory_debug_outputs") / f"run-{int(time.time())}"
    bundle_dir = output_dir / "rank0.tcgd-memory"
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
        ) as recorder:
            pool = torch.cuda.graph_pool_handle()
            recorder.mark("start")

            static_x = torch.randn(4096, 4096, device="cuda")
            static_w = torch.randn(4096, 4096, device="cuda")
            static_out = torch.empty_like(static_x)
            recorder.mark("after_static_alloc")

            static_out.copy_(static_x @ static_w)
            recorder.mark("after_warmup")

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                static_out.copy_(static_x @ static_w)
                recorder.mark("during_capture")

            recorder.mark("after_capture")
            graph.replay()
            recorder.mark("after_replay")

        run = recorder.result
        timeline = run.timeline(attribution=AttributionOptions(stacks=True))
        paths = timeline.write(output_dir / "timeline")
        print(timeline.to_text())
        print(f"bundle: {bundle_dir.resolve()}")
        print(f"report: {paths['html'].resolve()}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
