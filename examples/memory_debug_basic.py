"""Inspect CUDA graph allocator growth at explicit probe points."""

from __future__ import annotations

import torch

from torch_cudagraph_debug.memory_debug import (
    AttributionOptions,
    MemoryRecorder,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("memory_debug_basic.py requires CUDA")

    torch.cuda.memory._record_memory_history(
        enabled="state",
        context="state",
        stacks="python",
        clear_history=True,
    )
    try:
        recorder = MemoryRecorder(name="capture")
        pool = torch.cuda.graph_pool_handle()

        static_x = torch.randn(4096, 4096, device="cuda")
        static_w = torch.randn(4096, 4096, device="cuda")
        static_out = torch.empty_like(static_x)

        static_out.copy_(static_x @ static_w)
        recorder.mark("before_capture")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            graph_tmp_a = torch.empty_like(static_x)
            graph_tmp_a.copy_(static_x)
            recorder.mark("during_capture_after_first_alloc")

            graph_tmp_b = torch.empty((2048, 4096), device="cuda")
            graph_tmp_c = torch.empty((1024, 4096), device="cuda")
            graph_tmp_b.copy_(static_x[:2048])
            graph_tmp_c.copy_(static_x[:1024])
            static_out.copy_(graph_tmp_a)

        graph_buffers = (graph_tmp_a, graph_tmp_b, graph_tmp_c)
        recorder.mark("after_capture")

        graph.replay()
        recorder.mark("after_replay")
        run = recorder.finish()

        stacks = AttributionOptions(stacks=True)
        print(
            run.compare(
                "before_capture",
                "during_capture_after_first_alloc",
                attribution=stacks,
            ).to_text()
        )
        print()
        print(
            run.compare(
                "during_capture_after_first_alloc",
                "after_capture",
                attribution=stacks,
            ).to_text()
        )
        print()
        print(
            run.compare(
                "after_capture",
                "after_replay",
            ).to_text()
        )
        assert graph_buffers
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
