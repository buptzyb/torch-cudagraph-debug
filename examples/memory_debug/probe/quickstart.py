"""Capture and compare allocator snapshots without creating a run.

Run with: python examples/memory_debug/probe/quickstart.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.memory_debug import MemoryProbe

MIB = 1024 * 1024


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.empty_cache()

    probe = MemoryProbe(name="memory-quickstart")
    static_state = torch.empty(4 * MIB, dtype=torch.uint8, device="cuda")
    before_capture = probe.snapshot()

    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        graph_state = torch.empty(16 * MIB, dtype=torch.uint8, device="cuda")
        during_capture = probe.snapshot()
        graph_state.fill_(1)

    after_capture = probe.snapshot()
    graph.replay()
    after_replay = probe.snapshot()

    capture_growth = probe.compare(before_capture, after_capture)
    replay_delta = probe.compare(after_capture, after_replay)
    private_total = next(
        item
        for item in capture_growth.allocator_scope_comparisons
        if item.scope == "private"
    )
    assert private_total.candidate.active_bytes >= 16 * MIB
    assert static_state.numel() == 4 * MIB
    assert graph_state.numel() == 16 * MIB
    assert during_capture.snapshot_index == 1

    print(capture_growth.to_text(include_unchanged=False))
    print()
    print(replay_delta.to_text())


if __name__ == "__main__":
    main()
