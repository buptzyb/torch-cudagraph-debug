"""Compare allocator endpoints captured by independent probes.

Run with: python examples/memory_debug/probe/snapshot_comparison.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.memory_debug import MemoryProbe, compare_snapshots

MIB = 1024 * 1024


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.empty_cache()

    baseline_probe = MemoryProbe("before-capture")
    before = baseline_probe.snapshot()

    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        graph_state = torch.empty(16 * MIB, dtype=torch.uint8, device="cuda")
        graph_state.fill_(1)

    candidate_probe = MemoryProbe("after-capture")
    after = candidate_probe.snapshot()
    comparison = compare_snapshots(before, after)

    private_total = next(
        item
        for item in comparison.allocator_scope_comparisons
        if item.scope == "private"
    )
    assert private_total.candidate.active_bytes >= 16 * MIB
    assert graph_state.numel() == 16 * MIB
    assert "cross probe" in comparison.to_text()

    print(comparison.to_text(include_unchanged=False))


if __name__ == "__main__":
    main()
