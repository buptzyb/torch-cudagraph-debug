"""Record capture points and compare allocator state without history.

Run with: python examples/memory_debug/quickstart.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.memory_debug import MemoryRecorder


MIB = 1024 * 1024


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.empty_cache()

    recorder = MemoryRecorder(name="memory-quickstart")
    static_state = torch.empty(4 * MIB, dtype=torch.uint8, device="cuda")
    recorder.mark("before_capture")

    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        graph_state = torch.empty(16 * MIB, dtype=torch.uint8, device="cuda")
        recorder.mark("during_capture")
        graph_state.fill_(1)

    recorder.mark("after_capture")
    graph.replay()
    recorder.mark("after_replay")

    partial = recorder.snapshot_run()
    assert not partial.complete
    run = recorder.finish()
    assert run.complete

    capture_growth = run.compare("before_capture", "after_capture")
    replay_delta = run.compare("after_capture", "after_replay")
    private_total = next(
        item for item in capture_growth.totals if item.scope == "private"
    )
    assert private_total.after.active_bytes >= 16 * MIB
    assert static_state.numel() == 4 * MIB
    assert graph_state.numel() == 16 * MIB

    print(capture_growth.to_text(include_unchanged=False))
    print()
    print(replay_delta.to_text())


if __name__ == "__main__":
    main()
