"""Explain inactive memory retained by a CUDA Graph private pool.

Run with: python examples/memory_debug/probe/private_pool_inactive.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.memory_debug import MemoryProbe

MIB = 1024 * 1024
SCRATCH_BYTES = 64 * MIB
DEFAULT_POOL = (0, 0)


def _format_pool_id(pool_id: tuple[object, ...]) -> str:
    return "pool[" + ",".join(str(item) for item in pool_id) + "]"


def _format_bytes(value: int) -> str:
    return f"{value / MIB:.2f} MiB"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    # Allocator state is sufficient; stack and event history are not needed.
    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.empty_cache()

    probe = MemoryProbe(name="private-pool-inactive")
    before_capture = probe.snapshot()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
        scratch = torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device="cuda")
        scratch.fill_(1)

    # The captured kernel retains scratch's address even after its Python tensor
    # is gone. The block becomes inactive, while the graph keeps the pool alive.
    del scratch
    after_capture = probe.snapshot()

    graph.replay()
    after_replay = probe.snapshot()

    private_pool_ids = sorted(
        pool_id
        for pool_id in set(after_capture.pool_stats) - set(before_capture.pool_stats)
        if pool_id != DEFAULT_POOL
    )
    if len(private_pool_ids) != 1:
        raise RuntimeError(
            f"expected one new private pool, found {private_pool_ids}"
        )

    pool_id = private_pool_ids[0]
    captured = after_capture.pool_stats[pool_id]
    replayed = after_replay.pool_stats[pool_id]

    assert captured.inactive_bytes == captured.reserved_bytes - captured.active_bytes
    assert captured.inactive_bytes >= SCRATCH_BYTES
    assert replayed.reserved_bytes == captured.reserved_bytes
    assert replayed.active_bytes == captured.active_bytes

    print(_format_pool_id(pool_id))
    for label, stats in (("after capture", captured), ("after replay", replayed)):
        print(f"  {label}:")
        print(f"    reserved: {_format_bytes(stats.reserved_bytes)}")
        print(f"    active:   {_format_bytes(stats.active_bytes)}")
        print(f"    inactive: {_format_bytes(stats.inactive_bytes)}")

    print(
        "inactive memory is reusable within the private pool but remains "
        "reserved while the graph is alive"
    )


if __name__ == "__main__":
    main()
