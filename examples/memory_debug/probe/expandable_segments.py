"""Show cached and unmapped capacity in an expandable segment.

Run with:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python examples/memory_debug/probe/expandable_segments.py
"""

from __future__ import annotations

import torch

from torch_cudagraph_debug.memory_debug import MemoryProbe

MIB = 1024 * 1024
BUFFER_BYTES = 80 * MIB
DEFAULT_POOL_ID = (0, 0)


def _default_pool(comparison, device_index: int):
    return next(
        item
        for item in comparison.pool_comparisons
        if item.pool_key.device_index == device_index
        and item.pool_key.pool_id == DEFAULT_POOL_ID
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    # Allocator state and address lifecycle do not require allocator history.
    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.empty_cache()

    device_index = torch.cuda.current_device()
    stream = torch.cuda.current_stream(device_index)
    probe = MemoryProbe(
        name="expandable-segments",
        devices=device_index,
        synchronize=stream,
    )

    # The middle allocation spans complete allocator pages. Keeping its two
    # neighbors alive lets empty_cache() unmap an interior hole.
    left = torch.empty(BUFFER_BYTES, dtype=torch.uint8, device="cuda")
    middle = torch.empty(BUFFER_BYTES, dtype=torch.uint8, device="cuda")
    right = torch.empty(BUFFER_BYTES, dtype=torch.uint8, device="cuda")
    left.fill_(1)
    middle.fill_(2)
    right.fill_(3)
    fully_active = probe.snapshot()

    active_stats = next(
        stats
        for key, stats in fully_active.pool_stats.items()
        if key.device_index == device_index and key.pool_id == DEFAULT_POOL_ID
    )
    if active_stats.expandable_segment_count == 0:
        raise RuntimeError(
            "no expandable segments found; start a new process with "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        )

    del middle
    stream.synchronize()
    cached_free = probe.snapshot()
    cached_change = probe.compare(fully_active, cached_free)
    cached_pool = _default_pool(cached_change, device_index)
    cached_lifecycle = cached_pool.lifecycle
    assert cached_lifecycle is not None
    assert cached_pool.delta.expandable_reserved_bytes == 0
    assert cached_pool.delta.expandable_inactive_bytes >= BUFFER_BYTES
    assert cached_lifecycle.removed_segment_bytes == 0
    assert cached_lifecycle.became_inactive_bytes >= BUFFER_BYTES

    torch.cuda.empty_cache()
    after_unmap = probe.snapshot()
    unmap_change = probe.compare(cached_free, after_unmap)
    unmap_pool = _default_pool(unmap_change, device_index)
    unmap_lifecycle = unmap_pool.lifecycle
    assert unmap_lifecycle is not None
    assert unmap_pool.delta.expandable_reserved_bytes < 0
    assert unmap_pool.delta.expandable_inactive_bytes < 0
    assert (
        unmap_lifecycle.removed_segment_bytes
        == -unmap_pool.delta.expandable_reserved_bytes
    )
    assert unmap_lifecycle.new_segment_bytes == 0
    assert unmap_pool.candidate.segment_count == unmap_pool.reference.segment_count + 1
    assert left.numel() == BUFFER_BYTES
    assert right.numel() == BUFFER_BYTES

    print("=== Released allocation remains mapped ===")
    print(cached_change.to_text(include_unchanged=False))
    print()
    print("=== empty_cache() unmaps the interior hole ===")
    print(unmap_change.to_text(include_unchanged=False))


if __name__ == "__main__":
    main()
