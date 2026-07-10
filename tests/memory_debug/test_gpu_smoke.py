from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch

from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryDisplayOptions,
    MemoryHistoryTruncatedError,
    MemoryLifetimeOptions,
    MemoryLifetimeSelection,
    MemoryRecorder,
    MemoryRun,
    MemoryStats,
)
from torch_cudagraph_debug.memory_debug.aggregation import summarize_devices

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def _active_block(snapshot: dict[str, object], address: int):
    for segment in snapshot["segments"]:  # type: ignore[index]
        for block in segment.get("blocks", []):
            if block.get("address") == address and str(
                block.get("state", "")
            ).startswith("active"):
                return block
    raise AssertionError(f"active block for address {address} was not found")


def _allocate_with_named_stack(size: int) -> torch.Tensor:
    return torch.empty(size, device="cuda")


def _disable_history() -> None:
    torch.cuda.memory._record_memory_history(enabled=None)


def test_real_snapshot_stack_history_levels() -> None:
    _disable_history()
    torch.cuda.empty_cache()

    without_history = torch.empty(1_000_003, device="cuda")
    block_without = _active_block(
        torch.cuda.memory._snapshot(),
        without_history.data_ptr(),
    )
    assert block_without.get("frames", []) == []

    del without_history
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    torch.cuda.memory._record_memory_history(
        enabled="state",
        context="state",
        stacks="python",
        clear_history=True,
    )
    try:
        with_history = _allocate_with_named_stack(1_000_019)
        block_with = _active_block(
            torch.cuda.memory._snapshot(),
            with_history.data_ptr(),
        )
        assert block_with.get("frames")
        assert any(
            frame.get("name") == "_allocate_with_named_stack"
            for frame in block_with["frames"]
        )
    finally:
        _disable_history()


def test_real_full_history_produces_marker_delimited_events() -> None:
    _disable_history()
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=10_000,
        clear_history=True,
    )
    try:
        recorder = MemoryRecorder()
        recorder.record_point("before")
        tensor = _allocate_with_named_stack(1_000_033)
        torch.cuda.synchronize()
        recorder.record_point("after")
        run = recorder.finish()
        comparison = run.compare(
            "before",
            "after",
            attribution=MemoryAttributionOptions(
                events=True,
            ),
        )
        assert comparison.attribution_status.events.available is True
        assert comparison.attribution_status.events.complete is True
        assert any(event.action == "alloc" for event in comparison.allocator_events)
        assert any(
            "_allocate_with_named_stack" in event.stack_key
            for event in comparison.allocator_events
            if event.action == "alloc"
        )
        del tensor
    finally:
        _disable_history()


def test_real_full_history_attributes_allocation_free_completion() -> None:
    _disable_history()
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=10_000,
        clear_history=True,
    )
    try:
        tensor = _allocate_with_named_stack(1_000_043)
        torch.cuda.synchronize()

        recorder = MemoryRecorder()
        recorder.record_point("anchor")
        del tensor
        gc.collect()
        torch.cuda.synchronize()
        recorder.record_point("completed")

        report = recorder.finish().lifetimes(
            MemoryLifetimeSelection.active_at("anchor"),
            through="completed",
            options=MemoryLifetimeOptions(
                display=MemoryDisplayOptions(stack_depth=4),
            ),
        )

        assert report.history_available is True
        assert report.history_complete is True
        matching = [
            cohort
            for cohort in report.cohorts
            if cohort.free_completed_bytes > 0
            and cohort.points[0].active_bytes > 0
            and cohort.points[-1].active_bytes == 0
        ]
        assert matching, report.to_text()
    finally:
        _disable_history()


def test_real_cross_stream_free_waits_for_completion() -> None:
    _disable_history()
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=10_000,
        clear_history=True,
    )
    try:
        work_stream = torch.cuda.current_stream()
        side_stream = torch.cuda.Stream()
        tensor = _allocate_with_named_stack(1_000_049)
        work_stream.synchronize()

        recorder = MemoryRecorder(synchronize=work_stream)
        recorder.record_point("owned")

        torch.cuda._sleep(500_000_000)
        side_stream.wait_stream(work_stream)
        tensor.record_stream(side_stream)
        del tensor
        gc.collect()
        recorder.record_point("awaiting", synchronize=False)

        side_stream.synchronize()
        # Stream completion is observed when the allocator next polls pending frees.
        allocator_poll = torch.empty(1, dtype=torch.uint8, device="cuda")
        del allocator_poll
        gc.collect()
        recorder.record_point("completed", synchronize=False)
        report = recorder.finish().lifetimes(
            MemoryLifetimeSelection.active_at("owned"),
            options=MemoryLifetimeOptions(
                display=MemoryDisplayOptions(stack_depth=4),
            ),
        )

        matching = [
            cohort
            for cohort in report.cohorts
            if "_allocate_with_named_stack" in cohort.stack_key
            and cohort.points[0].owner_active_bytes > 0
            and cohort.points[1].awaiting_free_bytes > 0
            and cohort.points[2].active_bytes == 0
            and cohort.free_requested_bytes > 0
            and cohort.free_completed_bytes > 0
        ]
        assert matching, report.to_text()
    finally:
        _disable_history()


def test_real_full_history_keeps_event_only_born_and_freed_generation() -> None:
    _disable_history()
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=10_000,
        clear_history=True,
    )
    try:
        recorder = MemoryRecorder()
        recorder.record_point("before")

        transient = _allocate_with_named_stack(1_000_057)
        del transient
        gc.collect()
        torch.cuda.synchronize()
        recorder.record_point("after")

        report = recorder.finish().lifetimes(
            MemoryLifetimeSelection.born_between("before", "after"),
            options=MemoryLifetimeOptions(
                display=MemoryDisplayOptions(stack_depth=4),
            ),
        )

        matching = [
            cohort
            for cohort in report.cohorts
            if "_allocate_with_named_stack" in cohort.stack_key
            and cohort.born_bytes > 0
            and cohort.free_completed_bytes > 0
            and cohort.event_unreusable_peak_bytes > 0
            and all(point.active_bytes == 0 for point in cohort.points)
        ]
        assert matching, report.to_text()
    finally:
        _disable_history()


def test_real_ring_buffer_overwrite_raises_truncated_error() -> None:
    _disable_history()
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=600,
        clear_history=True,
    )
    try:
        recorder = MemoryRecorder()
        recorder.record_point("before")

        # Each alloc/free pair appends several trace entries; overflowing the
        # 600-entry ring buffer evicts the "before" boundary marker.
        for _ in range(600):
            transient = torch.empty(4_096, device="cuda")
            del transient
        gc.collect()
        torch.cuda.synchronize()
        recorder.record_point("after")
        run = recorder.finish()

        with pytest.raises(MemoryHistoryTruncatedError) as excinfo:
            run.lifetimes(
                MemoryLifetimeSelection.born_between("before", "after"),
                options=MemoryLifetimeOptions(
                    display=MemoryDisplayOptions(stack_depth=4),
                ),
            )
        message = str(excinfo.value)
        assert "before" in message
        assert "after" in message
        assert "max_entries" in message
    finally:
        _disable_history()


def test_graph_pool_capture_and_json_bundle_round_trip(
    tmp_path: Path,
) -> None:
    _disable_history()
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(
        enabled="state",
        context="state",
        stacks="python",
        clear_history=True,
    )
    try:
        static_x = torch.randn(1024, 1024, device="cuda")
        static_out = torch.empty_like(static_x)
        static_out.copy_(static_x)
        torch.cuda.synchronize()

        bundle = tmp_path / "graph.tcgd-memory"
        recorder = MemoryRecorder(bundle_dir=bundle, name="graph")
        recorder.record_point("before_capture")
        pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph, pool=pool):
            graph_tmp_a = torch.empty_like(static_x)
            graph_tmp_a.copy_(static_x)
            during = recorder.record_point("during_capture")
            graph_tmp_b = torch.empty((512, 1024), device="cuda")
            graph_tmp_b.copy_(static_x[:512])
            static_out.copy_(graph_tmp_a)

        graph_buffers = (graph_tmp_a, graph_tmp_b)
        after = recorder.record_point("after_capture")
        graph.replay()
        after_replay = recorder.record_point("after_replay")
        run = recorder.finish()

        # Real snapshots must never trip the schema-drift warnings.
        assert not any(
            "schema drift" in warning
            for point in run.points
            for warning in point.warnings
        )

        # CUDA Runtime sampling accompanies allocator snapshots, including
        # the point taken while current-stream capture is active.
        device = torch.cuda.current_device()
        for point in run.points:
            sample = point.device_memory[device]
            assert sample.total_bytes > 0
            assert sample.free_bytes + sample.used_bytes == sample.total_bytes
        assert not any(
            "could not sample device memory" in warning for warning in during.warnings
        )

        capture_comparison = run.compare(
            "before_capture",
            "after_capture",
            attribution=MemoryAttributionOptions(stacks=True),
        )
        replay_comparison = run.compare(
            "after_capture",
            "after_replay",
        )
        assert during.label == "during_capture"
        assert any(key.pool_id != (0, 0) for key in after.pool_stats)
        assert any(
            item.delta.reserved_bytes > 0
            for item in capture_comparison.pool_comparisons
        )
        assert all(
            item.delta.active_bytes == 0 for item in replay_comparison.pool_comparisons
        )

        loaded = MemoryRun.load(bundle)
        assert [point.label for point in loaded.points] == [
            "before_capture",
            "during_capture",
            "after_capture",
            "after_replay",
        ]
        loaded_comparison = loaded.compare(
            "before_capture",
            "after_capture",
            attribution=MemoryAttributionOptions(stacks=True),
        )
        assert (
            loaded_comparison.pool_comparison_rows()
            == capture_comparison.pool_comparison_rows()
        )
        assert [dict(point.device_memory) for point in loaded.points] == [
            dict(point.device_memory) for point in run.points
        ]
        assert graph_buffers
        assert after_replay.allocator_state()["segments"]
    finally:
        _disable_history()


def test_real_device_memory_sampling_tracks_allocator_state() -> None:
    _disable_history()
    torch.cuda.empty_cache()
    recorder = MemoryRecorder(name="device-memory")
    recorder.record_point("baseline")
    payload = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    recorder.record_point("grown")
    run = recorder.finish()

    device = torch.cuda.current_device()
    for point in run.points:
        sample = point.device_memory[device]
        assert sample.free_bytes + sample.used_bytes == sample.total_bytes
        reserved = (
            summarize_devices(point.pool_stats)
            .get(device, MemoryStats())
            .reserved_bytes
        )
        # Device-wide usage includes this process's reserved segments, the
        # CUDA context, and any other process on a shared GPU.
        assert sample.used_bytes >= reserved

    comparison = run.compare("baseline", "grown")
    (row,) = [
        item
        for item in comparison.device_comparisons
        if item.reference_device_index == item.candidate_device_index == device
    ]
    assert row.delta_used_bytes is not None
    assert row.delta_total_bytes is not None
    # Our own allocator growth is deterministic even on a shared GPU; the
    # device-wide used delta is not, so only the reserved delta is asserted.
    assert row.delta_allocator_reserved_bytes >= payload.numel()
    text = comparison.to_text()
    assert "CUDA scope: device-wide, includes other processes" in text
    assert f"device[{device}]" in text
    assert "residual:" in text
