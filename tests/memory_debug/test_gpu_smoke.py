from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch

from torch_cudagraph_debug.memory_debug import (
    AttributionOptions,
    MemoryRecorder,
    MemoryRun,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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
        recorder.mark("before")
        tensor = _allocate_with_named_stack(1_000_033)
        torch.cuda.synchronize()
        recorder.mark("after")
        run = recorder.finish()
        comparison = run.compare(
            "before",
            "after",
            attribution=AttributionOptions(
                events=True,
                on_missing="error",
            ),
        )
        assert comparison.events_available is True
        assert comparison.events_complete is True
        assert any(event.action == "alloc" for event in comparison.allocator_events)
        assert any(
            "_allocate_with_named_stack" in event.stack_key
            for event in comparison.allocator_events
            if event.action == "alloc"
        )
        del tensor
    finally:
        _disable_history()


def test_real_full_history_attributes_allocation_lifetime_release() -> None:
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
        recorder.mark("anchor")
        del tensor
        gc.collect()
        torch.cuda.synchronize()
        recorder.mark("released")

        report = recorder.finish().lifetimes(
            "anchor",
            through="released",
            attribution=AttributionOptions(
                events=True,
                on_missing="error",
                stack_depth=4,
            ),
        )

        assert report.history_available is True
        assert report.history_complete is True
        matching = [
            cohort
            for cohort in report.cohorts
            if cohort.event_exact_release_bytes > 0
            and cohort.snapshot_inferred_release_bytes == 0
            and cohort.points[0].active_bytes > 0
            and cohort.points[-1].active_bytes == 0
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
        recorder.mark("before")

        transient = _allocate_with_named_stack(1_000_057)
        del transient
        gc.collect()
        torch.cuda.synchronize()
        recorder.mark("after")

        report = recorder.finish().lifetimes(
            born_between=("before", "after"),
            attribution=AttributionOptions(
                events=True,
                on_missing="error",
                stack_depth=4,
            ),
        )

        matching = [
            cohort
            for cohort in report.cohorts
            if "_allocate_with_named_stack" in cohort.stack_key
            and cohort.event_exact_birth_bytes > 0
            and cohort.event_exact_release_bytes > 0
            and cohort.peak_live_bytes > 0
            and all(point.active_bytes == 0 for point in cohort.points)
        ]
        assert matching, report.to_text()
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
        recorder.mark("before_capture")
        pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph, pool=pool):
            graph_tmp_a = torch.empty_like(static_x)
            graph_tmp_a.copy_(static_x)
            during = recorder.mark("during_capture")
            graph_tmp_b = torch.empty((512, 1024), device="cuda")
            graph_tmp_b.copy_(static_x[:512])
            static_out.copy_(graph_tmp_a)

        graph_buffers = (graph_tmp_a, graph_tmp_b)
        after = recorder.mark("after_capture")
        graph.replay()
        after_replay = recorder.mark("after_replay")
        run = recorder.finish()

        capture_comparison = run.compare(
            "before_capture",
            "after_capture",
            attribution=AttributionOptions(stacks=True),
        )
        replay_comparison = run.compare(
            "after_capture",
            "after_replay",
        )
        assert during.label == "during_capture"
        assert any(pool_id != (0, 0) for pool_id in after.pools)
        assert any(item.delta.reserved_bytes > 0 for item in capture_comparison.pools)
        assert all(item.delta.active_bytes == 0 for item in replay_comparison.pools)

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
            attribution=AttributionOptions(stacks=True),
        )
        assert loaded_comparison.pool_rows() == capture_comparison.pool_rows()
        assert graph_buffers
        assert after_replay.raw_snapshot()["segments"]
    finally:
        _disable_history()
