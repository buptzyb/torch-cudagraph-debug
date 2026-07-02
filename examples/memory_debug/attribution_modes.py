"""Compare allocator attribution behavior across history configurations.

Run with:
  python examples/memory_debug/attribution_modes.py --output-dir /tmp/tcgd-attribution
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryHistoryError,
    MemoryRecorder,
    MemoryRun,
)

MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _all_total(comparison):
    return next(
        item for item in comparison.allocator_scope_comparisons if item.scope == "all"
    )


def _snapshot_only(output_dir: Path) -> None:
    bundle_dir = output_dir / "snapshot-only.tcgd-memory"
    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.empty_cache()

    recorder = MemoryRecorder(
        name="snapshot-only-attribution",
        rank=0,
        bundle_dir=bundle_dir,
    )
    recorder.record_point("before")
    visible = torch.empty(16 * MIB, dtype=torch.uint8, device="cuda")
    recorder.record_point("after_alloc")
    del visible
    gc.collect()
    torch.cuda.synchronize()
    recorder.record_point("after_free")
    recorder.finish()

    run = MemoryRun.load(bundle_dir, cache_snapshots=False)

    state = run.compare("before", "after_alloc")
    assert _all_total(state).delta.allocated_bytes >= 16 * MIB

    warned = run.compare(
        "before",
        "after_alloc",
        attribution=MemoryAttributionOptions(
            stacks=True,
            events=True,
            on_missing="warn",
            stack_depth=4,
        ),
    )
    assert warned.candidate_stack_coverage is not None
    assert warned.candidate_stack_coverage.unattributed_bytes >= 16 * MIB
    assert warned.events_available is False
    assert any("coverage is incomplete" in item for item in warned.warnings)
    assert any("allocator event history" in item for item in warned.warnings)
    warning_paths = warned.write(output_dir / "no-history-warning-comparison")
    assert warning_paths["json"].is_file()

    try:
        run.compare(
            "before",
            "after_alloc",
            attribution=MemoryAttributionOptions(
                stacks=True,
                events=True,
                on_missing="error",
                stack_depth=4,
            ),
        )
    except MemoryHistoryError as error:
        print(f"expected strict-policy error: {error}")
    else:
        raise AssertionError("strict missing-history policy did not raise")

    inferred = run.lifetimes(
        born_between=("before", "after_alloc"),
        through="after_free",
        attribution=MemoryAttributionOptions(
            events=False,
            on_missing="warn",
            stack_depth=4,
        ),
    )
    assert inferred.history_requested is False
    assert sum(item.snapshot_inferred_birth_bytes for item in inferred.cohorts) >= (
        16 * MIB
    )
    assert sum(item.snapshot_inferred_release_bytes for item in inferred.cohorts) >= (
        16 * MIB
    )
    assert sum(item.event_exact_birth_bytes for item in inferred.cohorts) == 0
    assert sum(item.event_exact_release_bytes for item in inferred.cohorts) == 0
    assert any(
        release.confidence == "snapshot_inferred"
        for cohort in inferred.cohorts
        for release in cohort.releases
    )
    assert any(
        "transient allocations may be missing" in item for item in inferred.warnings
    )
    inferred_paths = inferred.write(output_dir / "snapshot-only-lifetimes")
    assert inferred_paths["json"].is_file()

    print()
    print("=== No allocator history: warning policy ===")
    print(warned.to_text(include_unchanged=False))
    print()
    print("=== Snapshot-only lifetime inference ===")
    print(inferred.to_text())


def _full_history(output_dir: Path) -> None:
    bundle_dir = output_dir / "full-history.tcgd-memory"
    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=100_000,
        clear_history=True,
    )

    recorder = MemoryRecorder(
        name="full-history-attribution",
        rank=0,
        bundle_dir=bundle_dir,
    )
    recorder.record_point("before_work")

    persistent = torch.empty(32 * MIB, dtype=torch.uint8, device="cuda")
    transient = torch.empty(16 * MIB, dtype=torch.uint8, device="cuda")
    del transient
    gc.collect()
    torch.cuda.synchronize()
    recorder.record_point("after_work")

    del persistent
    gc.collect()
    torch.cuda.synchronize()
    recorder.record_point("after_cleanup")
    recorder.finish()

    run = MemoryRun.load(bundle_dir, cache_snapshots=False)
    options = MemoryAttributionOptions(
        stacks=True,
        events=True,
        lifetimes=True,
        on_missing="error",
        stack_depth=4,
        limit=20,
    )

    comparison = run.compare(
        "before_work",
        "after_work",
        attribution=options,
    )
    assert comparison.reference_stack_coverage is not None
    assert comparison.candidate_stack_coverage is not None
    assert comparison.candidate_stack_coverage.attributed_bytes >= 32 * MIB
    assert comparison.allocation_stack_comparisons
    assert comparison.events_available is True
    assert comparison.events_complete is True
    assert comparison.allocator_events
    assert comparison.allocation_lifetimes is not None
    assert comparison.allocation_lifetimes.history_available is True
    assert comparison.allocation_lifetimes.history_complete is True
    assert (
        sum(
            item.event_exact_birth_bytes
            for item in comparison.allocation_lifetimes.cohorts
        )
        >= 48 * MIB
    )
    assert (
        sum(
            item.event_exact_release_bytes
            for item in comparison.allocation_lifetimes.cohorts
        )
        >= 16 * MIB
    )
    comparison_paths = comparison.write(output_dir / "full-history-comparison")
    assert comparison_paths["events"].is_file()
    assert comparison_paths["cohorts"].is_file()

    timeline = run.timeline(attribution=options)
    assert len(timeline.point_comparisons) == 2
    assert all(
        item.events_available and item.events_complete
        for item in timeline.point_comparisons
    )
    assert timeline.allocation_lifetimes is not None
    assert timeline.allocation_lifetimes.history_available is True
    assert timeline.allocation_lifetimes.history_complete is True
    assert (
        sum(
            item.event_exact_birth_bytes
            for item in timeline.allocation_lifetimes.cohorts
        )
        >= 48 * MIB
    )
    assert (
        sum(
            item.event_exact_release_bytes
            for item in timeline.allocation_lifetimes.cohorts
        )
        >= 48 * MIB
    )
    timeline_paths = timeline.write(
        output_dir / "full-history-timeline",
        include_unchanged=False,
    )
    assert timeline_paths["events"].is_file()
    assert timeline_paths["cohorts"].is_file()

    print()
    print("=== Full allocator history: comparison attribution ===")
    print(comparison.to_text(include_unchanged=False))
    print()
    print("=== Full allocator history: timeline attribution ===")
    print(timeline.to_text(include_unchanged=False))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    output_dir = parse_args().output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    try:
        _snapshot_only(output_dir)
        _full_history(output_dir)
        print()
        print(f"bundles and reports: {output_dir}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
