"""Show which analyses need allocator history and how missing history fails.

State-only comparisons always work. Event and lifetime attribution require
complete ``torch.cuda.memory._record_memory_history()`` evidence. Stack
attribution accepts partial frame coverage, but zero coverage for nonempty active
state fails with a typed error.

Run with:
  python examples/memory_debug/recorder/history_requirements.py --output-dir /tmp/tcgd-history
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryDisplayOptions,
    MemoryHistoryDisabledError,
    MemoryLifetimeSelection,
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


def _record_without_history(output_dir: Path) -> None:
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
    torch.cuda.synchronize()
    recorder.record_point("after_free")
    recorder.finish()

    run = MemoryRun.load(bundle_dir, cache_snapshots=False)

    # State-only analysis never needs allocator history.
    state = run.compare("before", "after_alloc")
    assert _all_total(state).delta.allocated_bytes >= 16 * MIB
    state_paths = state.write(output_dir / "state-only-comparison")
    assert state_paths["json"].is_file()

    # Zero stack coverage and unavailable event history fail explicitly.
    try:
        run.compare(
            "before",
            "after_alloc",
            attribution=MemoryAttributionOptions(
                stacks=True,
                events=True,
                display=MemoryDisplayOptions(stack_depth=4),
            ),
        )
    except MemoryHistoryDisabledError as error:
        print(f"expected attribution error: {error}")
    else:
        raise AssertionError("missing history did not raise for attribution")

    try:
        run.lifetimes(
            MemoryLifetimeSelection.born_between("before", "after_alloc"),
            through="after_free",
        )
    except MemoryHistoryDisabledError as error:
        print(f"expected lifetime error: {error}")
    else:
        raise AssertionError("missing history did not raise for lifetimes")

    print()
    print("=== State-only comparison (no history required) ===")
    print(state.to_text(include_unchanged=False))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    output_dir = parse_args().output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    try:
        _record_without_history(output_dir)
        print()
        print(f"bundles and reports: {output_dir}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
