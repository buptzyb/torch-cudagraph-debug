"""Trace allocation cohorts from an anchor point through later cleanup."""

from __future__ import annotations

import gc

import torch

from torch_cudagraph_debug.memory_debug import (
    AttributionOptions,
    MemoryRecorder,
)


def _allocate_warmup_state() -> torch.Tensor:
    return torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def _allocate_persistent_state() -> torch.Tensor:
    return torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def _allocate_transient_state() -> torch.Tensor:
    return torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires a CUDA device")

    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=100_000,
        clear_history=True,
    )
    try:
        warmup_state = _allocate_warmup_state()
        persistent_state = _allocate_persistent_state()

        recorder = MemoryRecorder(name="allocation-lifetimes")
        recorder.mark("anchor")
        recorder.mark("before_transient")

        transient_state = _allocate_transient_state()
        del transient_state
        gc.collect()
        torch.cuda.synchronize()
        recorder.mark("after_transient")

        del warmup_state
        gc.collect()
        torch.cuda.synchronize()
        recorder.mark("after_cleanup")
        run = recorder.finish()

        report = run.lifetimes(
            "anchor",
            through="after_cleanup",
            attribution=AttributionOptions(
                events=True,
                on_missing="error",
                stack_depth=4,
                limit=20,
            ),
        )
        print(report.to_text())
        report.write("reports/allocation-lifetimes")

        born = run.lifetimes(
            born_between=("before_transient", "after_transient"),
            through="after_cleanup",
            attribution=AttributionOptions(
                events=True,
                on_missing="error",
                stack_depth=4,
                limit=20,
            ),
        )
        print()
        print(born.to_text())
        born.write("reports/allocation-births")

        # Keep this allocation live through the final memory point.
        assert persistent_state.numel() == 32 * 1024 * 1024
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
