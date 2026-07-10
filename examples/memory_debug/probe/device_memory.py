"""Compare CUDA device usage with the allocator of the recording process.

Run with:
  python examples/memory_debug/probe/device_memory.py

Each snapshot pairs allocator state with a device-wide
torch.cuda.mem_get_info() reading. The report separates allocator reserved
bytes from device usage not attributed to the allocator of the recording process.
That unattributed value can include raw CUDA allocations and context state
from the recording process, plus allocations from other processes on a
shared GPU.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from torch_cudagraph_debug.memory_debug import MemoryProbe

MIB = 1024 * 1024
BUFFER_BYTES = 128 * MIB


def _load_cudart() -> ctypes.CDLL:
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    names = [str(path) for path in sorted(torch_lib.glob("libcudart.so*"))]
    names += ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    for name in names:
        try:
            cudart = ctypes.CDLL(name)
            cudart.cudaMalloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
            ]
            cudart.cudaMalloc.restype = ctypes.c_int
            cudart.cudaFree.argtypes = [ctypes.c_void_p]
            cudart.cudaFree.restype = ctypes.c_int
            return cudart
        except OSError:
            continue
    raise RuntimeError("could not load the CUDA runtime library")


def _device_row(comparison, device_index: int):
    return next(
        item
        for item in comparison.device_comparisons
        if item.device_index == device_index
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    torch.cuda.init()
    # Warm up lazy CUDA context work so the examples focus on the
    # allocations introduced below.
    warmup = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    del warmup
    torch.cuda.empty_cache()
    device_index = torch.cuda.current_device()
    probe = MemoryProbe(name="device-memory", devices=device_index)

    baseline = probe.snapshot()
    sample = baseline.device_memory[device_index]
    assert sample.free_bytes + sample.used_bytes == sample.total_bytes

    # A tensor goes through the caching allocator, so allocator reserved
    # accounts for the allocator's known contribution.
    tensor = torch.empty(BUFFER_BYTES, dtype=torch.uint8, device="cuda")
    tensor.fill_(1)
    after_tensor = probe.snapshot()
    tensor_change = probe.compare(baseline, after_tensor)
    tensor_row = _device_row(tensor_change, device_index)
    assert tensor_row.delta_allocator_reserved_bytes >= BUFFER_BYTES
    assert tensor_row.delta_unattributed_device_bytes is not None
    assert tensor_row.delta_total_bytes is not None

    # A raw cudaMalloc bypasses the caching allocator, standing in for the
    # memory a communication or math library allocates on its own: allocator
    # reserved does not move, so its device-wide growth is unattributed to
    # the recording process allocator.
    cudart = _load_cudart()
    raw_pointer = ctypes.c_void_p()
    status = cudart.cudaMalloc(ctypes.byref(raw_pointer), ctypes.c_size_t(BUFFER_BYTES))
    if status != 0:
        raise RuntimeError(f"cudaMalloc failed with status {status}")
    try:
        after_raw = probe.snapshot()
        raw_change = probe.compare(after_tensor, after_raw)
        raw_row = _device_row(raw_change, device_index)
        assert raw_row.delta_allocator_reserved_bytes == 0
        assert raw_row.delta_unattributed_device_bytes is not None
        assert tensor.numel() == BUFFER_BYTES

        print("=== Tensor allocation: the allocator explains the growth ===")
        print(tensor_change.to_text(include_unchanged=False))
        print()
        print("=== Raw cudaMalloc: growth unattributed to the allocator ===")
        print(raw_change.to_text(include_unchanged=False))
    finally:
        free_status = cudart.cudaFree(raw_pointer)
        if free_status != 0:
            raise RuntimeError(f"cudaFree failed with status {free_status}")


if __name__ == "__main__":
    main()
