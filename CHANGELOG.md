# Changelog

## v0.1.0 - 2026-05-12

First public 0.1.0 release of `torch-cudagraph-debug`.

### Added

- `CudaGraphTensorProbe`, a CUDA graph tensor probe that returns its input tensor
  unchanged while capturing debug-side device-to-host copies and host callbacks.
- `TensorPrint` for compact native CPU-side printing from replay snapshots.
- `TensorRecord` for callback-free latest CPU snapshots per logical probe slot,
  exposed as `TensorSnapshot` objects in Python.
- Per-invocation pinned host staging for all actions, so callback-backed probes
  do not reuse one shared host buffer across logical slots.
- `TensorCompare` for replay-time comparison against per-invocation CPU tensor
  or NumPy ground truth lists, with sticky mismatch reporting via
  `probe.assert_ok()`.
- `invocation_index` on recorded snapshots and compare status so callers can
  distinguish repeated calls of the same probe within one replay.
- Single-capture probe ownership: one probe may be called multiple times inside
  one capture, slots may have different tensor metadata, and another graph
  capture must use a new probe.
- `CudaGraphTensorProbe.attach_grad()` for activation, output, and parameter
  gradient probing through PyTorch autograd hooks.
- `mode="capture"` as the default warmup-transparent mode and `mode="always"`
  as an explicit eager/debug escape hatch.
- `non_contiguous="copy"` for opt-in debug-only contiguous copies of
  non-contiguous inputs.
- `torch_cudagraph_debug.tensor_debug.postprocess.export_records_to_tensorboard()`
  for scalar and optional histogram summaries from recorded snapshots.
- Examples for basic tensor debugging, record/compare workflows, module-internal
  hidden tensor probes, gradient probe patterns, Python-side replay snapshot
  collection, and TensorBoard export.

### Compatibility Notes

- Linux CUDA environments only.
- Source builds only; prebuilt wheels are intentionally not provided for v0.1.
- Build against the CUDA-enabled PyTorch installation in the target runtime with
  `pip install --no-build-isolation`.
- Source-tree imports and all-disabled probes can run without the native
  extension, but enabled probes require a native extension built against
  CUDA-enabled PyTorch.
- Probe nodes are inline graph dependencies and can introduce large GPU bubbles;
  they are intended for correctness debugging, not performance measurement.
