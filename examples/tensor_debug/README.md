# Tensor Debug Examples

Read the [Tensor Debug guide](../../docs/tensor_debug.md) for probe semantics
and lifecycle details.

Start with `quickstart.py`, then follow the order in the parent
[examples index](../README.md). These scripts use the compiled native extension
and require Linux, CUDA, and a CUDA-enabled PyTorch build.

The examples deliberately keep the following lifecycle visible:

1. Create a `TensorProbe` before graph capture.
2. Eager warmup is a transparent no-op with the default `when="capture"`.
3. Capture and replay the graph.
4. Pass the replay stream to `snapshot()` or the first check-status query, then
   use `synchronize=False` for additional queries covered by that wait.
5. Observe that all invocation slots from one replay share one 1-based replay
   index inside one aggregate `TensorProbeSnapshot`.
6. Keep the probe alive while the graph may replay, then call `close()`.

`snapshot_comparison.py` shows the second quick workflow: collect one eager
snapshot and one CUDA Graph snapshot with independent probes, then compare them
directly without run metadata or bundles.

`record_and_check.py` catches one intentional `TensorCheckError` and still
exits successfully. `probe_modes.py` similarly catches the expected default
non-contiguous-input error. Those failures demonstrate user-facing diagnostics;
they are not test failures.

`non_contiguous="copy"` allocates a debug-only contiguous tensor in the graph
pool. Use it only when that memory cost is acceptable. `when="always"` enables
probe work on eager calls and is separate from the default capture-only mode.

## Complete Workflows

The quick-workflow examples above teach one probe and one graph. Use
[`eager_vs_cuda_graph.py`](eager_vs_cuda_graph.py) when values must be compared
across independent executions. It records the same named observations in eager
and CUDA Graph modes, loads both bundles, and writes text, JSON, CSV, and HTML
reports.

[`replay_series.py`](replay_series.py) records two graph replays against one
eager reference. Summary-only input and hidden observations demonstrate
`inconclusive` allclose results, while the full output payload provides a
conclusive mismatch and first-divergence evidence.

Both workflows require fresh output directories because bundle writers never
overwrite an existing nonempty bundle.
