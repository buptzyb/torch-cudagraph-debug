# Tensor Debug Examples

Start with `quickstart.py`, then follow the order in the parent
[examples index](../README.md). These scripts use the compiled native extension
and require Linux, CUDA, and a CUDA-enabled PyTorch build.

The examples deliberately keep the following lifecycle visible:

1. Create a `TensorProbe` before graph capture.
2. Eager warmup is a transparent no-op with the default `when="capture"`.
3. Capture and replay the graph.
4. Synchronize before reading snapshots or status.
5. Keep the probe alive while the graph may replay, then call `close()`.

`record_and_compare.py` catches one intentional `TensorMismatchError` and still
exits successfully. `probe_modes.py` similarly catches the expected default
non-contiguous-input error. Those failures demonstrate user-facing diagnostics;
they are not test failures.

`non_contiguous="copy"` allocates a debug-only contiguous tensor in the graph
pool. Use it only when that memory cost is acceptable. `when="always"` enables
probe work on eager calls and is separate from the default capture-only mode.
