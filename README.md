# torch-cudagraph-debug

`torch-cudagraph-debug` is a PyTorch extension for inspecting tensor values from
CUDA Graph replay. A tensor probe returns its input tensor unchanged while
inserting graph-captured debug work:

1. device-to-host copy into per-invocation pinned host staging memory;
2. optional CUDA host callback;
3. native print or compare logic on CPU memory.

The first domain is `tensor_debug`:

- `TensorPrint`: print a compact tensor summary and sample values.
- `TensorRecord`: keep the latest CPU snapshot for each logical probe slot without a host callback.
- `TensorCompare`: compare replay values against CPU or NumPy ground truth.

## Status

This is a v0.1 source-built package for Linux CUDA environments. It targets
PyTorch 2.6+ and builds against the CUDA-enabled PyTorch already installed in the
runtime where you install it. Prebuilt wheels are intentionally out of scope for
the first release.

## Install

Install CUDA-enabled PyTorch first, then install the v0.1.0 release tag from
GitHub with build isolation disabled so the extension builds against that exact
PyTorch:

```bash
pip install --no-build-isolation \
  "git+https://github.com/buptzyb/torch-cudagraph-debug.git@v0.1.0"
```

To test the latest development branch instead, install `main`:

```bash
pip install --no-build-isolation \
  "git+https://github.com/buptzyb/torch-cudagraph-debug.git@main"
```

From a source checkout:

```bash
cd torch-cudagraph-debug
pip install --no-build-isolation .
```

Enabled probes require a native extension built against CUDA-enabled PyTorch.
Source-tree imports and all-disabled probes can run without the extension, but
normal source installation for runtime use should happen in the target CUDA
environment.

## Documentation

- [API reference](docs/api.md): signatures, method semantics, execution modes,
  multi-action behavior, examples, and troubleshooting.
- [Release checklist](docs/release_checklist.md): validation steps for source
  releases and GPU smoke tests.
- [Changelog](CHANGELOG.md): release notes and compatibility notes.

## Basic Usage

```python
import torch

from torch_cudagraph_debug.tensor_debug import (
    CudaGraphTensorProbe,
    TensorCompare,
    TensorPrint,
    TensorRecord,
)

static_x = torch.ones(4, device="cuda")
expected = torch.full((4,), 3.0, device="cpu")
probe = CudaGraphTensorProbe(
    "mid",
    actions=[
        TensorPrint(max_items=8),
        TensorRecord(),
        TensorCompare([expected], rtol=1e-5, atol=1e-8),
    ],
)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    mid = static_x + 2
    mid = probe(mid)

g.replay()
torch.cuda.synchronize()

snapshots = probe.records()
probe.assert_ok()
probe.close()
```

`CudaGraphTensorProbe.__call__` always returns the original tensor object. By
default, eager and warmup calls are transparent no-ops; debug side effects are
installed only while the current CUDA stream is being captured.

## Debug Modes

`TensorPrint` is useful when you need a quick value check from graph replay:

```python
probe = CudaGraphTensorProbe("activation", [TensorPrint(max_items=16, every=10)])
```

`TensorRecord` keeps the latest CPU snapshot for each logical probe slot:

```python
probe = CudaGraphTensorProbe("activation", [TensorRecord()])
snapshots = probe.records()
latest_cpu_tensor = snapshots[0].tensor
```

All actions use retained pinned host memory per logical slot. This path exposes
those slots as latest records and does not capture a host callback when
`TensorRecord` is the only action. Snapshot `replay_index` is `0` because no
callback runs to count graph replays. A probe is a single-capture object: all
active calls must happen in the same capture session, and repeated calls in that
capture get logical slots in capture-call order. Slots may have different
shapes, dtypes, and sizes. Create a new probe for another graph capture. Read
records after `graph.replay()` and `torch.cuda.synchronize()` to get meaningful
values. CUDA graph capture records the D2H node but does not execute it, so
fresh record slots read before first replay contain zero-initialized bytes
rather than graph tensor values.

For a replay-indexed time series, collect `TensorRecord` snapshots in the Python
replay loop:

```python
snapshots_by_replay = []
for replay_index in range(1, 5):
    graph.replay()
    torch.cuda.synchronize()
    for snapshot in probe.records():
        snapshots_by_replay.append(
            (replay_index, snapshot.invocation_index, snapshot.tensor.clone())
        )
```

Each `TensorSnapshot` contains `probe_name`, `replay_index`,
`invocation_index`, `shape`, `dtype`, `device`, and the recorded CPU `tensor`.

`TensorCompare` checks replay values against a list of CPU tensors or NumPy
arrays, one per `invocation_index`. A mismatch sets a sticky failure flag; call
`probe.assert_ok()` after replay and synchronization.

```python
probe = CudaGraphTensorProbe(
    "activation",
    [TensorCompare([expected_cpu_tensor], rtol=1e-4, atol=1e-5)],
)
```

If one probe instance is called multiple times in a captured graph and each
logical slot has a different expected value, pass the expected tensors in
`invocation_index` order:

```python
expected_by_slot = [expected0, expected1, expected2]
probe = CudaGraphTensorProbe("layer.hidden", [TensorCompare(expected_by_slot)])
```

Every action accepts `enabled=False` for config-driven toggles. Disabled actions
are not installed into the native probe. If every action is disabled, the probe
is a pure no-op: it returns the input tensor unchanged, exposes no records, and
does not require the native extension to be loaded.

## Gradient Probes

Use `probe.attach_grad(tensor)` to register an autograd hook that probes a
tensor's backward gradient. It returns the original tensor, so activation probes
can stay inline:

```python
hidden = activation_value_probe(hidden)
hidden = activation_grad_probe.attach_grad(hidden)
```

For parameter gradients, prefer side-effect style so the code does not look like
it replaces module state:

```python
weight_grad_probe.attach_grad(module.weight)
```

This parameter hook observes the gradient when autograd produces it. If you want
the final `.grad` buffer after `backward()`, probe that buffer explicitly:

```python
loss.backward()
if module.weight.grad is not None:
    final_weight_grad_probe(module.weight.grad)
```

If you need to remove a long-lived hook, request the PyTorch hook handle:

```python
_, handle = weight_grad_probe.attach_grad(module.weight, return_handle=True)
handle.remove()
```

For a module-level example that inserts a probe into an internal hidden tensor
and controls actions from a small config object, see
[`examples/transformer_block_probe.py`](examples/transformer_block_probe.py).
For the common forward/gradient probe patterns, see
[`examples/grad_probe_patterns.py`](examples/grad_probe_patterns.py).
For repeated calls of one probe in a single graph, see
[`examples/multiple_invocations_record_compare.py`](examples/multiple_invocations_record_compare.py).
For TensorBoard summaries from recorded snapshots, see
[`examples/tensorboard_export_records.py`](examples/tensorboard_export_records.py).

## Custom Python Processing

Custom Python actions are intentionally not executed inside CUDA host callbacks.
Use `TensorRecord` to bring graph replay values back to CPU, then consume or
clone them after synchronization:

```python
probe = CudaGraphTensorProbe("hidden", [TensorRecord()])

graph.replay()
torch.cuda.synchronize()
for snapshot in probe.records():
    my_custom_action(snapshot.tensor)
```

If you need a replay-by-replay time series, clone the CPU tensors before the
next replay overwrites the latest-record slots.

## Execution Modes

The default execution mode is `mode="capture"`:

```python
probe = CudaGraphTensorProbe("activation", [TensorRecord()])
```

In this mode, `probe(tensor)` is completely transparent during eager warmup: it
does not validate the input tensor, allocate staging memory, enqueue D2H copies,
launch callbacks, print, record, or compare. When the same call
runs inside `torch.cuda.graph(...)`, the probe captures the debug D2H copy and
captures a host callback only when an action needs one (`TensorPrint` or
`TensorCompare`).

`mode="always"` is an explicit escape hatch:

```python
probe = CudaGraphTensorProbe(
    "activation",
    [TensorRecord()],
    mode="always",
)
```

Keep this mode for cases where eager side effects are intentional: testing the
probe without writing a CUDA graph, recording eager and graph values with the
same probe machinery, debugging non-graph CUDA stream code, or covering native
enqueue behavior in this package's tests. It is not the default because it can
pollute warmup records, print during warmup, or set compare failures before graph
replay. Eager calls in this mode do not claim CUDA graph capture ownership; the
first graph capture that uses the probe still owns the probe.

## Single-Capture Probe Topology

A single `CudaGraphTensorProbe` instance may be called multiple times in one
captured graph, for example inside a repeated layer stack. These calls define
logical slots with `invocation_index` values in capture order: `0`, `1`, `2`,
and so on. A probe called three times in one graph and replayed twice records:

```text
replay_index:     1  1  1  2  2  2
invocation_index: 0  1  2  0  1  2
```

A probe is bound to the first CUDA graph capture session that uses it. Reusing
the same probe in another graph capture, including recapturing the same Python
code, is an error. Create a new probe for each graph capture. Slots within that
one capture may have different tensor metadata. `TensorCompare` takes a list of
expected tensors and uses `expected[invocation_index]` for each slot. Callback
`replay_index` values are counted independently per slot. A missing expected
item for an observed invocation is an error; extra expected items are ignored.

Each logical slot owns its own pinned host staging buffer. This keeps
callback-backed actions from reading data overwritten by another invocation,
including invocations captured on different streams in the same graph. Pinned
host memory is roughly the sum of the largest tensor byte size observed for
each invocation slot:

```text
sum(slot_nbytes for slot in captured_probe_invocations)
```

## Non-Contiguous Inputs

The default non-contiguous policy is fail-fast when debug work is actually
installed:

```python
probe = CudaGraphTensorProbe("view", [TensorRecord()])
with torch.cuda.graph(g):
    probe(non_contiguous_tensor)  # raises
```

In the default `mode="capture"`, eager warmup calls are no-ops, so this policy is
checked during capture. If you explicitly allow it, the probe creates an
internal contiguous CUDA copy only for the debug path, captures the copy before
the D2H node, keeps that internal storage alive with the captured probe context,
and still returns the original tensor:

```python
probe = CudaGraphTensorProbe(
    "view",
    [TensorRecord()],
    non_contiguous="copy",
)
```

The copy policy costs roughly `tensor.numel() * tensor.element_size()` additional
CUDA graph-pool memory for each captured non-contiguous probe site. In tight
memory captures, prefer probing a smaller slice or making the debug copy explicit
in your model code so the memory cost is visible.

## Performance Caveat

Probe nodes are part of the captured graph dependency chain. Even though
`probe(tensor)` returns the original tensor, the D2H copy and CUDA host callback
are enqueued on the captured stream, so later dependent graph work waits for
that callback to finish. This can create large GPU bubbles and make performance
traces look much worse than the uninstrumented model. Per-invocation staging
also means large tensors probed at many call sites can retain substantial pinned
host memory. Treat this package as a correctness debugging aid, not as a
low-overhead profiling tool; remove or gate probes before measuring performance.

## Lifecycle Rules

- Keep the probe alive for at least as long as any captured `torch.cuda.CUDAGraph`
  that contains it can replay.
- Call `probe.close()` only after those graphs will never replay again.
- Call `torch.cuda.synchronize()` before reading records or asserting compare
  status if replay was launched asynchronously.
- Read `TensorRecord` values only after at least one graph replay; capture alone
  does not execute the captured D2H copy.
- Host callbacks do not call Python or CUDA APIs. Print and compare are native
  CPU operations on pinned staging memory.

## Limitations

- Linux/CUDA only.
- One probe is not designed for concurrent replay on multiple streams.
- One probe can be captured by only one CUDA graph capture session; create new
  probes for recapture or for separate graph captures.
- Inputs must use supported dense dtypes: float64, float32, float16, bfloat16,
  int64, int32, int16, int8, uint8, or bool.
- Compare expected values must be a non-empty sequence of CPU tensors or NumPy
  arrays, with each item matching the corresponding invocation's shape and dtype.

## Tested Environments

The `v0.1.0` release gate validates the public GitHub install path in
`nvcr.io/nvidia/pytorch:26.03-py3` on Computelab GPU nodes. The gate builds the
native extension from source, checks `native_extension_available True`, runs the
full CUDA pytest suite, and covers Python-side replay snapshot collection,
gradient probes, TensorBoard export, single-capture rejection, and
repeated-invocation examples.

The package is intended to build against CUDA-enabled PyTorch 2.6+
installations, but each CUDA/PyTorch/container combination should be verified in
the target environment before relying on it in a larger training workflow.

## Development

Local metadata and Python checks:

```bash
python -m py_compile $(find src tests examples -name '*.py')
python -m build --sdist --no-isolation
```

GPU validation requires a CUDA-enabled PyTorch environment:

```bash
pip install --no-build-isolation --no-deps .
rm -rf /tmp/tcgd-tests
cp -r tests /tmp/tcgd-tests
cd /tmp
pytest -q /tmp/tcgd-tests
```

Run the GPU test suite from outside the source checkout after installation.
Running `pytest` directly in the source tree can import the unbuilt `src/`
package and skip native CUDA tests.
