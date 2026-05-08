# torch-cudagraph-debug API Reference

This document describes the public `tensor_debug` API. The package is designed
for probes inserted into code that is later captured by `torch.cuda.CUDAGraph`.

## Import Path

```python
from torch_cudagraph_debug.tensor_debug import (
    CudaGraphTensorProbe,
    TensorPrint,
    TensorRecord,
    TensorCompare,
    TensorSnapshot,
    TensorCompareMismatchError,
)
from torch_cudagraph_debug.tensor_debug.postprocess import export_records_to_tensorboard
```

## CudaGraphTensorProbe

```python
CudaGraphTensorProbe(
    name: str,
    actions: Sequence[TensorPrint | TensorRecord | TensorCompare],
    *,
    non_contiguous: Literal["error", "copy"] = "error",
    mode: Literal["capture", "always"] = "capture",
)
```

`CudaGraphTensorProbe` is a transparent tensor operator: `probe(tensor)` returns
the same tensor object it received. When debug work is active, the native
extension enqueues a D2H copy into per-invocation pinned host staging memory for
non-empty tensors. A host callback is captured only when an action needs one:
`TensorPrint` or `TensorCompare`. `TensorRecord` alone is callback-free.

Arguments:

- `name`: non-empty probe name used in print output, records, and compare errors.
- `actions`: one or more action objects. Disabled actions are filtered before
  native probe creation.
- `non_contiguous`: probe-wide handling for non-contiguous inputs.
- `mode`: controls whether eager calls execute debug work.

### Execution Modes

`mode="capture"` is the default and should be used for model code that runs both
warmup and CUDA Graph capture:

```python
probe = CudaGraphTensorProbe("mid", [TensorRecord()])
y = probe(y)
```

In this mode, eager and warmup calls are no-ops. They do not validate the input
tensor, allocate pinned memory, enqueue copies, launch callbacks, print, record,
or compare. If the same call runs while the current CUDA stream is
being captured, the probe installs debug graph nodes.

`mode="always"` makes eager calls execute debug work too:

```python
probe = CudaGraphTensorProbe("mid", [TensorRecord()], mode="always")
```

Use `always` only when eager side effects are intentional:

- testing probe behavior without writing a CUDA graph;
- recording eager and graph values through the same API;
- debugging non-graph CUDA stream code;
- covering native enqueue behavior in tests.

It is not the default because eager execution can print during warmup, populate
record buffers before graph replay, or set sticky compare failures
before the graph being debugged has run.

### Non-Contiguous Inputs

`non_contiguous="error"` is the default. When debug work is active, a
non-contiguous input raises with guidance to opt into copy mode.

`non_contiguous="copy"` inserts an internal debug-only contiguous CUDA copy
before the D2H copy and still returns the original tensor. The internal tensor is
kept alive by the captured probe context so graph replay never uses a
reused graph-pool pointer. This costs roughly:

```text
tensor.numel() * tensor.element_size()
```

extra CUDA graph-pool memory for each captured non-contiguous probe site.

### Shape and Invocation Contract

A `CudaGraphTensorProbe` is a single-capture object. All active calls to the
same probe must happen in the same CUDA graph capture session. Different
invocations inside that capture may observe different tensor shapes, dtypes, and
sizes.

If the same probe instance is called multiple times in one graph capture, the
native code assigns logical slot `invocation_index` values in capture order.
The probe is bound to the first CUDA graph capture session that uses it. Reusing
the same probe in another graph capture, including recapturing the same Python
code, is an error. Create a new probe for each graph capture.

Each logical invocation slot owns a pinned host staging buffer sized for the
largest tensor observed at that slot. This keeps callback-backed actions from
reading data overwritten by another invocation, including when the same graph
capture uses side streams. Pinned host memory is roughly:

```text
sum(max_nbytes_per_invocation_slot)
```

For callback-backed actions, `replay_index` is counted independently for each
logical slot. In the common case where one graph contains every slot and is
replayed as a unit, callback replay counts look like graph replay numbers:

```text
replay_index:     1  1  1  2  2  2
invocation_index: 0  1  2  0  1  2
```

`TensorCompare` uses `expected[invocation_index]` for each logical slot. For
custom comparison logic or a replay-by-replay time series, record snapshots with
`TensorRecord` and clone them in Python after each replay.

Keep the probe alive until the graph that captured it is done replaying.

### Methods

```python
probe(tensor: torch.Tensor) -> torch.Tensor
```

Returns `tensor` unchanged. In `mode="capture"`, eager calls are no-ops and
capture calls install debug graph nodes. In `mode="always"`, eager calls enqueue
debug work on the current CUDA stream.

```python
probe.attach_grad(
    tensor: torch.Tensor,
    *,
    strict: bool = False,
    return_handle: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, RemovableHandle | None]
```

Registers a PyTorch autograd hook that probes `tensor`'s backward gradient and
returns `tensor` unchanged. The hook calls `probe(grad)` for its side effect and
then explicitly returns the original `grad`, so it is not a gradient transform.

If `tensor.requires_grad` is false, the default behavior is a no-op. Pass
`strict=True` to raise instead. With `return_handle=True`, the method returns
`(tensor, handle)` for tensors that require grad, or `(tensor, None)` for no-op
attachments.

```python
probe.records() -> list[TensorSnapshot]
```

Returns the latest CPU snapshot for each logical probe slot, ordered by
`invocation_index`. Call `torch.cuda.synchronize()` first if graph replay or
eager stream work may still be pending. `TensorRecord` snapshots have
`replay_index == 0` because the callback-free path does not count graph replays.
Record slots are allocated when debug work is active and are zero-initialized
when first allocated. CUDA graph capture records the D2H node but does not run
it, so reading records after capture and before the first replay returns
zero-initialized or previous slot contents rather than graph tensor values.
Call `graph.replay()` and `torch.cuda.synchronize()` before interpreting
records.

```python
probe.clear_records() -> None
```

Zeroes retained latest-record host buffers without dropping the captured slots.
The next graph replay writes fresh values into the same pinned buffers.

```python
probe.status() -> dict[str, object]
```

Returns compare state:

```python
{"ok": bool, "message": str, "replay_index": int, "invocation_index": int}
```

Compare failures are sticky until the probe is closed.

```python
probe.assert_ok() -> None
```

Raises `TensorCompareMismatchError` if any compare action has failed.

```python
probe.close() -> None
```

Releases native resources. Only call this after every CUDA graph that captured
the probe can never replay again.

## Actions

### TensorPrint

```python
TensorPrint(
    max_items: int = 16,
    every: int = 1,
    summary: bool = True,
    enabled: bool = True,
)
```

Prints a compact tensor summary from the native host callback.

- `max_items`: maximum number of values to include.
- `every`: print only when `replay_index % every == 0`.
- `summary`: include summary statistics when supported by the dtype.
- `enabled`: disable this action without changing surrounding config.

Example:

```python
probe = CudaGraphTensorProbe(
    "activation",
    [TensorPrint(max_items=8, every=10)],
)
```

### TensorRecord

```python
TensorRecord(
    enabled: bool = True,
)
```

Stores the latest CPU snapshot for each logical probe slot.

- `enabled`: disable this action without changing surrounding config.

Example:

```python
probe = CudaGraphTensorProbe("activation", [TensorRecord()])

g.replay()
torch.cuda.synchronize()
latest = probe.records()[0].tensor
```

`TensorRecord` exposes the retained per-invocation pinned host slots as latest
CPU snapshots. If it is the only action, no host callback is captured and
`replay_index` is reported as `0`.

### TensorCompare

```python
TensorCompare(
    expected: Sequence[torch.Tensor | numpy.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    enabled: bool = True,
)
```

Compares replay values against CPU or NumPy ground truth from the host callback.

- `expected`: non-empty sequence of CPU tensors or NumPy arrays. `expected[i]`
  is used for `invocation_index == i`. Wrap a single expected tensor as
  `TensorCompare([expected])`. A missing item for an observed invocation is an
  error; extra items are ignored.
- `rtol`, `atol`, `equal_nan`: tolerance options.
- `enabled`: disable this action without validating `expected`.

Example:

```python
expected = torch.full((4,), 3.0, device="cpu")
probe = CudaGraphTensorProbe(
    "mid",
    [TensorCompare([expected], rtol=1e-5, atol=1e-8)],
)
```

After replay:

```python
torch.cuda.synchronize()
probe.assert_ok()
```

## TensorSnapshot

`probe.records()` returns `TensorSnapshot` objects with:

- `probe_name`: probe name;
- `replay_index`: callback-observed replay count for this logical slot, or `0`
  for callback-free `TensorRecord` snapshots;
- `invocation_index`: logical slot index for this probe;
- `shape`: recorded tensor shape;
- `dtype`: recorded tensor dtype;
- `device`: original CUDA device string;
- `tensor`: CPU tensor containing the recorded bytes.

## Python-Side Time Series

Custom Python actions are intentionally not supported inside CUDA host
callbacks. Use `TensorRecord`, synchronize after each replay, and clone the CPU
snapshots before the next replay overwrites the latest-record slots:

```python
probe = CudaGraphTensorProbe("hidden", [TensorRecord()])
snapshots_by_replay: list[tuple[int, int, torch.Tensor]] = []

for replay_index in range(1, 5):
    graph.replay()
    torch.cuda.synchronize()
    for snapshot in probe.records():
        snapshots_by_replay.append(
            (replay_index, snapshot.invocation_index, snapshot.tensor.clone())
        )
```

## TensorBoard Export Helper

```python
from torch_cudagraph_debug.tensor_debug.postprocess import export_records_to_tensorboard

export_records_to_tensorboard(
    writer,
    records,
    *,
    tag_prefix: str = "",
    step: int | Callable[[TensorSnapshot], int] | None = None,
    write_scalars: bool = True,
    write_histograms: bool = False,
) -> None
```

Exports `TensorSnapshot` records to a TensorBoard-compatible writer. The helper
does not import TensorBoard and does not add a package dependency; pass an
existing `SummaryWriter` or compatible object.

Default scalar tags are:

- `numel`
- `mean`
- `std`
- `min`
- `max`
- `l2_norm`

The default TensorBoard step is `snapshot.replay_index`. Pass an integer `step`
to force one global step for all records, or pass a callable to map each
snapshot to a step. Empty tensors emit only `numel=0`. Histograms are disabled
by default because they can be expensive for large tensors.

The helper intentionally does not clear records, flush or close the writer, save
raw tensors, or catch writer errors. Raw large tensors should be saved
separately, for example with `torch.save(snapshot.tensor, path)`.

## Multi-Action and Performance Behavior

One active probe invocation captures one D2H copy for non-empty tensors. It
captures one host callback only if any action needs callback-side work.
`TensorRecord` alone is callback-free; zero-element tensors still capture the
callback if one is needed but skip the D2H copy. Multiple callback actions on
the same invocation do not create multiple callbacks:

```text
CUDA tensor
  -> D2H copy
  -> optional host callback
       -> callback action 0
       -> callback action 1
       -> callback action 2
```

The callback runs callback actions in the order passed to `CudaGraphTensorProbe`.
Disabled actions are filtered in Python before native probe creation. If all
actions are disabled, the probe is a pure no-op: it does not require a native
probe handle, `probe(tensor)` returns `tensor`, `records()` returns `[]`, and
`status()` is ok.

Staging memory is private per logical invocation slot. `TensorRecord` exposes
those slots through `records()`. Callback-only probes use the same slot storage
internally but keep `records()` empty unless a `TensorRecord` action is present.

The D2H copy and any host callback are graph dependencies. Returning the
original tensor avoids changing Python dataflow, but it does not make the debug
nodes independent of the captured CUDA stream. Later graph work waits for these
debug nodes to finish, so traces with probes can show large GPU bubbles,
especially when callback actions are enabled. Large tensors probed at many
invocation sites can also retain substantial pinned host memory. Use probes for
correctness debugging, and remove or gate them before performance measurement.

## Examples

### Print Only

```python
probe = CudaGraphTensorProbe("mlp.out", [TensorPrint(max_items=16)])

with torch.cuda.graph(g):
    y = probe(y)
```

### Record and Compare

```python
expected = torch.full((4,), 3.0, device="cpu")
probe = CudaGraphTensorProbe(
    "mid",
    [
        TensorRecord(),
        TensorCompare([expected], rtol=1e-5, atol=1e-8),
    ],
)

with torch.cuda.graph(g):
    mid = probe(x + 2)

g.replay()
torch.cuda.synchronize()

snapshots = probe.records()
probe.assert_ok()
```

### Repeated Probe Calls

```python
expected = [
    torch.full((4,), 1.0, device="cpu"),
    torch.full((4,), 2.0, device="cpu"),
    torch.full((4,), 3.0, device="cpu"),
]
probe = CudaGraphTensorProbe(
    "layer.hidden",
    [TensorRecord()],
)

with torch.cuda.graph(g):
    h0 = probe(x + 1)
    h1 = probe(x + 2)
    h2 = probe(x + 3)

snapshots_by_replay = []
for replay_index in range(1, 3):
    g.replay()
    torch.cuda.synchronize()
    for snapshot in probe.records():
        snapshots_by_replay.append(
            (replay_index, snapshot.invocation_index, snapshot.tensor.clone())
        )

for snapshot in probe.records():
    print(snapshot.replay_index, snapshot.invocation_index, snapshot.tensor)
    torch.testing.assert_close(snapshot.tensor, expected[snapshot.invocation_index])
```

For a complete runnable version, see
[`examples/multiple_invocations_record_compare.py`](../examples/multiple_invocations_record_compare.py).

### Warmup-Transparent Default

```python
probe = CudaGraphTensorProbe("mid", [TensorRecord()])

# Eager warmup: no records, no compare, no prints.
for _ in range(3):
    y = probe(model_step(x))

with torch.cuda.graph(g):
    y = probe(model_step(static_x))
```

### Intentional Eager Debug

```python
probe = CudaGraphTensorProbe(
    "eager.mid",
    [TensorRecord()],
    mode="always",
)

y = probe(x)
torch.cuda.synchronize()
print(probe.records()[0].tensor)
```

### Non-Contiguous Copy

```python
view = x.t()
expected = view.detach().cpu().contiguous()
probe = CudaGraphTensorProbe(
    "view",
    [TensorRecord(), TensorCompare([expected], rtol=0.0, atol=0.0)],
    non_contiguous="copy",
)

with torch.cuda.graph(g):
    y = probe(view)
```

### Gradient Hooks

Probe an activation's forward value with `probe(tensor)` and its backward
gradient with `probe.attach_grad(tensor)`:

```python
hidden = value_probe(hidden)
hidden = activation_grad_probe.attach_grad(hidden)
```

For a parameter gradient hook, use side-effect style so the code does not look
like it replaces module state:

```python
weight_grad_probe.attach_grad(module.weight)
```

This hook observes the gradient when autograd produces it. That is not
necessarily the same observation point as the final optimizer-facing `.grad`
buffer. To inspect the final buffer, probe it after `backward()`:

```python
loss.backward()
if module.weight.grad is not None:
    final_weight_grad_probe(module.weight.grad)
```

Long-lived hooks can be removed with the optional PyTorch hook handle:

```python
_, handle = weight_grad_probe.attach_grad(module.weight, return_handle=True)
handle.remove()
```

For a complete example covering forward activation values, activation gradients,
parameter gradient hooks, and final `.grad` buffers, see
[`examples/grad_probe_patterns.py`](../examples/grad_probe_patterns.py).

### Internal Hidden Tensor in a Module

For a complete module-level example, see
[`examples/transformer_block_probe.py`](../examples/transformer_block_probe.py).
It shows a block-shaped `torch.nn.Module` that owns a probe as a module field,
inserts it into an internal hidden tensor in `forward()`, controls print/record
and compare actions from a config object, relies on warmup-transparent default
behavior, and reads latest records after graph replay.

### TensorBoard Summaries

For a complete TensorBoard example, see
[`examples/tensorboard_export_records.py`](../examples/tensorboard_export_records.py).
It shows both the convenience helper:

```python
export_records_to_tensorboard(
    writer,
    records,
    tag_prefix="helper/",
    write_histograms=True,
)
```

and direct `SummaryWriter` calls for custom metrics:

```python
for snapshot in records:
    tensor = snapshot.tensor.float()
    step = snapshot.replay_index
    writer.add_scalar("manual/abs_max", tensor.abs().max().item(), step)
```

TensorBoard writes should happen after `torch.cuda.synchronize()`, never inside
the CUDA host callback.

### Custom Python Consumers

For custom processing, read `TensorRecord` snapshots after synchronization:

```python
probe = CudaGraphTensorProbe("mid", [TensorRecord()])

with torch.cuda.graph(g):
    y = probe(y)

g.replay()
torch.cuda.synchronize()
for snapshot in probe.records():
    my_custom_action(snapshot.tensor)
```

Python callbacks are not run from CUDA host callbacks.

## Troubleshooting

### Creating a probe raises `NativeExtensionUnavailableError`

The native extension is CUDA-only and source-built against the installed PyTorch.
Install with build isolation disabled after installing CUDA-enabled PyTorch:

```bash
pip install --no-build-isolation .
```

An all-disabled probe does not load the native extension.

### `records()` is empty

Common causes:

- the probe has not been captured yet;
- replay is asynchronous and `torch.cuda.synchronize()` has not run;
- every record action is disabled;
- the call happened during eager warmup in default `mode="capture"`;
- `TensorRecord` was not included in the probe actions.

For a replay-indexed time series, clone `TensorRecord` snapshots after each
replay.

### `records()` returns zero values before replay

That is expected for newly allocated record slots. CUDA graph capture records
the D2H node but does not execute it. Fresh slots are zero-initialized to avoid
uninitialized host memory, and the first replay overwrites them with real graph
tensor values. Run `graph.replay()` and `torch.cuda.synchronize()` before
interpreting record tensors.

### `assert_ok()` does not fail during eager warmup

That is expected in default `mode="capture"`. Eager calls are transparent no-ops.
Use `mode="always"` only if eager compare side effects are intentional.

### Non-contiguous input does not fail during warmup

That is expected in default `mode="capture"`. The contiguity policy is checked
when debug work is active: during capture or in `mode="always"`.

### `TensorCompare` reports shape or dtype mismatch

Each expected CPU tensor or NumPy array must have the same shape and dtype as
the tensor observed at the corresponding `invocation_index`. `TensorCompare`
copies expected bytes into native storage when the probe is created.

### Probe reused in another capture

One probe can be captured by only one CUDA graph capture session. Multiple
active calls inside that capture are supported and become logical slots, but a
second capture with the same probe raises immediately. If you need to recapture
or inspect another graph, create a new probe.

### Profiling shows a large bubble before `wait for host callable`

That is expected for inline probes. The host callback is captured on the same
dependency chain as the D2H copy, so later graph work cannot continue until the
callback returns. Use the trace only to confirm probe behavior, not to evaluate
model performance.

### Values are stale or missing

CUDA graph replay and eager stream work are asynchronous. Synchronize before
reading records or asserting compare state:

```python
g.replay()
torch.cuda.synchronize()
probe.assert_ok()
```
