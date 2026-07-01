# Tensor Debug Guide

`tensor_debug` inserts native print, record, and comparison probes into a
PyTorch CUDA Graph. This guide explains probe lifecycle and behavior. For exact
signatures, see the [API reference](api.md#tensor-debug); for runnable programs,
follow the [Tensor Debug examples](../examples/tensor_debug/README.md).

## Core Usage

Import tensor APIs from their domain module:

```python
from torch_cudagraph_debug.tensor_debug import (
    CompareTensor,
    PrintTensor,
    RecordTensor,
    TensorProbe,
)
```

For most debugging sessions, start with `RecordTensor`: it avoids a CUDA host
callback when used alone and keeps inspection in Python after synchronization.
It still enqueues a device-to-host copy into pinned staging memory for every
captured invocation, so keep probes limited to tensors needed for debugging.
Use `PrintTensor` for immediate console output and `CompareTensor` when expected
values are already available. Both add CUDA host-callback overhead and can
create a GPU bubble, so use them selectively.

Active actions support contiguous CUDA tensors with `float16`, `bfloat16`,
`float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, or `bool`
dtype. Other dtypes, including complex tensors, are rejected.

A probe returns the exact input tensor object while adding debug work during
capture:

```python
import torch

static_x = torch.ones(4, device="cuda")
expected = torch.full((4,), 3.0, device="cpu")
probe = TensorProbe(
    "mid",
    [
        PrintTensor(max_items=8),
        RecordTensor(),
        CompareTensor(expected),
    ],
)

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    mid = probe(static_x + 2)

replay_stream = torch.cuda.current_stream()
graph.replay()

for snapshot in probe.snapshots(synchronize=replay_stream):
    print(snapshot.replay_index, snapshot.invocation_index, snapshot.tensor)

# The snapshot query already waited for every node in this replay.
probe.assert_ok(synchronize=False)
probe.close()
```

The default `when="capture"` makes eager warmup calls transparent no-ops.
Use `when="always"` only when eager debug side effects are intentional.

## Query Synchronization

`snapshots()`, `clear_snapshots()`, `status()`, and `assert_ok()` accept one
keyword-only `synchronize` argument:

- `True` is the correctness-first default and synchronizes the probe's entire
  CUDA device. It may wait for unrelated streams and increases exposure to
  cross-stream dependency deadlocks.
- A `torch.cuda.Stream` synchronizes only that stream and is the recommended
  choice when the graph replay stream is known.
- A `torch.device` explicitly requests device-wide synchronization and must
  identify the probe's device.
- `False` performs no explicit synchronization. Use it only after another query
  or an application-owned stream dependency has made the results host-visible.

Only those three types are accepted; strings, integer device indices, and
`None` are rejected. A stream or device from another CUDA device is also an
error. A synchronization-enabled query during CUDA Graph capture raises an
error. Defer host queries until after capture; `False` skips synchronization but
does not make in-capture host reads meaningful.

When several queries follow one replay, synchronize once and use `False` for
the rest. This avoids repeated waits while preserving explicit ownership of the
ordering.

## Replay And Invocation Indices

Every enabled probe owns a zero-dimensional CUDA `int64` counter. It starts at
zero and a captured single-thread kernel increments it once per graph replay.
The first replay is 1. Calling the same probe several times in one capture does
not add more increments: those calls are invocation slots, and all slots from
one replay share the same `replay_index`.

The counter is per probe and monotonic for that probe's single capture. Eager
calls under `when="always"` do not increment it and eager results use index 0.
There is no reset operation; create a new probe for a new capture.

`probe.replay_index` returns a detached GPU clone, so mutating the returned
tensor cannot modify the internal counter. Getting the property itself performs
no device-to-host transfer. Operations that materialize the value on the host,
such as `print(counter)`, `counter.item()`, or `counter.cpu()`, synchronize and
transfer through PyTorch:

```python
graph.replay()
torch.cuda.current_stream().synchronize()

counter = probe.replay_index
assert counter is not None
print(counter)  # tensor(1, device='cuda:0')
```

A `RecordTensor`-only probe does not copy the counter to the host on every
replay. `snapshots()` is the explicit query point: it reads the current counter
and attaches that value to every returned capture slot. `PrintTensor` and
`CompareTensor` need the exact index inside their host callback, so probes using
those actions add one shared 8-byte counter copy per replay. When recording is
combined with either callback action, `snapshots()` reuses that pinned-host
counter value instead of performing a second counter transfer.

Each `snapshots()` call copies the latest staged bytes into a new CPU tensor.
Previously returned snapshots therefore survive later replays without an extra
`clone()`, but the probe retains only the latest value for each invocation slot,
not a replay history.

## Repeated Invocations

`CompareTensor` accepts either one CPU tensor/NumPy array or a sequence. A
single value corresponds only to invocation 0; it is not broadcast when the
same probe is called more than once in one capture. For repeated calls, pass
expected values in invocation order:

```python
probe = TensorProbe(
    "layers.hidden",
    [CompareTensor([expected_layer0, expected_layer1])],
)
```

Each capture-time call owns a logical slot with its own pinned staging storage.
One probe may have many slots in one graph, but it may not be reused by a
different capture session.

`PrintTensor` writes to `stderr`. `max_items` limits the displayed prefix,
`summary` controls aggregate statistics, and `every=N` prints all captured
invocations on graph replay indices divisible by N.

## Device Selection

`TensorProbe(..., device=None)` creates its counter on the current CUDA device.
Pass `device="cuda:1"`, `torch.device("cuda:1")`, or the integer device index
when constructing probes for another device. Every active input tensor must be
on that same device; a mismatch is an error.

An all-disabled probe remains a pure no-op: it does not load the native
extension, initialize CUDA, or allocate a counter, and `probe.replay_index` is
`None`.

## Gradient Probes

`watch_grad()` registers a normal PyTorch autograd hook and returns its
`RemovableHandle`, or `None` when the tensor does not require gradients and
`strict=False`:

```python
activation_grad_probe.watch_grad(hidden)
handle = weight_grad_probe.watch_grad(module.weight, strict=True)

# Later, after the hook is no longer needed:
handle.remove()
```

The hook calls the probe for its side effect and returns the original gradient.
It does not replace the tensor or transform gradients. The hook still follows
the probe's `when` policy: with the default `when="capture"`, an eager backward
call is a no-op and the backward work must itself be captured.

## Non-Contiguous Inputs

The default `non_contiguous="error"` avoids hidden graph-pool allocations.
`non_contiguous="copy"` explicitly permits a debug-only contiguous CUDA copy:

```python
probe = TensorProbe(
    "view",
    [RecordTensor()],
    non_contiguous="copy",
)
```

The copy consumes graph-pool memory, approximately one tensor payload per
captured probe site.

## TensorBoard

TensorBoard integration stays outside CUDA host callbacks:

```python
from torch_cudagraph_debug.tensor_debug.postprocess import (
    export_snapshots_to_tensorboard,
)

replay_stream = torch.cuda.current_stream()
graph.replay()
export_snapshots_to_tensorboard(
    writer,
    probe.snapshots(synchronize=replay_stream),
)
```

The exporter uses the real `snapshot.replay_index` as its default step. Query
and export after every synchronized replay when a replay-by-replay time series
is required.

## Operational Constraints

- Linux, CUDA, and the compiled native extension are required for enabled
  tensor probes.
- Every enabled probe adds one small device counter allocation and one
  single-thread increment kernel to its captured graph.
- Probe nodes can create large GPU bubbles and are intended for correctness
  debugging, not performance measurement.
- Shared staging means one probe's graph must not be replayed concurrently.
- Keep a probe alive while any graph containing it can replay; call `close()`
  only afterward.
- Prefer passing the replay stream to snapshot and status queries. The default
  device-wide synchronization is a correctness fallback when that stream is
  unknown; it is not the recommended performance path.

## Further Reading

- [Tensor Debug API reference](api.md#tensor-debug)
- [Tensor Debug examples](../examples/tensor_debug/README.md)
- [TensorBoard integration example](../examples/integrations/tensorboard_export.py)
