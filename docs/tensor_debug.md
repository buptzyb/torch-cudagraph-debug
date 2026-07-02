# Tensor Debug Guide

`tensor_debug` inserts native print, record, and check probes into a
PyTorch CUDA Graph and can compare tensor values across eager execution, graph
replays, processes, and code revisions. `TensorProbe` is the quick workflow for
one graph; `TensorRecorder` is the complete workflow for labeled points,
persistence, and multi-point analysis. Both produce ownerless
`TensorObservation` leaves through private domain collectors. For exact
signatures, see the [API reference](api.md#tensor-debug); for runnable programs,
follow the [Tensor Debug examples](../examples/tensor_debug/README.md).

## Core Usage

Import tensor APIs from their domain module:

```python
from torch_cudagraph_debug.tensor_debug import (
    CheckAction,
    PrintAction,
    RecordAction,
    TensorProbe,
    compare_snapshots,
)
```

For most debugging sessions, start with `RecordAction`: it avoids a CUDA host
callback when used alone and keeps inspection in Python after synchronization.
It still enqueues a device-to-host copy into pinned staging memory for every
captured invocation, so keep probes limited to tensors needed for debugging.
Use `PrintAction` for immediate console output and `CheckAction` when expected
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
        PrintAction(max_items=8),
        RecordAction(),
        CheckAction(expected),
    ],
)

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    mid = probe(static_x + 2)

replay_stream = torch.cuda.current_stream()
graph.replay()

snapshot = probe.snapshot(synchronize=replay_stream)
print("replay", snapshot.replay_index)
for observation in snapshot.observations:
    print(observation.invocation_index, observation.tensor())

# The snapshot query already waited for every node in this replay.
probe.assert_check_ok(synchronize=False)
probe.close()
```

One `TensorProbeSnapshot` represents one replay and aggregates every invocation
slot in capture-call order. `snapshot.tensor()` is shorthand for invocation 0.
The default `when="capture"` makes eager warmup calls transparent no-ops.
Use `when="always"` only when eager debug side effects are intentional.

## Complete Workflow

`TensorProbe` is the low-ceremony tool for one graph and its latest replay.
`TensorRecorder` adds named observations, ordered points, persistence, and
offline comparison. Use it when the question spans execution modes, replays,
processes, code revisions, or devices.

The same instrumented function can feed an eager reference and a CUDA Graph
candidate:

```python
from torch_cudagraph_debug.tensor_debug import (
    TensorRecorder,
    TensorRun,
    compare_points,
)

def forward(inputs, recorder):
    inputs = recorder.observe("input", inputs)
    hidden = recorder.observe("layers.0.hidden", inputs + 1)
    return recorder.observe("output", hidden.square())

with TensorRecorder(
    execution="eager",
    name="eager",
    bundle_dir="eager.tcgd-tensor",
) as eager_recorder:
    with eager_recorder.record_point("forward", synchronize=replay_stream):
        eager_output = forward(static_x, eager_recorder)

with TensorRecorder(
    execution="cuda_graph",
    name="cuda-graph",
    bundle_dir="cg.tcgd-tensor",
) as graph_recorder:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = forward(static_x, graph_recorder)

    with graph_recorder.record_point("replay-1", synchronize=replay_stream):
        graph.replay()

eager = TensorRun.load("eager.tcgd-tensor")
candidate = TensorRun.load("cg.tcgd-tensor")
comparison = compare_points(eager["forward"], candidate["replay-1"])
comparison.assert_ok()
print(comparison.to_text())
```

`observe()` returns the exact tensor object. Within one point, repeated calls
with the same name become invocation 0, 1, and so on. The stable cross-run key
is `(probe_name, invocation_index)`; replay index is evidence rather than
identity.

An eager recorder observes calls only inside `record_point()`. A CUDA Graph recorder
ignores eager warmup, uses calls made during graph capture to establish its
fixed slot layout, and reads those slots when a later `record_point()` wraps
`graph.replay()`. All named CG observations owned by one recorder share one
internal `RecordAction` session, one replay counter, and one counter increment
kernel. Tensor payload copies still occur once per observed slot.

`record_point()` applies the same synchronization policy as quick Probe queries.
Pass the replay or eager execution stream when it is known. `False` is valid
only after the application has already made every D2H copy host-visible.

### Full And Summary Payloads

The recorder default is `payload="full"`. Override it globally or for one
observation:

```python
recorder = TensorRecorder(execution="eager", payload="summary")
hidden = recorder.observe("hidden", hidden)
output = recorder.observe("output", output, payload="full")
```

Every observation stores metadata, SHA-256, and finite/NaN/Inf/zero counts plus
min, max, mean, standard deviation, and L2 norm.

- `full` also stores raw tensor bytes and supports allclose or exact
  comparison.
- `summary` omits raw bytes. It reduces bundle size but does not reduce the
  current D2H cost because digest and statistics are computed on the CPU.
- Equal digests prove an exact match.
- Different summary digests prove a bitwise change. They cannot determine
  whether the change is within a nonzero tolerance, so allclose reports
  `inconclusive`.
- Exact comparison may classify different summary digests as a mismatch.

`match`, `mismatch`, and `inconclusive` are distinct report states. A
comparison is successful only when every aligned observation matches.

### Point, Run, And Point-Series Comparison

For two standalone Probe snapshots, use either form:

```python
comparison = probe.compare(before, after)
comparison = compare_snapshots(eager_snapshot, graph_snapshot)
```

The method validates same-probe ownership and ordering. The top-level function
also supports independent probes, including eager-to-CUDA-Graph comparison.

`compare_points(reference, candidate)` is the primitive. It validates shape
and dtype, then compares values using allclose by default. Missing observations
and invocations are mismatches. Set `mode="exact"` for raw-value identity or
`dtype_policy="promote"` to explicitly compare different numeric dtypes after
promotion.

`compare_runs(reference, candidate)` aligns points by label and reports
missing points. `compare_point_series(reference_point, candidate_run)` compares one
reference against every candidate point in order. Point-series comparison is intended
for replay drift, stale static inputs, and state-update bugs.

Reports identify the first issue in reference execution order and label it as
`mismatch` or `inconclusive`; a later definite mismatch still makes the overall
status `mismatch`. They rank the worst value mismatches by mismatch fraction
and absolute error. Result objects provide text, JSON, CSV, and standalone HTML
output through `write()`.

The CLI performs the same offline analysis without CUDA or the native extension:

```bash
tcgd-tensor summary eager.tcgd-tensor
tcgd-tensor compare-points eager.tcgd-tensor cg.tcgd-tensor \
  --reference-point forward --candidate-point replay-1
tcgd-tensor compare-runs baseline.tcgd-tensor candidate.tcgd-tensor
tcgd-tensor compare-point-series eager.tcgd-tensor cg.tcgd-tensor \
  --reference-point forward --output tensor-report
```

Omit the candidate bundle from `compare-points` or `compare-point-series` to
reuse the reference bundle.

Use `--mode exact`, `--promote-dtypes`, `--rtol`, `--atol`,
`--equal-nan`, and `--only-changed` to control comparison and presentation.
Mismatch and inconclusive reports return a nonzero status.

Bundles use the `torch-cudagraph-debug/tensor-run` schema. The manifest is
strict JSON; full payloads are content-addressed raw byte blobs. Loading is lazy
and validates payload size and SHA-256 before materialization.

## Query Synchronization

`snapshot()`, `clear_snapshot()`, `check_status()`, and `assert_check_ok()` accept one
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

A `RecordAction`-only probe does not copy the counter to the host on every
replay. `snapshot()` is the explicit query point: it reads the current counter
and attaches that value to the aggregate snapshot. `PrintAction` and
`CheckAction` need the exact index inside their host callback, so probes using
those actions add one shared 8-byte counter copy per replay. When recording is
combined with either callback action, `snapshot()` reuses that pinned-host
counter value instead of performing a second counter transfer.

Each `snapshot()` call copies every invocation's latest staged bytes into new
CPU tensors and returns one aggregate snapshot. Previously returned snapshots
therefore survive later replays without an extra `clone()`, but the probe
retains only the latest value for each invocation slot, not a replay history.

## Repeated Invocations

`CheckAction` accepts either one CPU tensor/NumPy array or a sequence. A
single value corresponds only to invocation 0; it is not broadcast when the
same probe is called more than once in one capture. For repeated calls, pass
expected values in invocation order:

```python
probe = TensorProbe(
    "layers.hidden",
    [CheckAction([expected_layer0, expected_layer1])],
)
```

Each capture-time call owns a logical slot with its own pinned staging storage.
One probe may have many slots in one graph, but it may not be reused by a
different capture session.

`PrintAction` writes to `stderr`. `max_items` limits the displayed prefix,
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
    [RecordAction()],
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
snapshot = probe.snapshot(synchronize=replay_stream)
export_snapshots_to_tensorboard(
    writer,
    [snapshot],
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
- Prefer passing the replay stream to snapshot and check-status queries. The default
  device-wide synchronization is a correctness fallback when that stream is
  unknown; it is not the recommended performance path.

## Further Reading

- [Tensor Debug API reference](api.md#tensor-debug)
- [Tensor Debug examples](../examples/tensor_debug/README.md)
- [TensorBoard integration example](../examples/integrations/tensorboard_export.py)
