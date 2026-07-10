# Tensor Debug Guide

`tensor_debug` inserts native print, record, and check probes into a
PyTorch CUDA Graph and can compare tensor values across eager execution, graph
replays, processes, and code revisions. `TensorProbe` instruments one graph,
and `snapshot()` returns a standalone `TensorProbeSnapshot` for its latest
observations. `TensorRecorder` produces a `TensorRun` for labeled points,
persistence, and multi-point analysis. Probe snapshots and Recorder points
contain the same ownerless `TensorObservation` leaves; both workflows use
private domain collectors. See the [API reference](api.md#tensor-debug) for
exact signatures; for runnable programs,
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
values are already available. Both process tensor elements on the CUDA
host-callback thread, so cost grows with payload size and depends on dtype,
formatting or tolerance work, host CPU, and runtime. There is no portable byte
cutoff. Keep callback-backed probes small and targeted; use `RecordAction` plus
offline inspection or comparison for large tensors and latency-sensitive paths.

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
    mid = probe(static_x + 2, name="mid")

replay_stream = torch.cuda.current_stream()
graph.replay()

snapshot = probe.snapshot(synchronize=replay_stream)
print("replay", snapshot.replay_index)
for observation in snapshot.observations:
    print(observation.name, observation.invocation_index, observation.order, observation.tensor())

# The snapshot query already waited for every node in this replay.
probe.assert_check_ok(synchronize=False)
# A check mismatch is sticky: the first failure is latched and later
# correct replays still report it until the probe is replaced.
# No graph containing this probe may replay after close().
del graph
probe.close(synchronize=False)
```

One `TensorProbeSnapshot` aggregates every named capture slot visible at one
query point. `(name, invocation_index)` is the semantic key; the index starts
from zero independently for each name, while `order` preserves capture-call
order. After a graph replay, its `replay_index` identifies
that replay. By default, `snapshot.tensor()` uses `snapshot.probe_name` as
the observation name and `invocation_index=0`.
The default `when="capture"` makes eager warmup calls transparent no-ops.
Capture state is checked on the probe's configured device. Once that device is
capturing, the observed value must be a CUDA tensor on the same device; CPU or
cross-device inputs fail instead of being silently ignored.
Use `when="always"` only when eager debug side effects are intentional. Its
first eager call locks one CUDA stream; calls from another eager stream fail.
Eager calls may run before the probe's capture, but not after capture has
established the fixed slot layout.

## Complete Workflow

`TensorProbe` is the low-ceremony tool for one graph and its latest replay.
`TensorRecorder` uses the same named Observation identity and adds ordered
points, persistence, and structured offline comparison. Use it when the
question spans execution modes, replays, processes, code revisions, or devices.

The same instrumented function can feed an eager reference and a CUDA Graph
candidate:

```python
from torch_cudagraph_debug.tensor_debug import (
    TensorRecorder,
    TensorRun,
    compare_points,
)

def forward(inputs, recorder):
    inputs = recorder.observe(inputs, name="input")
    hidden = recorder.observe(inputs + 1, name="layers.0.hidden")
    return recorder.observe(hidden.square(), name="output")

replay_stream = torch.cuda.current_stream()

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

    # No future replay may occur after Recorder resources are closed.
    del graph, graph_output

eager = TensorRun.load("eager.tcgd-tensor")
candidate = TensorRun.load("cg.tcgd-tensor")
comparison = compare_points(eager["forward"], candidate["replay-1"])
comparison.assert_ok()
print(comparison.to_text())
```

`bundle_dir` must be an absent or empty directory; a nonempty target raises
`FileExistsError` at construction, so pick a fresh directory per run. Note
that after capture, before the first replay, recorded slots hold zeros:
capture installs the copy nodes but does not execute them, so query only
after a replay.

`observe()` returns the exact tensor object. Within one eager point, repeated
calls with the same name become invocation 0, 1, and so on. For a CUDA Graph
recorder, invocation indices are assigned per capture call. The stable
cross-run key is `(name, invocation_index)`; replay index is evidence rather
than identity.

An eager recorder observes calls only inside `record_point()`. A CUDA Graph
recorder ignores eager warmup, uses calls made during graph capture to establish
its fixed slot layout, and reads those slots when a later `record_point()` wraps
`graph.replay()`. All named CUDA Graph observations owned by one recorder share
one internal `RecordAction` session, one replay counter, and one counter
increment kernel. Tensor payload copies still occur once per observed slot.
Capture invocation identity is committed only after the private collector
accepts the slot. A recoverable host-side enqueue failure therefore leaves the
same `(name, invocation_index)` available for a retry.

Set `strict_scope=True` when an instrumentation call outside the active eager
point or CUDA Graph capture should be treated as a bug. The default leaves such
calls as transparent no-ops so one instrumented function can serve warmup and
collection.

The CUDA Graph scope check uses the recorder's configured device before input
validation, so a CPU or otherwise unsupported warmup value outside capture is
still a no-op; the same value is rejected when capture is active.

`record_point()` applies the same synchronization policy as quick Probe queries.
Pass the replay or eager execution stream when it is known. `False` is valid
only after the application has synchronized every relevant D2H copy.

Normal recorder context exit freezes a `complete=True` run. If the body raises,
the recorder instead freezes and persists the collected points as a terminal
`complete=False` run, blocks later collection, closes its native resources,
and lets the application exception propagate. `result` remains available for
postmortem analysis in either case.

### Full And Summary Payloads

The recorder default is `payload="full"`. Override it globally or for one
observation:

```python
recorder = TensorRecorder(execution="eager", payload="summary")
hidden = recorder.observe(hidden, name="hidden")
output = recorder.observe(output, name="output", payload="full")
```

Every observation stores metadata, SHA-256, and finite/NaN/Inf/zero counts plus
min, max, mean, standard deviation, and L2 norm. The numeric statistics use
finite elements only after conversion to `float64`; `std` is the population
standard deviation. `zero_count` is a subset of `finite_count`. Statistics are
`None` when no finite element exists.

- `full` also stores raw tensor bytes and supports allclose or exact
  comparison.
- `summary` omits raw bytes. It reduces bundle size but does not reduce the
  current D2H cost because digest and statistics are computed on the CPU.
- Equal digests prove a bitwise match. Under allclose with `equal_nan=False`,
  bit-identical NaN positions are still reported as mismatches.
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
Integer and bool comparisons remain exact. Their diagnostic differences are
computed as integers, so mismatch values and maximum absolute error preserve
adjacent `int64` values above `2**53` and differences spanning the signed
range. Mean and relative errors remain floating-point metrics.

Source stride is part of tensor metadata. The default
`layout_policy="strict"` reports a stride difference as a mismatch; use
`layout_policy="ignore"` only when different layouts are intentional.

`compare_runs(reference, candidate)` aligns points by label and reports missing
points. Pass `point_mapping` when semantically equivalent point labels differ;
unmapped labels align by identical label, and only labels covered by neither
the mapping nor auto-alignment are reported as one-sided.
`compare_point_series(reference_point, candidate_run)` compares one reference
against every candidate point in order and is intended for replay drift, stale
static inputs, and state-update bugs.

Reports identify the first issue in reference execution order and label it as
`mismatch` or `inconclusive`; a later definite mismatch still makes the overall
status `mismatch`. They rank the worst value mismatches by mismatch fraction
and absolute error. Comparison results provide `assert_ok()`, text, JSON, CSV,
and standalone HTML output. `write()` persists the report set.

The CLI performs the same offline analysis without CUDA or the native extension:

```bash
tcgd-tensor summary eager.tcgd-tensor
tcgd-tensor compare-points eager.tcgd-tensor cg.tcgd-tensor \
  --reference-point forward --candidate-point replay-1
tcgd-tensor compare-runs baseline.tcgd-tensor candidate.tcgd-tensor \
  --point-map eager-forward=graph-forward
tcgd-tensor group-summary eager-group --output eager-group-report
tcgd-tensor compare-run-groups eager-group cg-group --output group-comparison
tcgd-tensor compare-point-series eager.tcgd-tensor cg.tcgd-tensor \
  --reference-point forward --output tensor-report
```

Omit the candidate bundle from `compare-points` or `compare-point-series` to
reuse the reference bundle.

Use `--mode exact`, `--promote-dtypes`, `--ignore-layout`, `--rtol`, `--atol`,
`--equal-nan`, and `--only-changed` to control comparison and presentation.
Output directories are optional and require `--overwrite` when nonempty.
Mismatch and inconclusive reports return a nonzero status.

### Multi-Rank Run Groups

Distributed jobs write one rank-local bundle per process. Rank identity comes
from explicit `TensorRecorder(rank=..., group_id=..., world_size=...)`
arguments, or defaults from the `RANK`/`WORLD_SIZE` environment variables or
initialized `torch.distributed` (`group_id` is never auto-resolved); a run
without a rank cannot be loaded into a group. Load each parent
directory and compare common ranks without collapsing rank identity:

```python
from torch_cudagraph_debug.tensor_debug import TensorRunGroup, compare_run_groups

reference = TensorRunGroup.load("eager-group")
candidate = TensorRunGroup.load("cuda-graph-group")
comparison = compare_run_groups(reference, candidate)
print(comparison.to_text(include_unchanged=False))
comparison.assert_ok()
comparison.write("group-report")
```

`group.missing_ranks` and `group.complete` make rank coverage explicit. Missing
ranks, unknown world size, or incomplete bundles make an otherwise matching
group comparison `inconclusive`. Use `point_mapping` when point labels differ;
unmapped labels align by identical label.
Group summaries and comparisons can be rendered directly with `to_text()`,
`to_dict()`, and `to_html()` or persisted with `write()`. Use
`comparison.assert_ok()` when mismatch or inconclusive status should fail an
automated workflow.

The equivalent CLI commands are `group-summary` and `compare-run-groups`.

Bundles use the `torch-cudagraph-debug/tensor-run` schema. The manifest is
strict JSON; full payloads are content-addressed raw byte blobs. Loading is lazy
and validates payload size and SHA-256 before materialization.

## Query Synchronization

`snapshot()`, `check_status()`, `assert_check_ok()`, and `close()` accept one
keyword-only `synchronize` argument:

- `True` is the correctness-first default and synchronizes the probe's entire
  CUDA device. It may wait for unrelated streams and increases exposure to
  cross-stream dependency deadlocks.
- A `torch.cuda.Stream` synchronizes only that stream and is the recommended
  choice when the graph replay stream is known.
- A `torch.device` explicitly requests device-wide synchronization and must
  identify the probe's device.
- `False` performs no explicit synchronization. Use it only after a previous
  synchronized probe query or after the application has synchronized all
  relevant CUDA work.

Omitting `synchronize`, or passing `None`, inherits the Probe or Recorder
policy. Explicit targets accept only `bool`, `torch.cuda.Stream`, or CUDA
`torch.device`; strings, integer device indices, and CPU devices are rejected.
A stream or device from another CUDA device is also an error. A synchronization-enabled query during
CUDA Graph capture raises an error.
Defer host queries until after capture: `False` skips synchronization, and a
record-only probe still rejects `snapshot()` during capture outright because
reading its replay counter would invalidate the capture; callback-backed
probes return stale staging at best. Closing an enabled probe is
rejected during capture even with `False` because destroying captured resources
is never valid.

When several queries follow one replay, synchronize once and use `False` for
the rest. This avoids repeated waits while preserving explicit ownership of the
ordering.

`close()` inherits the object's configured synchronization policy when omitted.
Pass the replay stream to avoid waiting on unrelated streams, or pass `False`
only after prior synchronization has completed all probe work. For a captured
probe, every graph containing it must also be unable to replay again; otherwise
a later replay accesses resources released by `close()`. The probe cannot verify
either condition. An unsynchronized close is rejected while eager callbacks or
eager copies are provably in flight; the probe stays open and fully usable. Eager (`when="always"`)
probes keep one slot per observation name: repeated names sample in place
(latest value) and new names append slots, so `snapshot()` returns the latest
value of every name observed so far. In-place re-samples are counted per name
and disclosed as `TensorProbeSnapshot.eager_overwrites`; snapshot comparisons
warn when such a sample is aligned against multi-invocation observations from
a capture and identify which side should instead be collected with
`TensorRecorder`. Overwrite metadata is valid only for an eager snapshot;
names are unique, and each entry must refer to the sole invocation-0 observation
for that name. Treat observation names as a fixed vocabulary rather than
per-iteration labels. Failed validation does not consume a name's first-use
order or bind the probe to that call's eager stream.
Failed host-side insertion or replacement restores the prior eager layout and
value. Host-side capture preparation has the same rollback guarantee. A later
successful capture replaces the eager slot layout with its own; pre-capture
eager observations are dropped; their staging is retired immediately and
reclaimed only after
queued eager work has completed. If CUDA command submission fails after that
transaction is published, the probe rejects further collection and queries;
destroy the affected graph, close the probe, and create a new one.

## Replay And Invocation Indices

Every enabled probe owns a zero-dimensional CUDA `int64` counter. It starts at
zero and a captured single-thread kernel increments it once per graph replay.
The first replay is 1. Calling the same probe several times in one capture does
not add more increments: those calls create globally ordered slots, and all
slots from one replay share the same `replay_index`. Each slot also has a
semantic `(name, invocation_index)` key; repeated calls advance the
invocation index independently for that name.

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

Each `snapshot()` call copies every slot's latest staged bytes into new
CPU tensors and returns one aggregate snapshot. Previously returned snapshots
therefore survive later replays without an extra `clone()`, but the probe
retains only the latest value for each capture slot, not a replay history.

## Repeated Invocations

`CheckAction` accepts either one CPU tensor/NumPy array or a sequence. A
single value corresponds only to global capture order 0; it is not broadcast
when the same probe is called more than once in one capture. For repeated
calls, pass expected values in capture-call order:
```python
probe = TensorProbe(
    "layers.hidden",
    [CheckAction([expected_layer0, expected_layer1])],
)
```

For semantic binding independent of global capture order, pass a mapping from
`TensorObservationKey(name, invocation_index)` to expected tensors.

Relative and absolute tolerances apply only to finite values. Positive and
negative infinity require an identical same-sign expected value. NaNs match
only when `equal_nan=True`.

Each capture-time call owns a globally ordered slot with its own pinned staging
storage and a semantic `(name, invocation_index)` key.
One probe may have many slots in one graph, but it may not be reused by a
different capture session.

`PrintAction` writes to `stderr`. `max_items` limits the displayed prefix,
`summary` controls aggregate statistics, and `every=N` prints all captured
slots on graph replay indices divisible by N.

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
activation_grad_probe.watch_grad(hidden, name="activation.grad")
handle = weight_grad_probe.watch_grad(module.weight, name="weight.grad", strict=True)

# Later, after the hook is no longer needed:
handle.remove()
```

`close()` removes every hook the probe registered, so gradients computed
after close flow through untouched. Removing a handle yourself earlier is
still safe; the close-time removal is idempotent.

The optional name defaults to `probe.name`. The hook calls the probe for its side
effect and returns the original gradient. Invocation indices are assigned when
hooks actually fire, so repeated same-name hooks follow backward execution
order. It does not replace the tensor or transform gradients. The hook still follows
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

During capture, the copy consumes graph-pool memory, approximately one tensor
payload per captured probe site, and remains owned until close because the graph
references its address. In eager `when="always"` use, the temporary copy is
recorded on the probe's owning stream and is released after queued work
completes rather than retained for the probe lifetime.

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
- `PrintAction` and `CheckAction` host callbacks can create large GPU bubbles
  and are intended for targeted correctness debugging, not performance
  measurement. Their cost scales with payload and has no hardware-independent
  byte threshold; prefer `RecordAction` plus offline analysis for large tensors.
- Shared staging means one probe's graph must not be replayed concurrently.
- Eager `when="always"` use is also single-stream, and a probe cannot return to
  eager use after its capture.
- Keep a probe alive while any graph containing it can replay; call `close()`
  only afterward. `close()` accepts the same bool/stream/device synchronization
  targets as queries; an enabled probe rejects close during capture.
- Prefer passing the replay stream to snapshot and check-status queries. The
  default device-wide synchronization is a correctness fallback when that
  stream is unknown; it is not the recommended performance path.

## Further Reading

- [Tensor Debug API reference](api.md#tensor-debug)
- [Tensor Debug examples](../examples/tensor_debug/README.md)
- [TensorBoard integration example](../examples/integrations/tensorboard_export.py)
