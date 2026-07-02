# torch-cudagraph-debug API Reference

This reference describes the current public API.

## Package Root

```python
from torch_cudagraph_debug import (
    CudaGraphDebugError,
    NativeExtensionUnavailableError,
    __version__,
)
```

The package root contains only version and common errors. Import each debugging
domain explicitly.

## Tensor Debug

Concept guide: [Tensor Debug guide](tensor_debug.md). Runnable guide:
[Tensor Debug examples](../examples/tensor_debug/README.md).

### Supported Public Imports

```python
from torch_cudagraph_debug.tensor_debug import (
    TensorProbe,
    TensorRecorder,
    PrintAction,
    RecordAction,
    CheckAction,
    TensorProbeSnapshot,
    TensorCheckStatus,
    TensorRun,
    TensorPoint,
    TensorObservation,
    TensorObservationKey,
    TensorValueSummary,
    TensorComparisonOptions,
    TensorObservationComparison,
    TensorSnapshotComparison,
    TensorPointComparison,
    TensorRunComparison,
    TensorPointSeriesComparison,
    compare_snapshots,
    compare_points,
    compare_runs,
    compare_point_series,
    TensorDebugError,
    TensorCheckError,
    TensorComparisonError,
    TensorBundleError,
    TensorOwnershipError,
    TensorPayloadUnavailableError,
)
```

### TensorProbe

```python
TensorProbe(
    name: str,
    actions: Sequence[PrintAction | RecordAction | CheckAction],
    *,
    non_contiguous: Literal["error", "copy"] = "error",
    when: Literal["capture", "always"] = "capture",
    device: torch.device | str | int | None = None,
)
```

`name` and `actions` must be nonempty. Disabled actions are filtered before
native probe creation. If every action has `enabled=False`, the probe is a
pure no-op: it does not load the native extension, initialize CUDA, or allocate
a replay counter.

Every enabled probe owns a zero-dimensional CUDA `int64` replay counter on
`device`, or on the current CUDA device when `device=None`. An integer device
index is accepted. Every active source tensor must be on the same device.

`probe(tensor)` returns the exact input tensor object. In
`when="capture"`, calls outside CUDA stream capture perform no validation,
allocation, copy, callback, print, record, or check work. In
`when="always"`, eager calls execute debug work too. The first eager call
locks one CUDA stream; an eager call from another stream is rejected. Eager
calls may precede the probe's one graph capture, but eager calls after capture
are rejected.

The non-contiguous policy is probe-wide because all actions on one probe inspect
the same source tensor:

- `"error"`: reject a non-contiguous tensor when debug work is active.
- `"copy"`: create an internal contiguous CUDA copy for the debug path while
  returning the original tensor. Captured slots retain that source because the
  graph references its address. Eager copies are recorded on the owning stream
  and are not retained by the probe after queued work completes.

A probe is owned by the first CUDA Graph capture session that uses it. Multiple
calls in that capture create logical slots in invocation order. Reusing the
probe in another capture is an error. The GPU counter advances once per graph
replay, not once per invocation, so all slots in one replay share a 1-based
replay index. Eager `when="always"` calls do not advance the counter and use
index 0.

#### Methods

```python
probe(tensor: torch.Tensor) -> torch.Tensor
probe.replay_index -> torch.Tensor | None
probe.snapshot(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
) -> TensorProbeSnapshot
probe.compare(
    reference: TensorProbeSnapshot,
    candidate: TensorProbeSnapshot,
    *,
    options: TensorComparisonOptions | None = None,
) -> TensorSnapshotComparison
probe.clear_snapshot(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
) -> None
probe.check_status(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
) -> TensorCheckStatus
probe.assert_check_ok(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
) -> None
probe.watch_grad(
    tensor: torch.Tensor,
    *,
    strict: bool = False,
) -> torch.utils.hooks.RemovableHandle | None
probe.close(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
) -> None
```

`replay_index` returns a detached GPU clone of the internal scalar. Mutating
it cannot change the probe. Reading the property performs no device-to-host
copy; printing it, calling `.item()`, or moving it to CPU materializes the
value through PyTorch. It is `None` for an all-disabled probe.

The query methods and `close()` share one synchronization contract. `True`
synchronizes the probe's entire CUDA device and is the correctness-first
default. It can wait for unrelated streams and broaden cross-stream deadlock
risk. Passing a `torch.cuda.Stream` synchronizes only that stream and is the
recommended path when the replay stream is known. Passing a `torch.device`
explicitly requests device-wide synchronization. `False` skips explicit
synchronization and requires caller-owned ordering. A stream or device must
match the probe device. Strings, integer device indices, `None`, and CPU
devices are rejected. Synchronization-enabled queries are invalid during CUDA
Graph capture; for an enabled probe, `close()` is invalid during capture for
every synchronization setting.

`snapshot()` returns one aggregate `TensorProbeSnapshot` containing the latest
value of every capture-time invocation slot. Capture installs copy nodes but
does not execute them. A Record-only probe reads its GPU counter at this query
point. When Print or Check is also enabled, the query reuses their existing
pinned-host counter staging instead of issuing another counter transfer. It
raises `TensorDebugError` when no `RecordAction` is enabled or no invocation has
recorded a value.

`compare()` accepts two snapshots owned by this probe in chronological replay
order and returns the same result type as top-level `compare_snapshots()`.

`clear_snapshot()` synchronizes according to the same policy before zeroing
retained host storage. It does not remove graph nodes or release the probe.

`check_status()` returns:

```python
@dataclass(frozen=True)
class TensorCheckStatus:
    ok: bool
    message: str
    replay_index: int
    invocation_index: int
```

`check_status()` synchronizes callback-backed check state before returning it.
`assert_check_ok()` delegates to `check_status()` and therefore synchronizes at
most once; it raises `TensorCheckError` when the sticky check state is not OK.

`watch_grad()` registers an autograd hook. The hook probes the gradient for its
side effect and returns the original gradient. It obeys the probe's `when`
policy, so default capture-only probes do no work during eager backward. The
method returns a removable hook handle. If `tensor.requires_grad` is false, it
returns `None`; with `strict=True`, it raises `RuntimeError`.

`close()` first performs the requested synchronization, reclaims retired host
staging, and releases native resources. An unsynchronized close fails while an
eager host callback is still pending. Do not close a probe while a graph that
captured it may still replay. `TensorProbe` is also a context manager whose exit
uses the correctness-first default close; use that form only when every replay
occurs inside the context.

### Actions

```python
PrintAction(
    max_items: int = 16,
    every: int = 1,
    summary: bool = True,
    enabled: bool = True,
)

RecordAction(
    enabled: bool = True,
)

CheckAction(
    expected: torch.Tensor | numpy.ndarray | Sequence[torch.Tensor | numpy.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    enabled: bool = True,
)
```

When debug work is active, an enabled probe supports contiguous CUDA tensors
with `float16`, `bfloat16`, `float32`, `float64`, `uint8`, `int8`, `int16`,
`int32`, `int64`, or `bool` dtype. Other dtypes are rejected.

`PrintAction` writes to `stderr` from a native CUDA host callback. `max_items`
limits the displayed value prefix, `summary=False` omits aggregate statistics,
and `every=N` prints every invocation on graph replay indices divisible by N.

`RecordAction` enqueues a device-to-host tensor copy into pinned staging
memory without a host callback when used alone. The probe stores only the latest
staged value for each invocation slot. Its counter remains on the GPU during
replay and transfers only when `snapshot()` explicitly queries it. When Record
is combined with Print or Check, that query reuses the callback actions'
single shared 8-byte counter transfer.

`CheckAction.expected` values must be on CPU and match the captured source's
shape and dtype. A single tensor or array is normalized to a one-item expected
sequence and therefore applies only to invocation 0. It is never broadcast.
For a probe called multiple times in one capture, pass one expected value per
invocation in capture-call order.

A check mismatch is sticky until the probe is destroyed. The first mismatch
records the replay and invocation indices.

`PrintAction` and `CheckAction` process tensor elements on a CUDA host callback.
Their latency grows with payload size and also depends on dtype, formatting,
tolerance checks, host CPU, and runtime, so there is no portable byte cutoff.
Keep callback-backed probes small and targeted. For large tensors or
latency-sensitive paths, use `RecordAction`, synchronize outside replay, and
perform comparison or formatting offline.

### TensorProbeSnapshot

```python
@dataclass(frozen=True, eq=False)
class TensorProbeSnapshot:
    probe_id: str
    probe_name: str
    replay_index: int
    timestamp: float
    observations: tuple[TensorObservation, ...]

snapshot.by_key -> Mapping[TensorObservationKey, TensorObservation]
snapshot.observation(invocation_index=0) -> TensorObservation
snapshot.tensor(invocation_index=0) -> torch.Tensor
snapshot.descriptor() -> dict
```

Every `snapshot()` call materializes new CPU tensor copies from the latest
pinned staging bytes and records the probe's current replay counter. One
snapshot represents one query point; after a graph replay, its replay index
identifies that replay. Its observations represent invocation slots in
capture-call order. Returned snapshots remain unchanged across later replays,
so an additional `clone()` is not required for retention. A snapshot queried
after capture but before its first replay uses index 0; after replay, captured
snapshots are 1-based. Eager `when="always"` snapshots also use index 0.

`probe_id` is an ownership token used by `TensorProbe.compare()`. It is not a
run identity and is not attached to the ownerless `TensorObservation` leaves.

### TensorRecorder And TensorRun

```python
TensorRecorder(
    *,
    execution: Literal["eager", "cuda_graph"],
    name: str = "run",
    bundle_dir: str | Path | None = None,
    device: torch.device | str | int | None = None,
    payload: Literal["full", "summary"] = "full",
    non_contiguous: Literal["error", "copy"] = "error",
    synchronize: bool | torch.cuda.Stream | torch.device = True,
    rank: int | None = None,
    group_id: str | None = None,
    world_size: int | None = None,
    run_metadata: Mapping[str, Any] | None = None,
)

recorder.observe(name, tensor, *, payload=None) -> torch.Tensor
recorder.watch_grad(
    name,
    tensor,
    *,
    payload=None,
    strict=False,
) -> RemovableHandle | None
recorder.record_point(
    label,
    *,
    metadata=None,
    synchronize=None,
) -> ContextManager[None]
recorder.snapshot_run() -> TensorRun
recorder.finish() -> TensorRun
recorder.result -> TensorRun
recorder.close(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
) -> None
```

`execution` is explicit because eager execution and graph replay have
different collection boundaries. Eager `observe()` calls are active only
inside `record_point()`. A CUDA Graph recorder uses `observe()` calls during its
one capture session to define slots and reads their latest values when
`record_point()` later wraps a replay. Calls outside those active paths are
transparent no-ops.

All observations must be supported CUDA tensors on the recorder device.
`non_contiguous` has the same error/copy behavior as `TensorProbe`. One
recorder-wide native `RecordAction` session collects every CUDA Graph observation,
so logical names do not create one replay counter kernel each.

`record_point()` labels must be nonempty and unique, and point contexts cannot
nest. Its synchronization target inherits the recorder default when omitted.
Interrupted contexts do not append a point.

`finish()` is idempotent, writes `complete=True`, and rejects later points. It
does not release captured native storage. `close()` releases that storage after
the requested bool/stream/device synchronization and requires that the graph
can no longer replay. A normal context-manager exit calls `finish()` and then
`close()` with the recorder's configured synchronization target. If the block
raises, the recorder instead freezes and persists its partial result with
`complete=False`, blocks later collection, closes resources, and does not
suppress the application exception. `result` and `snapshot_run()` expose that
same terminal incomplete run.

```python
@dataclass(frozen=True)
class TensorRun:
    run_id: str
    name: str
    execution: Literal["eager", "cuda_graph"]
    rank: int | None
    created_at: float
    finished_at: float | None
    complete: bool
    default_payload: Literal["full", "summary"]
    points: tuple[TensorPoint, ...]
    group_id: str | None
    world_size: int | None
    provenance: Mapping[str, Any]
    run_metadata: Mapping[str, Any]
    bundle_dir: Path | None

TensorRun.load(bundle_dir, *, cache_tensors=False) -> TensorRun
run.point(ref: str | int | TensorPoint) -> TensorPoint
run[ref: str | int] -> TensorPoint
run.compare(reference, candidate, *, options=None) -> TensorPointComparison
```

A `TensorPoint` owns a point-level optional `replay_index` and ordered
`TensorObservation` objects.
`point.observation(name, invocation_index=0)` performs stable-key lookup.
A foreign point passed to `run.point()` raises `TensorOwnershipError`.

`TensorObservation` is an ownerless leaf shared by Probe snapshots and Recorder
points. It records local order, name, invocation, shape, stride, dtype, source
device, payload kind, nbytes, SHA-256, and a `TensorValueSummary`.
`observation.tensor()` lazily returns a CPU tensor for a full payload and
validates blob size and digest. It raises `TensorPayloadUnavailableError` for a
summary-only observation.

### Offline Tensor Comparison

```python
TensorComparisonOptions(
    mode: Literal["allclose", "exact"] = "allclose",
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    dtype_policy: Literal["strict", "promote"] = "strict",
    limit: int = 20,
)

compare_snapshots(reference, candidate, *, options=None) -> TensorSnapshotComparison
compare_points(reference, candidate, *, options=None) -> TensorPointComparison
compare_runs(
    reference,
    candidate,
    *,
    point_mapping=None,
    options=None,
) -> TensorRunComparison
compare_point_series(reference, candidates, *, options=None) -> TensorPointSeriesComparison
```

`compare_snapshots()` compares standalone Probe snapshots. It does not require
matching `probe_id` values, so eager and CUDA Graph snapshots collected by
different probes can be compared directly. `TensorSnapshotComparison` and
`TensorPointComparison` are sibling result types with direct `reference`,
`candidate`, `options`, and `observation_comparisons` fields; snapshot results
do not contain a synthetic `point_comparison`.

The stable observation key is `(probe_name, invocation_index)`.
`compare_points()` follows reference order and appends candidate-only keys.
Missing keys, shape changes, and strict dtype changes are mismatches.

Allclose uses the reference tensor in
`atol + rtol * abs(reference)`. Integer and bool values compare exactly.
`dtype_policy="promote"` explicitly converts both values with
`torch.promote_types()`. Exact comparison uses raw value bytes when dtypes
match.

`TensorObservationComparison` contains status, reason, both observation
descriptors, mismatch count and fraction, max absolute and relative error, mean
absolute error, and the first mismatching coordinate and values when full
payloads make those metrics available.

`TensorSnapshotComparison`, `TensorPointComparison`, `TensorRunComparison`,
and `TensorPointSeriesComparison` expose `status`, `ok`, `conclusive`, `to_text()`,
`to_dict()`, `to_html()`, and `write()`. A point report provides `first_issue`,
`worst_observation_comparisons()`, and `assert_ok()`. Text, HTML, and CSV include
matches by default and accept `include_unchanged=False`; JSON remains complete.

Summary comparison uses three states. Equal digests match. Different digests
are a mismatch in exact mode, but are inconclusive in allclose mode when either
full payload is unavailable.

### Tensor Bundle And CLI

The schema is `torch-cudagraph-debug/tensor-run`:

```text
manifest.json
blobs/
  <sha256>.bin
  ...
```

The strict manifest stores run identity, execution mode, rank fields,
provenance, metadata, points, observation descriptors, summaries, and
content-addressed blob paths. Writes use temporary siblings and atomic
replacement. Identical raw bytes across points share one blob. Loading is
CPU-only, lazy, and defaults to no tensor cache.

```text
tcgd-tensor summary BUNDLE
tcgd-tensor compare-points REFERENCE_BUNDLE [CANDIDATE_BUNDLE] \
  --reference-point LABEL --candidate-point LABEL
tcgd-tensor compare-runs REFERENCE_BUNDLE CANDIDATE_BUNDLE
tcgd-tensor compare-point-series REFERENCE_BUNDLE [CANDIDATE_BUNDLE] \
  --reference-point LABEL
```

For `compare-points` and `compare-point-series`, omitting `CANDIDATE_BUNDLE`
reuses the reference bundle.

Comparison commands accept `--mode`, `--rtol`, `--atol`,
`--equal-nan`, `--promote-dtypes`, `--limit`, `--only-changed`, and
`--output`. Exit status is zero only for a complete match, one for mismatch
or inconclusive, and two for invalid input or an operational error.

`TensorBundleError` reports malformed or unreadable bundles.
`TensorOwnershipError` reports foreign run objects, and
`TensorPayloadUnavailableError` reports attempts to materialize summary-only
values.

### TensorBoard Export

```python
from torch_cudagraph_debug.tensor_debug.postprocess import (
    export_snapshots_to_tensorboard,
)

export_snapshots_to_tensorboard(
    writer,
    snapshots,
    *,
    tag_prefix: str = "",
    step: int | Callable[[TensorProbeSnapshot], int] | None = None,
    write_scalars: bool = True,
    write_histograms: bool = False,
) -> None
```

The helper accepts an existing TensorBoard-compatible writer and does not import
TensorBoard itself. The caller owns synchronization and writer lifecycle. By
default it uses the real `snapshot.replay_index` as the TensorBoard step. Pass
an integer or callable through `step` only when an application-specific step is
preferred. Scalar export writes `numel`,
`mean`, `std`, `min`, `max`, and `l2_norm`; histogram export is opt-in. See the
[TensorBoard integration example](../examples/integrations/tensorboard_export.py)
for replay-by-replay retention and export. Every observation in an aggregate
snapshot is exported; repeated invocations receive an `/invocation_N` suffix.

## Memory Debug

Concept guide: [Memory Debug guide](memory_debug.md). Runnable guide:
[Memory Debug examples](../examples/memory_debug/README.md).

### Supported Public Imports

```python
from torch_cudagraph_debug.memory_debug import (
    MemoryProbe,
    MemoryProbeSnapshot,
    MemoryRecorder,
    MemoryRun,
    MemoryRunGroup,
    MemoryPoint,
    MemoryObservation,
    MemoryObservationKey,
    MemoryStats,
    MemoryRange,
    MemoryTimeline,
    MemorySnapshotComparison,
    MemoryPointComparison,
    MemoryPhaseComparison,
    MemoryRunGroupSummary,
    MemoryRunGroupPhaseComparison,
    MemoryAllocationLifetimeAnalysis,
    MemoryAttributionOptions,
    compare_snapshots,
    compare_points,
    compare_phases,
    compare_run_group_phases,
    MemoryDebugError,
    MemoryHistoryError,
    MemoryBundleError,
    MemoryOwnershipError,
)
```

The facade is intentionally limited to the names above. Raw snapshot helpers are
in `memory_debug.advanced` and are experimental.

### MemoryProbe

```python
MemoryProbe(
    name: str = "memory",
    *,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
)

probe.snapshot(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device | None = None,
) -> MemoryProbeSnapshot
probe.compare(
    reference: MemoryProbeSnapshot,
    candidate: MemoryProbeSnapshot,
    *,
    pool_mapping: Mapping[object, object] | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemorySnapshotComparison
```

`MemoryProbe` is the low-ceremony workflow for one-off allocator snapshots and
two-point comparison. It discovers every pool and stream present in
`torch.cuda.memory._snapshot()`; callers do not pass pool handles. `snapshot()`
uses the Probe default synchronization policy unless an override is supplied.

```python
@dataclass(frozen=True)
class MemoryProbeSnapshot:
    probe_id: str
    probe_name: str
    index: int
    timestamp: float
    boundary_marker: str
    observations: tuple[MemoryObservation, ...]
    warnings: tuple[str, ...]

snapshot.by_key -> Mapping[MemoryObservationKey, MemoryObservation]
snapshot.observation_stats -> Mapping[MemoryObservationKey, MemoryStats]
snapshot.pool_stats -> Mapping[PoolId, MemoryStats]
snapshot.allocator_scope_stats -> Mapping[str, MemoryStats]
snapshot.observation(pool_id, stream) -> MemoryObservation
snapshot.raw_snapshot() -> Mapping | Sequence
snapshot.descriptor() -> dict
```

`probe.compare()` requires snapshots owned by that probe in increasing index
order. Top-level `compare_snapshots()` also accepts independent probes. A
same-probe comparison can use marker-delimited allocator events and lifetime
analysis when the application enabled allocator history before the allocations
of interest. Cross-probe comparison supports state and allocation stacks but
rejects events and lifetimes because there is no shared event interval. An
embedded same-probe lifetime report uses `source_kind="probe"`; a Run lifetime
report uses `source_kind="run"`. Neither path fabricates the other workflow.

`MemoryProbeSnapshot` is not persisted and does not belong to a run. Use
`MemoryRecorder` when labels, a timeline, phases, bundles, run groups, or other
run-scoped features are needed.

### MemoryRecorder

```python
MemoryRecorder(
    *,
    name: str = "run",
    bundle_dir: str | pathlib.Path | None = None,
    rank: int | None = None,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
    group_id: str | None = None,
    world_size: int | None = None,
    run_metadata: Mapping[str, JSONValue] | None = None,
)
```

The recorder always collects `torch.cuda.memory._snapshot()`. It does not
accept pool handles, `include_traces`, or allocator-history configuration.
When `rank=None`, a numeric `RANK` environment variable is used if present.
`world_size=None` similarly checks `WORLD_SIZE`; initialized
`torch.distributed` is the fallback for both. When both values are known, rank
must be in `[0, world_size)`.

When `bundle_dir` is set, the directory must be absent or empty. Construction
creates an incomplete manifest. Every `record_point()` writes one gzip JSON
snapshot and updates that manifest; `finish()` marks it complete. A nonempty
target directory raises `FileExistsError`.

The synchronization contract matches the tensor domain. `True` synchronizes
the current CUDA device, a CUDA stream synchronizes only that stream, a CUDA
device explicitly requests device-wide synchronization, and `False` leaves
ordering to the caller. Synchronization is always skipped during current-stream
capture because it is capture-illegal; the point records a warning.

`group_id` identifies per-rank bundles from one distributed execution.
`run_metadata` is application-owned JSON metadata for source revision,
scenario, command, or other reproducibility fields. The recorder also captures
package/Python/platform/PyTorch/CUDA provenance automatically. Initialized GPU
properties are added only after a real snapshot, so recorder construction does
not initialize CUDA for metadata collection.

#### record_point

```python
recorder.record_point(
    label: str,
    *,
    metadata: Mapping[str, JSONValue] | None = None,
    synchronize: bool | torch.cuda.Stream | torch.device | None = None,
) -> MemoryPoint
```

Labels must be nonempty and unique. Outside CUDA capture, `record_point()`
synchronizes according to the Recorder default or its per-call override. During
current-stream capture it skips requested synchronization and snapshots
immediately.

The recorder temporarily sets a PyTorch allocator metadata marker around the
snapshot when those private APIs are available. This marker delimits same-run
event windows. Marker failures become point warnings.

Snapshot and metadata values must be JSON-compatible: null, string, bool,
finite number, list/tuple, or a mapping with string keys. Validation errors
include the path to the unsupported value.

#### Lifecycle

```python
recorder.snapshot_run() -> MemoryRun
recorder.finish() -> MemoryRun
recorder.result -> MemoryRun
```

`snapshot_run()` returns an immutable incomplete view without stopping
collection. `finish()` is idempotent, makes later `record_point()` calls
invalid, writes `complete=True`, and returns an immutable run. On an exception,
context-manager exit instead freezes the collected points, writes a terminal
`complete=False` manifest, makes later collection invalid, and does not
suppress the application exception. `result` is available after either normal
or exceptional context exit; `snapshot_run()` and a later idempotent `finish()`
return that same terminal result.

For tests only, `MemoryRecorder._from_snapshot_provider(...)` injects synthetic
snapshots without exposing a provider in the public constructor.

### MemoryPoint

A point owns an ordered tuple of allocator observations. Each observation is one
`(pool_id, stream)` state; pool and scope mappings are cached derivations.

```python
@dataclass(frozen=True)
class MemoryObservation:
    order: int
    pool_id: PoolId
    stream: Any
    stats: MemoryStats

point.observations -> tuple[MemoryObservation, ...]
point.by_key -> Mapping[MemoryObservationKey, MemoryObservation]
point.observation(pool_id, stream) -> MemoryObservation
point.observation_stats
point.pool_stats
point.allocator_scope_stats  # all/default/private MemoryStats
point.raw_snapshot() -> Mapping | Sequence
point.descriptor() -> dict
```

`MemoryObservation` is an ownerless leaf shared by Probe snapshots and Recorder
points. Ownership, labels, timestamps, and boundary markers live on the
containing `MemoryProbeSnapshot` or `MemoryPoint`.

`MemoryStats` contains `reserved_bytes`, `allocated_bytes`, `active_bytes`,
`requested_bytes`, `segment_count`, `block_count`, and
`largest_inactive_block_bytes`. Its `inactive_bytes` and
`fragmentation_bytes` properties are derived values.

For persisted runs, `raw_snapshot()` lazily reads the point's gzip JSON payload.
Runs loaded with `cache_snapshots=True` retain the decompressed payload after
the first call; `cache_snapshots=False` rereads it on every call.

### MemoryRun

```python
MemoryRun.load(bundle_dir, *, cache_snapshots=True) -> MemoryRun
run.descriptor() -> dict
run.point(label_or_index_or_point) -> MemoryPoint
run[label_or_index] -> MemoryPoint
run.between(start, end) -> MemoryRange
run.compare(reference, candidate, *, attribution=None) -> MemoryPointComparison
run.timeline(*, attribution=None) -> MemoryTimeline
run.lifetimes(
    at=None,
    *,
    born_between=None,
    through=None,
    attribution=None,
) \
    -> MemoryAllocationLifetimeAnalysis
```

`MemoryRun` also carries `rank`, `group_id`, `world_size`, immutable automatic
`provenance`, and immutable user `run_metadata`.

A point passed to `point()`, `between()`, or `compare()` must carry the
same `run_id` and correspond to a point in that run. Otherwise
`MemoryOwnershipError` is raised. `end` must have a larger index than `start`.

Same-run comparison can use allocator addresses to report lifecycle
observations:

- new and removed segment bytes;
- bytes that became active;
- bytes released back to inactive storage.

These observations are snapshot-to-snapshot address comparisons, not historical
allocator events.

### MemoryRange

```python
@dataclass(frozen=True)
class MemoryRange:
    run: MemoryRun
    start: MemoryPoint
    end: MemoryPoint

memory_range.compare(*, attribution=None) -> MemoryPointComparison
```

`run.between()` validates that both points belong to the run and that `end`
follows `start`. `MemoryRange.compare()` is equivalent to calling
`run.compare(memory_range.start, memory_range.end, ...)`. Phase comparison
accepts one baseline range and one candidate range.

### MemoryAttributionOptions

```python
MemoryAttributionOptions(
    stacks: bool = False,
    events: bool = False,
    lifetimes: bool = False,
    on_missing: Literal["warn", "error"] = "warn",
    stack_depth: int = 2,
    limit: int = 20,
)
```

The application, not the recorder, controls
`torch.cuda.memory._record_memory_history()`.

- No history: allocator state and snapshot-inferred lifecycle analysis remain
  available; allocation stacks and historical events are generally unavailable.
- `enabled="state", context="state", stacks="python"`: live block allocation
  stacks are available.
- `enabled="all", context="all", stacks="python"`: live block stacks and
  historical `device_traces` events are available.

`stacks=True` groups active bytes by pool and allocation call stack, with a
separate by-stream detail. Blocks without frames remain in `<unattributed>`.
Coverage is reported for both points.

`events=True` aggregates same-run events by pool, stream, action, and event
call stack. Events begin after the earlier point's marker and end at the later
snapshot endpoint. An event's reported pool ID is preferred; otherwise address
ranges are used and unresolved events remain in `pool[unknown]`.

`lifetimes=True` embeds a top-cohort lifetime summary in same-run comparisons
and timelines. In a phase comparison it applies independently to the baseline
and candidate change ranges. Cross-run lifetime requests are rejected.

For snapshots containing traces from multiple CUDA devices, event windows are
built independently for every device that owns an endpoint segment or has a
trace entry in the interval. This preserves event-only transient allocations
on devices with no endpoint segment. Markers from one device cannot satisfy
another device's boundary.

Missing requested history adds a warning or raises `MemoryHistoryError`
according to `on_missing`.

### Allocation Cohort Lifetimes

```python
run.lifetimes(
    at: str | int | MemoryPoint | None = None,
    *,
    born_between: tuple[
        str | int | MemoryPoint,
        str | int | MemoryPoint,
    ] | None = None,
    through: str | int | MemoryPoint | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemoryAllocationLifetimeAnalysis
```

With `at` set, the report starts there and retains only allocation instances
active at that anchor. With `at=None`, it scans from the first point and keeps
all cohorts observed through `through` or the final point. The end point must
not precede the start point and must belong to the same run.

`born_between=(start, end)` instead selects generations allocated in the
half-open marker range `(start, end]`; it is mutually exclusive with `at`.
With complete allocator events, a generation that is allocated and freed
entirely between snapshots remains in the report. `peak_live_bytes` captures
its event-derived peak even when every point state is zero. With events
disabled or incomplete, visible snapshot births remain available with
`snapshot_inferred` confidence and a warning that transient allocations may be
missing.

An allocation instance is tracked by device, block address, size, requested
size, pool, and stream. A matching `free_requested` followed by `alloc` at the
same address splits the old and new generations. Event size matching accepts
either the snapshot's allocator-rounded block size or its requested size
because PyTorch traces may report the latter.

Instances are grouped into cohorts by device, pool, and allocation stack. Each
cohort retains point-by-point active/requested bytes, block counts, stream IDs,
a unique-instance size histogram, allocation births, release stacks, and
point/event peaks.

Release evidence is explicit:

- `event_exact` means a matching `free_requested` event exists in the
  marker-delimited interval;
- `snapshot_inferred` means an observed block disappeared without exact event
  evidence;
- `still_active_bytes` counts observed instances active at the end point.

The application must enable full allocator history before the allocations of
interest to obtain exact release timing and free call stacks. With events
disabled or unavailable, snapshot-inferred results remain usable.
Snapshot-only matching cannot detect a free-and-reallocate cycle that reuses
the same address and shape entirely between two points.

Calling `run.lifetimes()` without explicit options defaults to stack depth 4,
event attribution, warning on missing history, and the top 20 cohorts. Use
`MemoryAttributionOptions(events=False, ...)` for snapshot-only analysis.
Cohorts are ranked by bytes born for `born_between`, by bytes active at an
anchor, or by peak-to-minimum impact when there is no selection.

This API reports allocator-block evidence. It does not recover Python tensor
names, object identity, dtype, shape, or higher-level ownership unless those
details are inferable from the recorded call stacks and allocation sizes.

### Timeline

`run.timeline()` reports absolute state and a delta for every observed pool
and `(pool, stream)` observation at every point. The first point is relative to
zero. If a pool disappears, the current state is zero and its negative delta
remains visible. Unchanged rows are retained by default.

The default timeline is manifest-only and does not read raw snapshots;
`timeline.point_comparisons` is empty. Requesting stack or event attribution
streams each point once and stores attributed same-run point comparisons.

With `lifetimes=True`, one full-run cohort report is attached as
`timeline.allocation_lifetimes`. Adjacent comparisons do not repeat the same
lifetime scan.

### Cross-Run Comparison

```python
compare_points(
    reference: MemoryPoint,
    candidate: MemoryPoint,
    *,
    pool_mapping: Mapping[PoolId, PoolId] | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemoryPointComparison
```

The points must have different `run_id` values. Without `stacks=True`, the
comparison uses compact manifest state and never reads raw snapshots. Cross-run
matching is conservative:

1. Default `(0,0)` pools match automatically when present in both runs.
2. Private pools match only through a one-to-one `pool_mapping`; every mapped
   reference and candidate must exist at the selected points.
3. Identical raw private IDs are still unmatched without that mapping.
4. Streams are never matched across runs; every pool/stream observation is
   reference-only or candidate-only.
5. Address lifecycle is disabled.
6. `events=True` is rejected.
7. `lifetimes=True` is rejected.
8. Stack deltas are computed only for matched pools.

Every comparison separately exposes `all`, `default`, and `private` totals.
They include unmatched private pools and therefore remain complete even when
pool-level cross-run identity is intentionally unavailable.

### Phase Comparison

```python
compare_phases(
    baseline: MemoryRange,
    candidate: MemoryRange,
    *,
    pool_mapping: Mapping[PoolId, PoolId] | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemoryPhaseComparison
```

The result contains `baseline_change`, `candidate_change`, `start_gap`,
`end_gap`, one `pool_decomposition` entry for each matched pool and metric, and
one `allocator_scope_decomposition` entry for each `all`, `default`, and
`private` scope. Every row verifies:

```text
end_gap = start_gap + candidate_change - baseline_change
```

Every explicit private-pool mapping must exist at all four selected phase
points because the mapping is applied to both cross-run endpoint comparisons.
Seed a shared graph pool before the phase starts when the phase itself will grow
that pool.

Event attribution applies to the two same-run change comparisons. Start/end
cross-run comparisons never compare events.
Lifetime attribution follows the same rule: each change range contains its
cohort report, while start/end cross-run comparisons do not.

### Multi-Rank Run Groups

```python
MemoryRunGroup.load(root, *, cache_snapshots=False) -> MemoryRunGroup
MemoryRunGroup.from_runs(runs, *, root=None) -> MemoryRunGroup
group.ranks -> tuple[int, ...]
len(group) -> int
group[rank] -> MemoryRun
group.descriptor() -> dict
group.summary() -> MemoryRunGroupSummary

compare_run_group_phases(
    baseline: MemoryRunGroup,
    candidate: MemoryRunGroup,
    *,
    baseline_start: str | int,
    baseline_end: str | int,
    candidate_start: str | int,
    candidate_end: str | int,
    attribution: MemoryAttributionOptions | None = None,
) -> MemoryRunGroupPhaseComparison
```

`load()` reads direct `*.tcgd-memory` child directories. A group requires
non-null unique ranks, one run name, one ordered point-label sequence, and no
conflicting non-null group IDs or world sizes. Declared-but-missing ranks,
missing identity fields, incomplete bundles, runtime provenance differences,
and user metadata differences are warnings. The default does not retain
decompressed raw snapshots across ranks or points; set
`cache_snapshots=True` only for workloads that repeatedly inspect the same
raw payloads.

`MemoryRunGroupSummary` emits per-rank point/scope states plus min, max, spread,
and worst rank. `MemoryRunGroupPhaseComparison` pairs common ranks and
aggregates the per-rank four-point equations. Neither API sums GPU memory across
ranks.

### Result Objects

`MemorySnapshotComparison`, `MemoryPointComparison`, `MemoryTimeline`,
`MemoryPhaseComparison`, `MemoryAllocationLifetimeAnalysis`,
`MemoryRunGroupSummary`, and `MemoryRunGroupPhaseComparison` provide rendering
and serialization. Their main programmatic fields are:

- `MemorySnapshotComparison` and `MemoryPointComparison`: sibling result types
  with `reference`, `candidate`, `allocator_scope_comparisons`,
  `pool_comparisons`, `observation_comparisons`, optional attribution fields, and
  `warnings`.
- `MemoryTimeline`: `run`, `allocator_scope_entries`, `pool_entries`,
  `observation_entries`, optional `point_comparisons` and allocation lifetimes,
  and derived `warnings`.
- `MemoryPhaseComparison`: `baseline_change`, `candidate_change`, `start_gap`,
  `end_gap`, `allocator_scope_decomposition`, `pool_decomposition`, and `warnings`.
- `MemoryAllocationLifetimeAnalysis`: `source_kind`, `source_id`, `source_name`,
  selection states, `cohorts`, history coverage, attributed bytes, and `warnings`.
- `MemoryRunGroupSummary`: `run_group`, per-rank `rank_point_entries`,
  cross-rank `point_aggregates`, and `warnings`.
- `MemoryRunGroupPhaseComparison`: `baseline_group`, `candidate_group`,
  `rank_comparisons`, `rank_decomposition`, `phase_aggregates`, and `warnings`.

Rendering and serialization use:

```python
result.to_text(include_unchanged=True) -> str
result.to_dict() -> dict
result.to_html(include_unchanged=True) -> str
result.write(output_dir, include_unchanged=True) -> dict[str, Path]
```

`include_unchanged=False` filters zero-change rows from text, HTML, and CSV.
`to_dict()` and `report.json` always retain complete data. Timeline and phase
objects aggregate warnings from their component comparisons at the top level.

Comparison data uses nested absolute and delta state:

```json
{
  "reference": {"allocated_bytes": 1024},
  "candidate": {"allocated_bytes": 2048},
  "delta": {"allocated_bytes": 1024}
}
```

CSV flattens these as `reference_allocated_bytes`,
`candidate_allocated_bytes`, and `delta_allocated_bytes`.

Every `write()` creates `report.txt`, `report.json`, and `report.html`.
Pool-oriented results also create `allocator_scopes.csv`, `pools.csv`, and
`observations.csv`, with optional `allocation_stack_comparisons.csv`,
`events.csv`, `pool_decomposition.csv`, and
`allocator_scope_decomposition.csv`.

`MemoryAllocationLifetimeAnalysis` and pool-oriented results that embed one
create `cohorts.csv`, `cohort_points.csv`, and `size_histograms.csv`. They add
`birth_stacks.csv` or `release_stacks.csv` when those observations exist.
Lifetime JSON keeps point states, histograms, birth/release confidence,
point/event peaks, and exact/inferred/still-active totals nested under each
cohort.

Group summaries create `rank_point_entries.csv` and `point_aggregates.csv`.
Group phase reports create `rank_decomposition.csv` and
`phase_aggregates.csv`.

### Bundle Format

The bundle schema is `torch-cudagraph-debug/memory-run`. A bundle has one
plain JSON manifest and `snapshots/NNNN.json.gz` files. Writes use a temporary
file followed by atomic replacement. Snapshot paths are validated to remain
inside the bundle.

Manifest, point, and observation objects have canonical required fields.
Observation rows store only the seven base `MemoryStats` fields; inactive and
fragmentation values are derived after loading. Missing or unknown fields are
rejected.

Loading reads only manifest summaries and never executes pickle. A bundle has
one writer; distributed users create one bundle per rank.

### CLI

The [CLI workflow example](../examples/memory_debug/cli/workflows.sh) creates
its own bundles and exercises every command below.

```text
tcgd-memory summary BUNDLE
tcgd-memory allocation-lifetimes BUNDLE \
  [--at POINT | --born-between START END] [--through POINT] \
  [--no-events] --output DIR
tcgd-memory timeline BUNDLE --output DIR
tcgd-memory compare-points REFERENCE_BUNDLE [CANDIDATE_BUNDLE] \
  --reference-point POINT --candidate-point POINT --output DIR
tcgd-memory compare-phases BASELINE_BUNDLE CANDIDATE_BUNDLE \
  --baseline-start POINT --baseline-end POINT \
  --candidate-start POINT --candidate-end POINT \
  [--pool-map A=B] --output DIR
tcgd-memory summarize-run-group GROUP_DIR --output DIR
tcgd-memory compare-run-group-phases BASELINE_GROUP CANDIDATE_GROUP \
  --baseline-start POINT --baseline-end POINT \
  --candidate-start POINT --candidate-end POINT --output DIR
```

Omitting `CANDIDATE_BUNDLE` from `compare-points` compares two ordered points
in the reference run; `--pool-map` is valid only across independent runs.

`timeline`, `compare-points`, `compare-phases`, and
`compare-run-group-phases` accept the common attribution and presentation
options: `--stacks`, `--events`, `--lifetimes`,
`--on-missing {warn,error}`, `--stack-depth`, `--limit`, and `--only-changed`.
`allocation-lifetimes` accepts the same options, always groups by allocation
stack, enables events by default, and uses stack depth 4; `--no-events`
requests snapshot-only inference. `summary` accepts one bundle and writes to
standard output. `summarize-run-group` accepts one group directory plus
`--output`, without attribution options. Cross-run event and lifetime requests
are rejected. Pool IDs use comma-separated components, for example
`--pool-map 0,1=0,4`.

## Experimental Advanced API

```python
from torch_cudagraph_debug.memory_debug import advanced
```

The advanced facade is intended for custom snapshot analysis and may change in
a minor release. Use the high-level facade above for normal Recorder, Run,
Point, Observation, and comparison workflows. Advanced exports are grouped
below.

Snapshot and state types:

```python
advanced.AllocatorSnapshotData
advanced.MemoryObservationKey
advanced.MemoryStats
advanced.AllocatorTraceEntry
```

Snapshot normalization and aggregation:

```python
advanced.normalize_pool_id(pool_id)
advanced.normalize_snapshot(snapshot)
advanced.normalize_trace_entries(snapshot)
advanced.summarize_snapshot(snapshot)
advanced.summarize_segments(segments)
advanced.summarize_pools(observations)
```

`normalize_snapshot()` accepts either the full mapping returned by
`torch.cuda.memory._snapshot()` or a segment sequence. `summarize_snapshot()`
and `summarize_segments()` return `Mapping[MemoryObservationKey, MemoryStats]`;
`summarize_pools()` removes the stream dimension.

Allocation-stack types and helpers:

```python
advanced.AllocationStackCoverage
advanced.AllocationStackSummary
advanced.AllocationStackDelta
advanced.allocation_stack_coverage(snapshot, *, pool_id=None)
advanced.summarize_allocation_stacks(
    snapshot,
    *,
    pool_id=None,
    stream=None,
    stack_depth=2,
    by_stream=False,
    top=None,
)
advanced.compare_allocation_stacks(
    reference,
    candidate,
    *,
    pool_id=None,
    stream=None,
    stack_depth=2,
    by_stream=False,
    top=None,
    include_unchanged=False,
)
```

Allocator-event types and helpers:

```python
advanced.EventWindow
advanced.AllocatorEventSummary
advanced.extract_event_window(
    entries,
    *,
    start_marker,
    end_marker,
    start_label,
)
advanced.summarize_allocator_events(
    entries,
    *,
    reference_segments,
    candidate_segments,
    stack_depth=2,
    top=20,
)
```

Identity, stack-key, and formatting helpers:

```python
advanced.pool_id_label(pool_id)
advanced.stream_label(stream)
advanced.stack_key_from_frames(frames, *, depth=2)
advanced.format_bytes(value)
advanced.format_delta_bytes(value)
advanced.format_comparison(reference, candidate, delta)
```

## Errors

- `CudaGraphDebugError`: package-wide base.
- `NativeExtensionUnavailableError`: compiled tensor extension cannot be loaded.
- `TensorDebugError`: tensor domain base.
- `TensorCheckError`: online `CheckAction` mismatch reported by `assert_check_ok()`.
- `TensorComparisonError`: failed offline comparison assertion.
- `TensorBundleError`: malformed, unsupported, or unreadable tensor bundle.
- `TensorOwnershipError`: point used with a run that does not own it.
- `TensorPayloadUnavailableError`: full values requested from a summary observation.
- `MemoryDebugError`: memory domain base.
- `MemoryHistoryError`: requested history unavailable or incomplete.
- `MemoryBundleError`: malformed, unsupported, or unreadable bundle.
- `MemoryOwnershipError`: point used with a run that does not own it.
