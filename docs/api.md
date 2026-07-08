# torch-cudagraph-debug API Reference

This reference describes the current public API.

## Package Root

```python
from torch_cudagraph_debug import (
    CudaGraphDebugError,
    FrozenJSONValue,
    JSONScalar,
    JSONValue,
    NativeExtensionUnavailableError,
    __version__,
)
```

The package root contains version information, common errors, and shared JSON
metadata types. Import each debugging domain explicitly.

## Public Type Index

The domain facades are the compatibility boundary. The lists below include
workflow objects, immutable data, policy aliases, result leaves, and analysis
functions. Experimental raw parsing remains under `memory_debug.advanced`.

Tensor facade:

- Workflows and data: `TensorProbe`, `TensorProbeSnapshot`,
  `TensorRecorder`, `TensorRun`, `TensorPoint`, `TensorObservation`,
  `TensorObservationKey`, `TensorValueSummary`, and `TensorCheckStatus`.
- Actions and input types: `PrintAction`, `RecordAction`, `CheckAction`,
  `TensorExpected`, and `TensorExpectedValue`.
- Policies: `ComparisonMode`, `ComparisonStatus`, `DTypePolicy`,
  `LayoutPolicy`, `ObservationComparisonKind`, and
  `TensorComparisonOptions`.
- Results: `TensorObservationComparison`, `TensorSnapshotComparison`,
  `TensorPointComparison`, `TensorRunComparison`,
  `TensorPointSeriesComparison`, `TensorRankPointSummary`,
  `TensorRankRunComparison`, `TensorRunGroup`,
  `TensorRunGroupSummary`, and `TensorRunGroupComparison`.
- Functions: `compare_snapshots`, `compare_points`, `compare_runs`,
  `compare_point_series`, and `compare_run_groups`.
- Errors: `TensorDebugError`, `TensorCheckError`,
  `TensorComparisonError`, `TensorBundleError`, `TensorOwnershipError`,
  and `TensorPayloadUnavailableError`.

Memory facade:

- Workflows and data: `MemoryProbe`, `MemoryProbeSnapshot`,
  `MemoryRecorder`, `MemoryRun`, `MemoryPoint`, `MemoryRange`,
  `MemoryObservation`, `MemoryObservationKey`, `MemoryPoolKey`,
  `MemoryLifetimeSelection`, `PoolId`, and `StreamId`.
- State and policies: `AllocatorScope`, `MemoryStats`,
  `MemoryStatsDelta`, `MemoryStatMetric`, `MemoryDisplayOptions`,
  `MemoryAttributionOptions`, `MemoryLifetimeOptions`,
  `MemoryEvidenceStatus`, `MemoryAttributionStatus`,
  `MatchKind`, and `PhaseMetric`.
- Comparison and timeline leaves: `MemoryLifecycleDelta`,
  `MemoryAllocatorScopeComparison`, `MemoryPoolComparison`,
  `MemoryObservationComparison`, `MemoryAllocatorScopeTimelineEntry`,
  `MemoryPoolTimelineEntry`, `MemoryObservationTimelineEntry`,
  `MemoryPhaseComponents`, `MemoryAllocatorScopePhaseDecomposition`, and
  `MemoryPoolPhaseDecomposition`.
- Attribution leaves: `AllocationStackCoverage`,
  `AllocationStackSummary`, `AllocationStackDelta`,
  `AllocatorEventSummary`, `AllocationCohort`, `CohortBirth`,
  `CohortFreeRequest`, `CohortFreeCompletion`, `CohortPointState`,
  `CohortSizeBucket`, and `CohortSizeOutcome`.
- Results and groups: `MemorySnapshotComparison`, `MemoryPointComparison`,
  `MemoryTimeline`, `MemoryPhaseComparison`,
  `MemoryAllocationLifetimeAnalysis`, `MemoryRunGroup`,
  `MemoryRunGroupSummary`, `MemoryRunGroupPhaseComparison`,
  `MemoryRankPointState`, `MemoryRankPointAggregate`,
  `MemoryRankPhaseDecomposition`, `MemoryRankPoolPhaseDecomposition`,
  `MemoryMetricExtrema`, and `MemoryRunGroupPhaseAggregate`.
- Functions: `compare_snapshots`, `compare_points`, `compare_phases`, and
  `compare_run_group_phases`.
- Errors: `MemoryDebugError`, `MemoryHistoryError`,
  `MemoryHistoryDisabledError`, `MemoryHistoryBoundaryError`,
  `MemoryHistoryTruncatedError`,
  `MemoryReconciliationError`, `MemoryBundleError`, and
  `MemoryOwnershipError`.

## Tensor Debug

Concept guide: [Tensor Debug guide](tensor_debug.md). Runnable guide:
[Tensor Debug examples](../examples/tensor_debug/README.md).

### Public Facade

The complete public surface is listed in the [Public Type Index](#public-type-index).
Typical workflow imports are:

```python
from torch_cudagraph_debug.tensor_debug import (
    CheckAction,
    RecordAction,
    TensorComparisonOptions,
    TensorObservationKey,
    TensorProbe,
    TensorRecorder,
    TensorRun,
    TensorRunGroup,
    compare_point_series,
    compare_points,
    compare_run_groups,
    compare_runs,
    compare_snapshots,
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
    synchronize: bool | torch.cuda.Stream | torch.device = True,
)
```

`name` and `actions` must be nonempty. Disabled actions are filtered before
native probe creation. If every action has `enabled=False`, the probe is a
pure no-op: it does not load the native extension, initialize CUDA, or allocate
a replay counter.

Every enabled probe owns a zero-dimensional CUDA `int64` replay counter on
`device`, or on the current CUDA device when `device=None`. An integer device
index is accepted. Every active source tensor must be on the same device.

`probe(tensor, name=...)` returns the exact input tensor object. The optional
name identifies the logical observation; omitting it uses `probe.name`. In
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
calls in that capture create globally ordered slots. Observation identity is
`(name, invocation_index)`, where the index starts at zero independently for
each name; `order` preserves capture-call order. Reusing the
probe in another capture is an error. The GPU counter advances once per graph
replay, not once per invocation, so all slots in one replay share a 1-based
replay index. Eager `when="always"` calls do not advance the counter and
report replay index 0. Eager observations own one slot per observation name:
a repeated name samples in place (latest value) with `invocation_index=0`,
while each new name appends a slot. Positional `CheckAction` entries bind
eager slots in first-use name order. Observation names are a fixed
vocabulary — generating a fresh name per iteration grows a slot each time.
Validation and host-side slot preparation failures do not consume a
name's first-use order; a failed replacement also preserves the previous
eager value. The probe's one capture then establishes its own slot
layout; eager observations recorded before the capture are dropped. Their
staging is retired immediately and reclaimed only after previously queued eager
work completes.

The same-name repeat therefore means two different things by mode, and the
difference is disclosed rather than silent: within one capture it creates a
new invocation (a distinct position in that execution), while an eager repeat
re-samples in place. Every in-place re-sample increments a per-name counter
exposed as `TensorProbeSnapshot.eager_overwrites` (pairs) and
`eager_overwrite_counts` (mapping); snapshot comparisons emit a warning when
an eager sample with a nonzero count is aligned against multi-invocation
observations from a capture, because the sample keeps only the latest
occurrence. The warning identifies which side requires complete collection;
record that side with `TensorRecorder` when invocation alignment matters.
Overwrite metadata is valid only on an eager snapshot (`replay_index=0`);
names are unique, and each entry must refer to that name's sole invocation-0
observation in the snapshot.

#### Methods

```python
probe(tensor: torch.Tensor, *, name: str | None = None) -> torch.Tensor
probe.replay_index -> torch.Tensor | None
probe.snapshot(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device | None = None,
) -> TensorProbeSnapshot
probe.compare(
    reference: TensorProbeSnapshot,
    candidate: TensorProbeSnapshot,
    *,
    options: TensorComparisonOptions | None = None,
) -> TensorSnapshotComparison
probe.check_status(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device | None = None,
) -> TensorCheckStatus
probe.assert_check_ok(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device | None = None,
) -> None
probe.watch_grad(
    tensor: torch.Tensor,
    *,
    name: str | None = None,
    strict: bool = False,
) -> torch.utils.hooks.RemovableHandle | None
probe.close(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device | None = None,
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
match the probe device. Omitting a method-level override or passing `None`
inherits the probe's configured policy. Explicit
strings, integer device indices, and CPU devices are rejected.
Synchronization-enabled queries are invalid during CUDA Graph capture; for an
enabled probe, `close()` is invalid during capture for every synchronization
setting.

`snapshot()` returns one aggregate `TensorProbeSnapshot` containing the
latest value in the probe's current slot layout: one slot per eager observation
name before capture, or one slot per capture-time invocation after capture.
Capture installs copy nodes but does not execute them. A Record-only probe reads
its GPU counter at this query point. When Print or Check is also enabled, the
query reuses their existing pinned-host counter staging instead of issuing
another counter transfer. It raises `TensorDebugError` when no `RecordAction`
is enabled or no slot has recorded a value.

`compare()` accepts two snapshots owned by this probe in chronological replay
order and returns the same result type as top-level `compare_snapshots()`.

`check_status()` returns:

```python
@dataclass(frozen=True)
class TensorCheckStatus:
    ok: bool
    message: str
    replay_index: int
    order: int
    name: str | None
    invocation_index: int

    @property
    def key(self) -> TensorObservationKey | None: ...
```

A successful status uses `order=-1`, `name=None`, and `invocation_index=-1`.
A mismatch reports both the semantic `key` and its global capture `order`.

`check_status()` synchronizes callback-backed check state before returning it.
`assert_check_ok()` delegates to `check_status()` and therefore synchronizes at
most once; it raises `TensorCheckError` when the sticky check state is not OK.

`watch_grad()` registers an autograd hook. Its optional `name` follows the
same default as `probe()`. The hook probes the gradient for its side effect and
returns the original gradient. Invocation indices are assigned when hooks
actually fire, so repeated same-name hooks follow backward execution order.
It obeys the probe's `when`
policy, so default capture-only probes do no work during eager backward. The
method returns a removable hook handle. If `tensor.requires_grad` is false, it
returns `None`; with `strict=True`, it raises `RuntimeError`. Every registered
hook is removed by `close()`, so a backward pass after close never fires a
hook against the closed probe; removing a handle earlier yourself remains
safe.

`close()` first performs the requested synchronization, reclaims retired host
staging, and releases native resources. An unsynchronized close verifies
pending eager work and fails while an eager host callback or eager device
copy is provably still in flight. Replays cannot be verified from inside the
probe: after capture, closing asserts that no replay is in flight and that no
graph containing the probe can replay again. A current or later replay would
access freed staging, replay-counter, and callback resources. Passing
`synchronize=False` additionally asserts that required synchronization already
occurred. Host-side enqueue preparation is transactional: a rejected call does
not consume its slot or capture order. If CUDA command submission itself fails
after the transaction is published, the probe becomes unusable for further
collection or queries; destroy any affected graph, close the probe, and create a
new one. `TensorProbe` is also a context manager whose exit uses the
correctness-first default close; use that form only when every replay occurs
inside the context.

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
    expected: (
        TensorExpectedValue
        | Sequence[TensorExpectedValue]
        | Mapping[TensorObservationKey, TensorExpectedValue]
    ),
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
and `every=N` prints every captured slot on graph replay indices divisible by N.

`RecordAction` enqueues a device-to-host tensor copy into pinned staging
memory without a host callback when used alone. The probe stores only the latest
staged value for each capture slot. Its counter remains on the GPU during
replay and transfers only when `snapshot()` explicitly queries it. When Record
is combined with Print or Check, that query reuses the callback actions'
single shared 8-byte counter transfer.

`CheckAction.expected` values must be on CPU and match the captured source's
shape and dtype. A single tensor or array is normalized to a one-item expected
sequence and therefore applies only to global capture order 0. It is never
broadcast. For a probe called multiple times in one capture, pass one expected
value per slot in capture-call order.
A `Mapping[TensorObservationKey, TensorExpectedValue]` instead binds expected
values by semantic name and local invocation rather than global order.

A check mismatch is sticky until the probe is destroyed. The first mismatch
records the replay index, semantic observation key, and global order.

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
    snapshot_index: int
    replay_index: int
    timestamp: float
    observations: tuple[TensorObservation, ...]

snapshot.by_key -> Mapping[TensorObservationKey, TensorObservation]
snapshot.observation(name: str | None = None, invocation_index: int = 0) -> TensorObservation
snapshot.tensor(name: str | None = None, invocation_index: int = 0) -> torch.Tensor
snapshot.descriptor() -> dict
```

Every `snapshot()` call materializes new CPU tensor copies from the latest
pinned staging values and records the current GPU replay counter.
`snapshot_index` is Probe-local query order; `replay_index` identifies the
replay that supplied the values, so multiple queries can share one replay
index. Observations follow global capture order and use
`(name, invocation_index)` as stable identity, with invocation indices assigned
independently per name.

Omitting `name` from `observation()` or `tensor()` uses `snapshot.probe_name`
and invocation zero. Returned tensors are independent copies, and retained
snapshots do not change after later replays. A query before the first replay,
or an eager `when="always"` query, uses replay index zero.

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
    strict_scope: bool = False,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
    rank: int | None = None,
    group_id: str | None = None,
    world_size: int | None = None,
    run_metadata: Mapping[str, Any] | None = None,
)

recorder.observe(tensor, *, name, payload=None) -> torch.Tensor
recorder.watch_grad(
    tensor,
    *,
    name,
    payload=None,
    strict=False,
) -> RemovableHandle | None
recorder.record_point(
    label,
    *,
    metadata=None,
    synchronize=None,
) -> ContextManager[None]
recorder.preview() -> TensorRun
recorder.finish() -> TensorRun
recorder.result -> TensorRun
recorder.close(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device | None = None,
) -> None
```

`execution` is explicit because eager execution and graph replay have
different collection boundaries. Eager `observe()` calls are active only
inside `record_point()`. A CUDA Graph recorder uses `observe()` calls during its
one capture session to define slots and reads their latest values when
`record_point()` later wraps a replay. Calls outside those active paths are
transparent no-ops by default; `strict_scope=True` turns them into errors.
`preview()` returns an immutable nonterminal view without finishing.

All observations must be supported CUDA tensors on the recorder device.
`non_contiguous` has the same error/copy behavior as `TensorProbe`. One
recorder-wide native `RecordAction` session collects every CUDA Graph observation,
so logical names do not create one replay counter kernel each.

`record_point()` labels must be nonempty and unique, and point contexts cannot
nest. Its synchronization target inherits the recorder default when omitted.
Interrupted contexts do not append a point.

`finish()` is idempotent, writes `complete=True`, and rejects later points. It
does not release captured native storage. `close()` releases that storage after
the requested bool/stream/device synchronization, removes every gradient hook
registered through `watch_grad()`, and requires that the graph can no longer
replay. A normal context-manager exit calls `finish()` and then
`close()` with the recorder's configured synchronization target. If the block
raises, the recorder instead freezes and persists its partial result with
`complete=False`, blocks later collection, closes resources, and does not
suppress the application exception. `result` and `preview()` expose that
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

```python
@dataclass(frozen=True)
class TensorObservationKey:
    name: str
    invocation_index: int
```

A `TensorPoint` owns a point-level optional `replay_index` and ordered
`TensorObservation` objects.
`point.observation(name, invocation_index=0)` performs stable-key lookup.
A foreign point passed to `run.point()` raises `TensorOwnershipError`.

`TensorObservation` is an ownerless leaf shared by Probe snapshots and Recorder
points. It records global order, name, per-name invocation, shape, stride,
dtype, source device, payload kind, nbytes, SHA-256, and a `TensorValueSummary`.
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
    layout_policy: Literal["strict", "ignore"] = "strict",
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
do not contain a synthetic `point_comparison`. Serialized point and snapshot
comparisons preserve every `TensorComparisonOptions` field above, including
`layout_policy`, under `options`. Run and point-series reports retain those
complete comparison objects under `point_comparisons`.

The stable observation key is `(name, invocation_index)`.
`compare_points()` follows reference order and appends candidate-only keys.
Missing keys, shape changes, strict dtype changes, and source-stride changes
under the default `layout_policy="strict"` are mismatches. Use
`layout_policy="ignore"` only when layout differences are intentional.

`compare_runs()` aligns points by identical label. Pass `point_mapping` when
semantically equivalent point labels differ; explicit entries take precedence
and unmapped labels still align by identical label. A candidate label may be
claimed only once across the merged mapping, and only labels covered by
neither the mapping nor auto-alignment are reported as one-sided points.

Allclose uses the reference tensor in
`atol + rtol * abs(reference)`. Integer and bool values compare exactly.
`dtype_policy="promote"` explicitly converts both values with
`torch.promote_types()`. Exact comparison uses raw value bytes when dtypes
match; with `equal_nan=True`, positions where both sides are NaN are exempted
while every other position keeps the bitwise distinction, including signed
zeros and NaN payload bits. Across differing dtypes under
`dtype_policy="promote"`, exact comparison uses value equality on the promoted
values with `equal_nan` honored, so bit distinctions that promotion erases
(such as signed zeros and NaN payload bits) are not detected there.

`TensorObservationComparison` contains status, reason, both observation
descriptors, mismatch count and fraction, max absolute and relative error, mean
absolute error, and the first mismatching coordinate and values when full
payloads make those metrics available.
Integer diagnostics subtract integer values without floating-point conversion.
Mismatch values and maximum absolute error remain exact across the full `int64`
range; mean and relative errors remain floating-point metrics.

`TensorSnapshotComparison`, `TensorPointComparison`, `TensorRunComparison`,
`TensorPointSeriesComparison`, and `TensorRunGroupComparison` expose `status`,
`ok`, `conclusive`, `assert_ok()`, `to_text()`, `to_dict()`, `to_html()`, and
`write()`. Point reports additionally provide `first_issue` and
`worst_observation_comparisons()`. Text, HTML, and CSV include matches by
default and accept `include_unchanged=False`; JSON remains complete. Report
files are written atomically. `write()` rejects a nonempty directory unless
`overwrite=True`.
When overwriting, known tcgd report artifacts from the previous write are
removed; unrelated files in the directory are preserved.


Summary comparison uses three states. Equal digests match, except under
allclose with `equal_nan=False` when the summary reports NaNs: bit-identical
NaN positions still fail allclose and are reported as mismatches. Different
digests are a mismatch in exact mode, but are inconclusive in allclose mode
when either full payload is unavailable.

### Tensor Run Groups

```python
TensorRunGroup.load(root, *, cache_tensors=False) -> TensorRunGroup
TensorRunGroup.from_runs(runs, *, root=None) -> TensorRunGroup
group.ranks -> tuple[int, ...]
group.missing_ranks -> tuple[int, ...]
group.complete -> bool
group.summary() -> TensorRunGroupSummary
compare_run_groups(
    reference,
    candidate,
    *,
    point_mapping=None,
    options=None,
) -> TensorRunGroupComparison
```

A group loads direct child bundles and requires unique non-null ranks, one run
name, one execution mode, and one point-label sequence. An incomplete rank
whose labels are a strict prefix of the longest rank sequence (a crashed rank)
is accepted with the incomplete-bundle warning; a complete rank with fewer
points or any non-prefix sequence is an error. Conflicting non-null
group IDs or world sizes are errors; missing identity, missing declared ranks,
incomplete bundles, provenance differences, and metadata differences are
warnings. `TensorRankPointSummary` describes rank/point payload inventory, and
`TensorRankRunComparison` owns one rank's run comparison. A comparison with
missing ranks, unknown world size, or incomplete bundles is `inconclusive`, not
`match`. Group summary and comparison results provide `to_text()`, `to_dict()`,
`to_html()`, and `write()`; comparisons also provide `assert_ok()`. Group
summaries write `rank_points.csv`; group comparisons write
`rank_comparisons.csv`.

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
tcgd-tensor compare-runs REFERENCE_BUNDLE CANDIDATE_BUNDLE \
  [--point-map REFERENCE=CANDIDATE]
tcgd-tensor group-summary GROUP_DIR
tcgd-tensor compare-run-groups REFERENCE_GROUP CANDIDATE_GROUP \
  [--point-map REFERENCE=CANDIDATE]
tcgd-tensor compare-point-series REFERENCE_BUNDLE [CANDIDATE_BUNDLE] \
  --reference-point LABEL
```

For `compare-points` and `compare-point-series`, omitting `CANDIDATE_BUNDLE`
reuses the reference bundle.

Comparison commands accept `--mode`, `--rtol`, `--atol`, `--equal-nan`,
`--promote-dtypes`, `--ignore-layout`, `--limit`, `--only-changed`, and optional
`--output`. Reusing a nonempty report directory requires `--overwrite`.
Exit status is zero for a match, one for a mismatch or inconclusive result, and
two for invalid input or an operational error.

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

### Public Facade

The complete public surface is listed in the [Public Type Index](#public-type-index).
Typical workflow imports are:

```python
from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryLifetimeSelection,
    MemoryPoolKey,
    MemoryProbe,
    MemoryRecorder,
    MemoryRun,
    MemoryRunGroup,
    compare_phases,
    compare_points,
    compare_run_group_phases,
    compare_snapshots,
)
```

Low-level raw snapshot parsers remain experimental under
`torch_cudagraph_debug.memory_debug.advanced`.

### MemoryProbe

```python
MemoryProbe(
    name: str = "memory",
    *,
    devices: torch.device | str | int | Sequence[torch.device | str | int]
        | Literal["all"] | None = None,
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
    attribution: MemoryAttributionOptions | None = None,
) -> MemorySnapshotComparison
```
`MemoryProbe` discovers every pool and stream on selected devices; users never
pass pool handles. `devices=None` binds the current device lazily on the first
snapshot, an explicit device or sequence selects specific devices, and `"all"`
selects every visible device. Collection uses
`torch.cuda.memory._snapshot()`. `snapshot()` inherits the Probe synchronization
policy unless an override is supplied.

```python
@dataclass(frozen=True)
class MemoryProbeSnapshot:
    probe_id: str
    probe_name: str
    snapshot_index: int
    timestamp: float
    boundary_marker: str
    observations: tuple[MemoryObservation, ...]
    warnings: tuple[str, ...]

snapshot.by_key -> Mapping[MemoryObservationKey, MemoryObservation]
snapshot.observation_stats -> Mapping[MemoryObservationKey, MemoryStats]
snapshot.pool_stats -> Mapping[MemoryPoolKey, MemoryStats]
snapshot.allocator_scope_stats -> Mapping[AllocatorScope, MemoryStats]
snapshot.observation(device_index, pool_id, stream) -> MemoryObservation
snapshot.allocator_settings -> Mapping[str, FrozenJSONValue]
snapshot.allocator_state() -> Mapping[str, FrozenJSONValue]
snapshot.raw_snapshot() -> FrozenJSONValue
snapshot.descriptor() -> dict
```

`snapshot_index` is Probe-local query order. Device index is part of every
pool and observation key, so identical pool IDs on different devices remain
separate. Same-Probe comparison can use marker-delimited history; independent
probes support state and stacks but reject events and lifetimes.

`allocator_state()` returns the point-in-time allocator envelope without cumulative
`device_traces`; `raw_snapshot()` remains available on Probe snapshots for
custom trace inspection. Both views are recursively immutable. Use
`memory_debug.advanced.mutable_snapshot(snapshot)` for an editable copy.
Probe snapshots are not persisted; use Recorder for named points and bundles.

### MemoryRecorder

```python
MemoryRecorder(
    *,
    name: str = "run",
    bundle_dir: str | pathlib.Path | None = None,
    devices: torch.device | str | int | Sequence[torch.device | str | int]
        | Literal["all"] | None = None,
    rank: int | None = None,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
    group_id: str | None = None,
    world_size: int | None = None,
    run_metadata: Mapping[str, JSONValue] | None = None,
)
```

The recorder always collects `torch.cuda.memory._snapshot()`; it does not
accept pool handles or allocator-history configuration. Device selection follows
`MemoryProbe`. Rank and world-size defaults come from environment variables or
initialized `torch.distributed`.

When `bundle_dir` is set, the directory must be absent or empty. Construction
creates an incomplete manifest. Every `record_point()` writes one allocator-state
file and, after the first point, one event-evidence file for the preceding interval,
then updates the manifest; `finish()` marks it complete. A nonempty
target directory raises `FileExistsError`.

`True` synchronizes every selected device, a CUDA stream synchronizes only
that stream, an explicit CUDA device synchronizes that selected device, and
`False` leaves ordering to the caller. Synchronization is skipped during
current-stream capture because it is capture-illegal.

`group_id` identifies per-rank bundles from one distributed execution.
`run_metadata` is application-owned JSON metadata for source revision,
scenario, command, or other reproducibility fields. The recorder also captures
package/Python/platform/PyTorch/CUDA provenance automatically. Initialized GPU
properties are added under a `devices` list only after a real snapshot.

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

The recorder temporarily sets a PyTorch allocator metadata marker around each
snapshot when those private APIs are available. Each ending point stores only the
raw event entries between the previous boundary and itself. Marker failures and
history gaps are recorded per device and become typed errors only when an analysis
requests the affected event evidence.

Snapshot and metadata values must be JSON-compatible: null, string, bool,
finite number, list/tuple, or a mapping with string keys. Validation errors
include the path to the unsupported value.

#### Lifecycle

```python
recorder.preview() -> MemoryRun
recorder.finish() -> MemoryRun
recorder.result -> MemoryRun
```

`preview()` returns an immutable incomplete view without stopping
collection. `finish()` is idempotent, makes later `record_point()` calls
invalid, writes `complete=True`, and returns an immutable run. On an exception,
context-manager exit instead freezes the collected points, writes a terminal
`complete=False` manifest, makes later collection invalid, and does not
suppress the application exception. `result` is available after either normal
or exceptional context exit; `preview()` and a later idempotent `finish()`
return that same terminal result.

For tests only, `MemoryRecorder._from_snapshot_provider(...)` injects synthetic
snapshots without exposing a provider in the public constructor.

### MemoryPoint

A point owns ordered device/pool/stream observations and one recursively
immutable allocator-state view. Event evidence from the previous point to this
point is an internal, lazily loaded analysis input rather than a second public
container.

```python
@dataclass(frozen=True)
class MemoryObservation:
    order: int
    device_index: int
    pool_id: PoolId
    stream: StreamId
    stats: MemoryStats

observation.key -> MemoryObservationKey
point.observation(device_index, pool_id, stream) -> MemoryObservation
point.observation_stats -> Mapping[MemoryObservationKey, MemoryStats]
point.pool_stats -> Mapping[MemoryPoolKey, MemoryStats]
point.allocator_scope_stats -> Mapping[AllocatorScope, MemoryStats]
point.allocator_settings -> Mapping[str, FrozenJSONValue]
point.allocator_state() -> Mapping[str, FrozenJSONValue]
point.descriptor() -> dict
```

`MemoryStats` stores reserved, allocated, active, requested, segment count,
block count, inactive block count, largest inactive block, expandable segment
count, and expandable reserved bytes. Three byte metrics are derived:
`awaiting_free_bytes = active_bytes - allocated_bytes`,
`inactive_bytes = reserved_bytes - active_bytes`, and
`internal_fragmentation_bytes = active_bytes - requested_bytes`. Allocator states
and nested metadata are recursively frozen; `advanced.mutable_snapshot(point)`
returns an editable deep copy.

### MemoryRun

```python
MemoryRun.load(bundle_dir, *, cache_snapshots=True) -> MemoryRun
run.descriptor() -> dict
run.validate_payloads() -> None
run.point(label_or_index_or_point) -> MemoryPoint
run[label_or_index] -> MemoryPoint
run.between(start, end) -> MemoryRange
run.compare(reference, candidate, *, attribution=None) -> MemoryPointComparison
run.timeline(*, attribution=None) -> MemoryTimeline
run.lifetimes(
    selection: MemoryLifetimeSelection | None = None,
    *,
    through=None,
    options: MemoryLifetimeOptions | None = None,
) -> MemoryAllocationLifetimeAnalysis
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
- bytes that became inactive and allocator-reusable.
`lifecycle_available` indicates whether address lifecycle was computed.
`lifecycle_confidence` is `exact` when both states retain every segment and
active-block address, `approximate` when missing addresses require size-based
multiset matching, and `unavailable` for independent states. Approximate results
remain visible but carry a warning because equal-size allocation churn can
cancel without stable addresses.

Allocator events and allocation lifetimes are separate optional attribution
results controlled by `MemoryAttributionOptions`.

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

### Memory Attribution And Display Options

```python
MemoryDisplayOptions(stack_depth: int = 2, limit: int = 20)

MemoryAttributionOptions(
    stacks: bool = False,
    events: bool = False,
    lifetimes: bool = False,
    display: MemoryDisplayOptions = MemoryDisplayOptions(),
)

MemoryLifetimeOptions(
    display: MemoryDisplayOptions = MemoryDisplayOptions(stack_depth=4),
)
```

The application owns `torch.cuda.memory._record_memory_history()`; this
package never enables it. State totals never require history. Stack
attribution uses live block frames:
partial frame coverage returns exact attributed rows plus an `<unattributed>`
bucket, while zero coverage for nonempty active state raises
`MemoryHistoryDisabledError`. Events and lifetime transitions require the
application to keep complete allocator event history enabled throughout every
analyzed interval. Missing event history raises `MemoryHistoryDisabledError`;
an unavailable metadata boundary raises `MemoryHistoryBoundaryError`; and a
missing start marker raises `MemoryHistoryTruncatedError` because the tool
cannot distinguish ring-buffer overwrite from history enabled late. Recorder
intervals use the ending snapshot's trace end because its own marker is normally
absent; PyTorch does not expose a signal that verifies history stayed enabled
between the points. Invalid boundary order or complete history that cannot be
reconciled with allocator state raises
`MemoryReconciliationError` unconditionally. Display limits affect only text
and HTML, never structured result tuples, JSON, or CSV.

```python
MemoryEvidenceStatus(requested: bool, available: bool, complete: bool)
MemoryAttributionStatus(
    stacks: MemoryEvidenceStatus,
    events: MemoryEvidenceStatus,
    lifetimes: MemoryEvidenceStatus,
)
```

Every comparison stores `attribution_status`, distinguishing unrequested
analysis, partial stack coverage, and complete evidence. Stack byte coverage
remains available separately. Event windows and address identity
are device-specific.

### Allocation Cohort Lifetimes

```python
MemoryLifetimeSelection.all()
MemoryLifetimeSelection.active_at(point)
MemoryLifetimeSelection.born_between(start, end)

run.lifetimes(
    selection: MemoryLifetimeSelection | None = None,
    *,
    through: str | int | MemoryPoint | None = None,
    options: MemoryLifetimeOptions | None = None,
) -> MemoryAllocationLifetimeAnalysis
```

The default selection is `all()`. `active_at(point)` retains generations
not allocator-reusable at that point. `born_between(start, end)` retains
generations allocated in `(start, end]`. `through` cannot precede the
selection anchor or born-between end. Complete event history preserves
transient generations that are active in neither endpoint snapshot; incomplete
history is rejected.

An allocation generation is tracked by device, block address, size, requested
size, pool, and stream. Its state transitions are:

```text
alloc -> free_requested -> free_completed
```

`free_requested` ends application ownership, but the block may remain
`active_awaiting_free` until prior stream work completes. `free_completed` ends
the generation because the allocator can then reuse the block. It does not imply
that the containing segment was returned to CUDA. Address reuse after completion
starts a new generation. Event size matching accepts either allocator-rounded
block size or requested size because PyTorch traces may report the latter.
Stream synchronization makes the dependency complete, but allocator bookkeeping
may remain `active_awaiting_free` until a later allocator operation polls
pending events and emits `free_completed`. A snapshot does not itself force that
poll.

Instances are grouped into cohorts by device, pool, and the complete normalized
allocation stack. `stack_depth` changes display only. Allocations without stack
frames also include block and requested sizes in their cohort identity so
unrelated unattributed sizes are not merged. Each cohort retains:

- owner-active and awaiting-free bytes/counts at every point;
- streams, a unique-generation size histogram, and size-by-terminal-state rows;
- birth, free-request, and free-completion transitions, each marked with
  an `event` or `range_boundary` origin;
- snapshot peaks plus event-derived owner-active and unreusable peaks;
- owner-active and awaiting-free terminal totals.

Full allocator history must be enabled before the allocations of interest.
Lifetime analysis rejects unavailable or truncated event windows rather than
inferring transitions from snapshot disappearance. Replay contradictions are
summarized by reason and device with a total count and up to three example
addresses, then raised as `MemoryReconciliationError`.

Calling `run.lifetimes()` without explicit options requires complete event
history and displays up to 20 cohorts with four stack frames.
The complete ordered cohort set remains available through `report.cohorts`,
`to_dict()`, `report.json`, and CSV. Cohort IDs are stable fingerprints of
identity; `display_rank` records the current report ordering.

This API reports allocator-block evidence. It does not recover Python tensor
names, object identity, dtype, shape, or higher-level ownership unless those
details are inferable from the recorded call stacks and allocation sizes.

### Timeline

`run.timeline()` reports absolute state for every allocator scope, pool, and
`(device, pool, stream)` observation at every point. The first point has
`delta=None` because no prior measurement exists. Later points compare with the
previous point; disappeared pools remain visible as zero state with a negative
delta. Unchanged rows are retained by default.

The default timeline is manifest-only and does not read allocator-state files;
`timeline.point_comparisons` is empty. Requesting stack or event attribution
loads each required point once and stores attributed same-run point
comparisons.

With `lifetimes=True`, one full-run cohort report is attached as
`timeline.allocation_lifetimes`. Adjacent comparisons do not repeat the same
lifetime scan.

### Cross-Run Comparison

```python
compare_points(
    reference: MemoryPoint,
    candidate: MemoryPoint,
    *,
    pool_mapping: Mapping[MemoryPoolKey, MemoryPoolKey] | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemoryPointComparison
```

The points must have different `run_id` values. Without `stacks=True`, the
comparison uses compact manifest state and never reads allocator-state files. Cross-run
matching is conservative:

1. Default `(0,0)` pools match automatically when present in both runs.
2. Private pools match only through a one-to-one `pool_mapping`; every mapped
   reference and candidate must exist at the selected points.
3. Identical raw private IDs are still unmatched without that mapping.
4. Stream IDs are process-local CUDA handles with no stable cross-run identity;
   every device/pool/stream observation is reference-only or candidate-only.
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
    pool_mapping: Mapping[MemoryPoolKey, MemoryPoolKey] | None = None,
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

A private-pool mapping is validated against the union of each run's phase
endpoints. The mapped pool may be absent at one endpoint; that endpoint is
represented by zero state, so phases that create or destroy a pool still retain
a valid four-point equation.

Event attribution applies to the two same-run change comparisons. Start/end
cross-run comparisons never compare events.
Lifetime attribution follows the same rule: each change range contains its
cohort report, while start/end cross-run comparisons do not.

### Multi-Rank Run Groups

```python
MemoryRunGroup.load(root, *, cache_snapshots=False) -> MemoryRunGroup
MemoryRunGroup.from_runs(runs, *, root=None) -> MemoryRunGroup
group.ranks -> tuple[int, ...]
group.missing_ranks -> tuple[int, ...]
group.complete -> bool
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
    pool_mappings: Mapping[int, Mapping[MemoryPoolKey, MemoryPoolKey]] | None = None,
    attribution: MemoryAttributionOptions | None = None,
) -> MemoryRunGroupPhaseComparison
```

`load()` reads direct `*.tcgd-memory` child directories. A group requires
non-null unique ranks, one run name, one ordered point-label sequence, and no
conflicting non-null group IDs or world sizes. Declared-but-missing ranks,
missing identity fields, incomplete bundles, runtime provenance differences,
and user metadata differences are warnings. The default does not retain
decompressed allocator-state or event payloads across ranks or points; set
`cache_snapshots=True` only for workloads that repeatedly inspect the same
payloads.

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
  `pool_comparisons`, `observation_comparisons`, `lifecycle_available`,
  `lifecycle_confidence`, optional attribution fields, and `warnings`.
- `MemoryTimeline`: `run`, `allocator_scope_entries`, `pool_entries`,
  `observation_entries`, optional `point_comparisons` and allocation lifetimes,
  and derived `warnings`.
- `MemoryPhaseComparison`: `baseline_change`, `candidate_change`, `start_gap`,
  `end_gap`, `allocator_scope_decomposition`, `pool_decomposition`, and `warnings`.
- `MemoryAllocationLifetimeAnalysis`: `source_kind`, `source_id`, `source_name`,
  selection states, `cohorts`, history coverage, attributed bytes, and `warnings`.
- `MemoryRunGroupSummary`: `run_group`, per-rank `rank_points`,
  cross-rank `point_aggregates`, and `warnings`.
- `MemoryRunGroupPhaseComparison`: `baseline_group`, `candidate_group`,
  `rank_comparisons`, `rank_decomposition`, `rank_pool_decomposition`,
  `phase_aggregates`, and `warnings`.

State comparisons, timelines, phase comparisons, and group phase comparisons
use:

```python
result.to_text(
    include_unchanged=True, limit=None, stack_depth=None
) -> str
result.to_dict() -> dict
result.to_html(
    include_unchanged=True, limit=None, stack_depth=None
) -> str
result.write(
    output_dir,
    include_unchanged=True,
    limit=None,
    stack_depth=None,
    overwrite=False,
) -> dict[str, Path]
```

`include_unchanged=False` filters zero-change allocator-state rows from
text, HTML, and CSV. Attribution rows are always complete in memory, `to_dict()`,
JSON, and CSV. `limit` and `stack_depth` restrict only their text/HTML
presentation. Timeline, phase, and group-phase reports apply the limit
independently to each interval or component and aggregate component warnings at
the top level.

Lifetime analyses accept presentation overrides without changing structured
results:

```python
result.to_text(limit=None, stack_depth=None) -> str
result.to_dict() -> dict
result.write(output_dir, limit=None, stack_depth=None, overwrite=False) -> dict[str, Path]
```
Run-group summaries have no unchanged-row filter and use `to_text()`,
`to_dict()`, `to_html()`, and `write(output_dir, overwrite=False)`.

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
create `cohorts.csv`, `cohort_points.csv`, `size_histograms.csv`, and
`size_outcomes.csv`. They add `birth_stacks.csv`, `free_request_stacks.csv`, or
`free_completion_stacks.csv` when those transitions exist. Lifetime JSON keeps
full stack identity, split point states, size outcomes, transition origins,
and point/event peaks nested under each cohort.

Group summaries create `rank_points.csv` and `point_aggregates.csv`.
Group phase reports create `rank_decomposition.csv`,
`rank_pool_decomposition.csv`, and `phase_aggregates.csv`.

### Bundle Format

The bundle schema is `torch-cudagraph-debug/memory-run` with
`format_version=1`:

```text
run.tcgd-memory/
  manifest.json
  states/0000.json.gz
  events/0000-0001.json.gz
```

Every point owns one state file. Point zero has no event file; each later point
owns the raw allocator events from the previous point through itself. State files
preserve the `_snapshot()` envelope except `device_traces`; event entries preserve
unknown allocator fields. Point manifests store the SHA-256 of every state and
event gzip payload. Every payload and manifest write uses a temporary file
followed by atomic replacement, and every path is validated to remain inside the
bundle.

Manifest, point, and observation objects have canonical required fields.
Observation rows persist the ten base `MemoryStats` fields. `awaiting_free_bytes`,
`inactive_bytes`, and `internal_fragmentation_bytes` are derived after loading.
Missing or unknown fields are rejected.

Loading reads only manifest summaries and never executes pickle. On first raw
payload access, the loader verifies its SHA-256; state access also recomputes the
device/pool/stream summaries and rejects disagreement with the manifest cache.
`run.validate_payloads()` explicitly bypasses those caches, rereads every
persisted state and event payload, and performs the same checks. Allocator state
and event evidence have separate lazy caches for ordinary access, so manifest-only
analysis never reads payload files. A bundle has one writer;
distributed users create one bundle per rank.

### CLI

The [CLI workflow example](../examples/memory_debug/cli/workflows.sh) creates
its own bundles and exercises every command below.

```text
tcgd-memory summary BUNDLE
tcgd-memory allocation-lifetimes BUNDLE \
  [--active-at POINT | --born-between START END] [--through POINT] \
  [--output DIR]
tcgd-memory timeline BUNDLE [--output DIR]
tcgd-memory compare-points REFERENCE_BUNDLE [CANDIDATE_BUNDLE] \
  --reference-point POINT --candidate-point POINT \
  [--pool-map DEVICE:POOL0,POOL1=DEVICE:POOL0,POOL1] [--output DIR]
tcgd-memory compare-phases BASELINE_BUNDLE CANDIDATE_BUNDLE \
  --baseline-start POINT --baseline-end POINT \
  --candidate-start POINT --candidate-end POINT \
  [--pool-map DEVICE:POOL0,POOL1=DEVICE:POOL0,POOL1] [--output DIR]
tcgd-memory group-summary GROUP_DIR [--output DIR]
tcgd-memory compare-run-group-phases BASELINE_GROUP CANDIDATE_GROUP \
  --baseline-start POINT --baseline-end POINT \
  --candidate-start POINT --candidate-end POINT \
  [--pool-map RANK@DEVICE:POOL0,POOL1=DEVICE:POOL0,POOL1] [--output DIR]
```

Omitting `CANDIDATE_BUNDLE` from `compare-points` compares two ordered points
in the reference run; `--pool-map` is valid only across independent runs.
For group phases, repeat rank-qualified `--pool-map` for each private-pool pair.

`timeline`, `compare-points`, `compare-phases`, and
`compare-run-group-phases` accept `--stacks`, `--events`, `--lifetimes`,
`--stack-depth`, `--limit`, and `--only-changed`. Lifetime analysis always uses
complete event history internally; `--events` independently controls whether an
allocator-event table is included. `allocation-lifetimes` accepts `--stack-depth` and `--limit` and
always uses event history. `summary` writes to standard output.
Every report-producing command prints to standard output and writes files only
when `--output` is present. Reusing a nonempty report directory requires
`--overwrite`. Cross-run event and lifetime requests are rejected.

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
advanced.KNOWN_TRACE_ACTIONS
advanced.ALLOCATION_LIFETIME_ACTIONS
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
advanced.allocation_stack_coverage(snapshot, *, pool=None)
advanced.summarize_allocation_stacks(
    snapshot,
    *,
    pool=None,
    stream=None,
    by_stream=False,
)
advanced.compare_allocation_stacks(
    reference,
    candidate,
    *,
    pool=None,
    stream=None,
    by_stream=False,
    include_unchanged=False,
)
```

`AllocationStackDelta` stores `reference_key` and `candidate_key`
separately because explicitly mapped private pools can have different IDs.
Its size, requested-byte, and count fields each retain reference, candidate,
and delta values. Advanced stack and event helpers always group by complete
normalized stacks and return every row; callers perform any custom slicing
afterward.

`AllocationStackSummary`, `AllocationStackDelta`, and
`AllocatorEventSummary` expose the complete normalized stack as
`stack_frames`. `stack_key` remains a location-only convenience string, while
`stack_fingerprint` includes every normalized frame field, including optional
`fx_node_op`, `fx_node_name`, and `fx_original_trace` metadata. `to_dict()`
emits `stack_frames` as a JSON array; flat `to_row()` output uses the canonical
JSON column `stack_frames_json` so CSV remains lossless. Text and HTML render
FX metadata alongside its frame location.

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
)
```

Allocator-event rows use the same structured-frame, location-key, fingerprint,
JSON, and CSV contracts as allocation-stack rows.

`extract_event_window` bounds the window by the last occurrence of each
requested marker. Pass `end_marker=None` to read from the start marker
through the end of the trace — this is the correct call when the end
boundary is the snapshot that produced the trace, because a snapshot's own
boundary marker is never visible in its own trace. A non-`None` end marker
that is not found classifies the window as truncated rather than silently
reading to the end of the trace.

Identity, stack-key, and formatting helpers:

```python
advanced.pool_id_label(pool_id)
advanced.stream_label(stream)
advanced.stack_key_from_frames(frames)
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
- `MemoryHistoryDisabledError`: history never recorded on an analyzed device.
- `MemoryHistoryBoundaryError`: a point boundary could not be recorded.
- `MemoryHistoryTruncatedError`: a required marker is missing from the trace;
  ring-buffer overwrite and history enabled late are not always distinguishable.
- `MemoryReconciliationError`: boundary order is invalid or complete history
  contradicts allocator state, indicating corrupted input or a package bug.
- `MemoryBundleError`: malformed, unsupported, or unreadable bundle.
- `MemoryOwnershipError`: point used with a run that does not own it.
