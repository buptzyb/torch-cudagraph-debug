# torch-cudagraph-debug API Reference

This reference describes the public 0.2 API.

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
    CompareTensor,
    PrintTensor,
    RecordTensor,
    TensorDebugError,
    TensorMismatchError,
    TensorProbe,
    TensorProbeStatus,
    TensorSnapshot,
)
```

### TensorProbe

```python
TensorProbe(
    name: str,
    actions: Sequence[PrintTensor | RecordTensor | CompareTensor],
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
allocation, copy, callback, print, record, or compare work. In
`when="always"`, eager calls execute debug work too.

The non-contiguous policy is probe-wide because all actions on one probe inspect
the same source tensor:

- `"error"`: reject a non-contiguous tensor when debug work is active.
- `"copy"`: create and retain an internal contiguous CUDA copy for the debug
  path while returning the original tensor.

A probe is owned by the first CUDA graph capture session that uses it. Multiple
calls in that capture create logical slots in invocation order. Reusing the
probe in another capture is an error. The GPU counter advances once per graph
replay, not once per invocation, so all slots in one replay share a 1-based
replay index. Eager `when="always"` calls do not advance the counter and use
index 0.

#### Methods

```python
probe(tensor: torch.Tensor) -> torch.Tensor
probe.replay_index -> torch.Tensor | None
probe.snapshots(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
) -> list[TensorSnapshot]
probe.clear_snapshots(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
) -> None
probe.status(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
) -> TensorProbeStatus
probe.assert_ok(
    *,
    synchronize: bool | torch.cuda.Stream | torch.device = True,
) -> None
probe.watch_grad(
    tensor: torch.Tensor,
    *,
    strict: bool = False,
) -> torch.utils.hooks.RemovableHandle | None
probe.close() -> None
```

`replay_index` returns a detached GPU clone of the internal scalar. Mutating
it cannot change the probe. Reading the property performs no device-to-host
copy; printing it, calling `.item()`, or moving it to CPU materializes the
value through PyTorch. It is `None` for an all-disabled probe.

The query methods share one synchronization contract. `True` synchronizes the
probe's entire CUDA device and is the correctness-first default. It can wait
for unrelated streams and broaden cross-stream deadlock risk. Passing a
`torch.cuda.Stream` synchronizes only that stream and is the recommended path
when the replay stream is known. Passing a `torch.device` explicitly requests
device-wide synchronization. `False` skips explicit synchronization. A stream
or device must match the probe device. Strings, integer device indices, `None`,
and CPU devices are rejected. Synchronization-enabled queries are invalid
during CUDA Graph capture.

`snapshots()` exposes the latest `RecordTensor` slots as CPU tensors. Capture
installs copy nodes but does not execute them. A Record-only probe reads its GPU
counter at this query point. When Print or Compare is also enabled, the query
reuses their existing pinned-host counter staging instead of issuing another
counter transfer.

`clear_snapshots()` synchronizes according to the same policy before zeroing
retained host storage. It does not remove graph nodes or release the probe.

`status()` returns:

```python
@dataclass(frozen=True)
class TensorProbeStatus:
    ok: bool
    message: str
    replay_index: int
    invocation_index: int
```

`status()` synchronizes callback-backed status before returning it.
`assert_ok()` delegates to `status()` and therefore synchronizes at most once;
it raises `TensorMismatchError` when the sticky status is not OK.

`watch_grad()` registers an autograd hook. The hook probes the gradient for its
side effect and returns the original gradient. It obeys the probe's `when`
policy, so default capture-only probes do no work during eager backward. The
method returns a removable hook handle. If `tensor.requires_grad` is false, it
returns `None`; with `strict=True`, it raises `RuntimeError`.

`close()` releases native resources. Do not close a probe while a graph that
captured it may still replay. `TensorProbe` is also a context manager whose exit
calls `close()`; use that form only when every replay occurs inside the context.

### Actions

```python
PrintTensor(
    max_items: int = 16,
    every: int = 1,
    summary: bool = True,
    enabled: bool = True,
)

RecordTensor(
    enabled: bool = True,
)

CompareTensor(
    expected: torch.Tensor | numpy.ndarray | Sequence[torch.Tensor | numpy.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    enabled: bool = True,
)
```

All enabled actions accept contiguous CUDA tensors with `float16`, `bfloat16`,
`float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, or `bool`
dtype. Other dtypes are rejected.

`PrintTensor` writes to `stderr` from a native CUDA host callback. `max_items`
limits the displayed value prefix, `summary=False` omits aggregate statistics,
and `every=N` prints every invocation on graph replay indices divisible by N.

`RecordTensor` enqueues a device-to-host tensor copy into pinned staging
memory without a host callback when used alone. The probe stores only the latest
staged value for each invocation slot. Its counter remains on the GPU during
replay and transfers only when `snapshots()` explicitly queries it. When Record
is combined with Print or Compare, that query reuses the callback actions'
single shared 8-byte counter transfer.

`CompareTensor.expected` values must be on CPU and match the captured source's
shape and dtype. A single tensor or array is normalized to a one-item expected
sequence and therefore applies only to invocation 0. It is never broadcast.
For a probe called multiple times in one capture, pass one expected value per
invocation in capture-call order.

A compare mismatch is sticky until the probe is destroyed. The first mismatch
records the replay and invocation indices.

### TensorSnapshot

```python
@dataclass(frozen=True, eq=False)
class TensorSnapshot:
    probe_name: str
    replay_index: int
    tensor: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: str
    invocation_index: int
```

Every `snapshots()` call materializes a new CPU tensor copy from the latest
pinned staging bytes and records the probe's current replay counter.
Returned snapshots remain unchanged across later replays, so an additional
`clone()` is not required for retention. The probe stores only the latest value
for each capture-time invocation slot. A capture slot queried before its first
replay uses index 0; after replay, captured snapshots are 1-based. Eager
`when="always"` snapshots also use index 0.

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
    step: int | Callable[[TensorSnapshot], int] | None = None,
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
for replay-by-replay retention and export.

## Memory Debug

Concept guide: [Memory Debug guide](memory_debug.md). Runnable guide:
[Memory Debug examples](../examples/memory_debug/README.md).

### Supported Public Imports

```python
from torch_cudagraph_debug.memory_debug import (
    AllocationLifetimeReport,
    AttributionOptions,
    MemoryBundleError,
    MemoryComparison,
    MemoryDebugError,
    MemoryHistoryError,
    MemoryOwnershipError,
    MemoryGroupSummary,
    MemoryPoint,
    MemoryRange,
    MemoryRecorder,
    MemoryRun,
    MemoryRunGroup,
    MemoryTimeline,
    GroupPhaseComparison,
    PhaseComparison,
    compare_group_phases,
    compare_phases,
    compare_points,
)
```

The facade is intentionally limited to these 19 names. Raw snapshot helpers are
in `memory_debug.advanced` and are experimental.

### MemoryRecorder

```python
MemoryRecorder(
    *,
    name: str = "run",
    bundle_dir: str | pathlib.Path | None = None,
    rank: int | None = None,
    synchronize: bool = True,
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
creates an incomplete manifest, every `mark()` writes one gzip JSON snapshot
and updates that manifest, and `finish()` marks it complete. A nonempty target
directory raises `FileExistsError`.

`synchronize=True` synchronizes outside stream capture before every snapshot.
Set it to `False` only when the application establishes the required stream
ordering itself. Synchronization is always skipped during current-stream
capture because it is capture-illegal.

`group_id` identifies per-rank bundles from one distributed execution.
`run_metadata` is application-owned JSON metadata for source revision,
scenario, command, or other reproducibility fields. The recorder also captures
package/Python/platform/PyTorch/CUDA provenance automatically. Initialized GPU
properties are added only after a real snapshot, so recorder construction does
not initialize CUDA for metadata collection.

#### mark

```python
recorder.mark(
    label: str,
    *,
    metadata: Mapping[str, JSONValue] | None = None,
) -> MemoryPoint
```

Labels must be nonempty and unique. Outside CUDA capture, `mark()` synchronizes
by default. During current-stream capture it skips synchronization and snapshots
immediately.

The recorder temporarily sets a PyTorch allocator metadata marker around the
snapshot when those private APIs are available. This marker delimits same-run
event windows. Marker failures become point warnings.

Snapshot and metadata values must be JSON-compatible: null, string, bool,
finite number, list/tuple, or a mapping with string keys. Validation errors
include the path to the unsupported value.

#### lifecycle

```python
recorder.snapshot_run() -> MemoryRun
recorder.finish() -> MemoryRun
recorder.result -> MemoryRun
```

`snapshot_run()` returns an immutable incomplete view without stopping
collection. `finish()` is idempotent, makes later `mark()` calls invalid,
writes a complete manifest, and returns an immutable run. `result` is
available only after finish.

As a context manager, the recorder calls `finish()` on exit and does not
suppress exceptions.

For tests only, `MemoryRecorder._from_snapshot_provider(...)` injects synthetic
snapshots without exposing a provider in the public constructor.

### MemoryPoint

A point exposes `run_id`, `index`, `label`, `timestamp`, immutable
metadata, its event `boundary_marker`, compact
`groups: Mapping[GroupKey, MemoryStats]`, and warnings. Pool and scope mappings
are cached derivations of that state.

```python
point.groups
point.pools
point.totals  # all/default/private MemoryStats
point.raw_snapshot() -> Mapping | Sequence
point.descriptor() -> dict
```

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
run.between(before, after) -> MemoryRange
run.compare(before, after, *, attribution=None) -> MemoryComparison
run.timeline(*, attribution=None) -> MemoryTimeline
run.lifetimes(
    at=None,
    *,
    born_between=None,
    through=None,
    attribution=None,
) \
    -> AllocationLifetimeReport
```

`MemoryRun` also carries `rank`, `group_id`, `world_size`, immutable automatic
`provenance`, and immutable user `run_metadata`.

A point passed to `point()`, `between()`, or `compare()` must carry the
same `run_id` and correspond to a point in that run. Otherwise
`MemoryOwnershipError` is raised. `after` must have a larger index than
`before`.

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
    before: MemoryPoint
    after: MemoryPoint

memory_range.compare(*, attribution=None) -> MemoryComparison
```

`run.between()` validates that both points belong to the run and that `after`
follows `before`. `MemoryRange.compare()` is equivalent to calling
`run.compare(memory_range.before, memory_range.after, ...)`. Phase comparison
accepts one baseline range and one candidate range.

### AttributionOptions

```python
AttributionOptions(
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

- No history: state and lifecycle work; allocation stacks and events are
  generally unavailable.
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
and candidate growth ranges. Cross-run lifetime requests are rejected.

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
    attribution: AttributionOptions | None = None,
) -> AllocationLifetimeReport
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
`AttributionOptions(events=False, ...)` for snapshot-only analysis. Cohorts are
ranked by bytes born for `born_between`, by bytes active at an anchor, or by
peak-to-minimum impact when there is no selection.

This API reports allocator-block evidence. It does not recover Python tensor
names, object identity, dtype, shape, or higher-level ownership unless those
details are inferable from the recorded call stacks and allocation sizes.

### Timeline

`run.timeline()` reports absolute state and a delta for every observed pool
and `(pool, stream)` group at every point. The first point is relative to zero.
If a pool disappears, the current state is zero and its negative delta remains
visible. Unchanged rows are retained by default.

The default timeline is manifest-only and does not read raw snapshots;
`timeline.adjacent` is empty. Requesting stack or event attribution streams each
point once and stores attributed same-run comparisons in `timeline.adjacent`.

With `lifetimes=True`, one full-run cohort report is attached as
`timeline.allocation_lifetimes`. Adjacent comparisons do not repeat the same
lifetime scan.

### Cross-Run Comparison

```python
compare_points(
    before: MemoryPoint,
    after: MemoryPoint,
    *,
    pool_mapping: Mapping[PoolId, PoolId] | None = None,
    attribution: AttributionOptions | None = None,
) -> MemoryComparison
```

The points must have different `run_id` values. Without `stacks=True`, the
comparison uses compact manifest state and never reads raw snapshots. Cross-run
matching is conservative:

1. Default `(0,0)` pools match automatically when present in both runs.
2. Private pools match only through a one-to-one `pool_mapping`; every mapped
   source and target must exist at the selected points.
3. Identical raw private IDs are still unmatched without that mapping.
4. Streams are never matched across runs; all pool/stream rows are before-only
   or after-only.
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
    attribution: AttributionOptions | None = None,
) -> PhaseComparison
```

The result contains `baseline_growth`, `candidate_growth`, `start_delta`,
`end_delta`, a decomposition row for each matched pool and metric, and a
`total_decomposition` for each `all`/`default`/`private` scope. Every row
verifies:

```text
end_delta = start_delta + candidate_growth - baseline_growth
```

Every explicit private-pool mapping must exist at all four selected phase
points because the mapping is applied to both cross-run endpoint comparisons.
Seed a shared graph pool before the phase starts when the phase itself will grow
that pool.

Event attribution applies to the two same-run growth comparisons. Start/end
cross-run comparisons never compare events.
Lifetime attribution follows the same rule: each growth range owns its cohort
report, while start/end cross-run comparisons do not.

### Multi-Rank Run Groups

```python
MemoryRunGroup.load(root, *, cache_snapshots=False) -> MemoryRunGroup
MemoryRunGroup.from_runs(runs, *, root=None) -> MemoryRunGroup
group.ranks -> tuple[int, ...]
len(group) -> int
group[rank] -> MemoryRun
group.descriptor() -> dict
group.summary() -> MemoryGroupSummary

compare_group_phases(
    baseline: MemoryRunGroup,
    candidate: MemoryRunGroup,
    *,
    baseline_start: str | int,
    baseline_end: str | int,
    candidate_start: str | int,
    candidate_end: str | int,
    attribution: AttributionOptions | None = None,
) -> GroupPhaseComparison
```

`load()` reads direct `*.tcgd-memory` child directories. A group requires
non-null unique ranks, one run name, one ordered point-label sequence, and no
conflicting non-null group IDs or world sizes. Declared-but-missing ranks,
missing identity fields, incomplete bundles, runtime provenance differences,
and user metadata differences are warnings. The default does not retain
decompressed raw snapshots across ranks or points; set
`cache_snapshots=True` only for workloads that repeatedly inspect the same
raw payloads.

`MemoryGroupSummary` emits per-rank point/scope states plus min, max, spread,
and worst rank. `GroupPhaseComparison` pairs common ranks and aggregates the
per-rank four-point total equations. Neither API sums GPU memory across ranks.

### Result Objects

`MemoryComparison`, `MemoryTimeline`, `PhaseComparison`,
`AllocationLifetimeReport`, `MemoryGroupSummary`, and
`GroupPhaseComparison` own rendering. Their main programmatic fields are:

- `MemoryComparison`: `before`, `after`, `totals`, `pools`,
  `pool_streams`, optional stacks/events/lifetimes, and `warnings`.
- `MemoryTimeline`: `run`, pointwise `totals`, `pools`, `pool_streams`, optional
  `adjacent` comparisons and lifetimes, and derived `warnings`.
- `PhaseComparison`: `baseline_growth`, `candidate_growth`, `start_delta`,
  `end_delta`, total decomposition, matched-pool decomposition, and `warnings`.
- `AllocationLifetimeReport`: selection points, `cohorts`, history coverage,
  attributed bytes, and `warnings`.
- `MemoryGroupSummary`: `group`, per-rank `rank_points`, cross-rank
  `point_summary`, and `warnings`.
- `GroupPhaseComparison`: both groups, per-rank comparisons, `rank_phase`,
  `phase_summary`, and `warnings`.

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
  "before": {"allocated_bytes": 1024},
  "after": {"allocated_bytes": 2048},
  "delta": {"allocated_bytes": 1024}
}
```

CSV flattens these as `before_allocated_bytes`,
`after_allocated_bytes`, and `delta_allocated_bytes`.

Every `write()` creates `report.txt`, `report.json`, and `report.html`.
Pool-oriented results also create `totals.csv`, `pools.csv`, and
`pool_streams.csv`, with optional `allocation_stacks.csv`, `events.csv`,
`phase.csv`, and `phase_totals.csv`.

`AllocationLifetimeReport`, and pool-oriented results that embed one, create
`cohorts.csv`, `cohort_points.csv`, and `size_histograms.csv`. They add
`birth_stacks.csv` or `release_stacks.csv` when those observations exist.
Lifetime JSON keeps point states, histograms, birth/release confidence,
point/event peaks, and exact/inferred/still-active totals nested under each
cohort.

Group summaries create `rank_points.csv` and `point_summary.csv`. Group phase
reports create `rank_phase.csv` and `phase_summary.csv`.

### Bundle Format

The bundle schema is `torch-cudagraph-debug/memory-run`. A bundle has one
plain JSON manifest and `snapshots/NNNN.json.gz` files. Writes use a temporary
file followed by atomic replacement. Snapshot paths are validated to remain
inside the bundle.

Manifest, point, and group objects have canonical required fields. Group rows
store only the seven base `MemoryStats` fields; inactive and fragmentation
values are derived after loading. Missing or unknown fields are rejected.

Loading reads only manifest summaries and never executes pickle. A bundle has
one writer; distributed users create one bundle per rank.

### CLI

The [CLI workflow example](../examples/memory_debug/cli_workflows.sh) creates
its own bundles and exercises every command below.

```text
tcgd-memory lifetimes BUNDLE \
  [--at POINT | --born-between START END] [--through POINT] \
  [--no-events] --output DIR
tcgd-memory timeline BUNDLE --output DIR
tcgd-memory compare BUNDLE --before POINT --after POINT --output DIR
tcgd-memory compare-runs BEFORE_BUNDLE AFTER_BUNDLE \
  --before POINT --after POINT [--pool-map A=B] --output DIR
tcgd-memory compare-phases BASELINE_BUNDLE CANDIDATE_BUNDLE \
  --baseline-start POINT --baseline-end POINT \
  --candidate-start POINT --candidate-end POINT \
  [--pool-map A=B] --output DIR
tcgd-memory summarize-group GROUP_DIR --output DIR
tcgd-memory compare-group-phases BASELINE_GROUP CANDIDATE_GROUP \
  --baseline-start POINT --baseline-end POINT \
  --candidate-start POINT --candidate-end POINT --output DIR
```

Every command except `summarize-group` accepts `--stacks`, `--events`,
`--lifetimes`, `--on-missing {warn,error}`, `--stack-depth`, `--limit`, and
`--only-changed`; `summarize-group` accepts only its group directory and
`--output`. The dedicated `lifetimes` command always groups by
allocation stack, enables events by default, and uses stack depth 4;
`--no-events` requests snapshot-only inference.
Cross-run event and lifetime requests are rejected. Pool IDs use
comma-separated components, for example `--pool-map 0,1=0,4`.

## Experimental Advanced API

```python
from torch_cudagraph_debug.memory_debug import advanced
```

The advanced facade is intended for custom snapshot analysis and may change in
a minor release. The supported public facade above follows semantic versioning.
All 26 advanced exports are grouped below.

Snapshot and state types:

```python
advanced.SnapshotInput
advanced.GroupKey
advanced.MemoryStats
advanced.TraceEntry
```

Snapshot normalization and aggregation:

```python
advanced.normalize_pool_id(pool_id)
advanced.normalize_snapshot(snapshot)
advanced.normalize_trace_entries(snapshot)
advanced.summarize_snapshot(snapshot)
advanced.summarize_segments(segments)
advanced.summarize_pools(groups)
```

`normalize_snapshot()` accepts either the full mapping returned by
`torch.cuda.memory._snapshot()` or a segment sequence. `summarize_snapshot()`
and `summarize_segments()` return `Mapping[GroupKey, MemoryStats]`;
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
    before,
    after,
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
    before_segments,
    after_segments,
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
advanced.format_before_after(before, after, delta)
```

## Errors

- `CudaGraphDebugError`: package-wide base.
- `NativeExtensionUnavailableError`: compiled tensor extension cannot be loaded.
- `TensorDebugError`: tensor domain base.
- `TensorMismatchError`: sticky tensor compare mismatch.
- `MemoryDebugError`: memory domain base.
- `MemoryHistoryError`: requested history unavailable or incomplete.
- `MemoryBundleError`: malformed, unsupported, or unreadable bundle.
- `MemoryOwnershipError`: point used with a run that does not own it.
