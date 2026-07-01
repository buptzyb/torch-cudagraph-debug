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

### Stable Imports

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
)
```

`name` and `actions` must be nonempty. Disabled actions are filtered before
native probe creation. If every action has `enabled=False`, the probe is a
pure no-op and does not load the native extension.

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
probe in another capture is an error.

#### Methods

```python
probe(tensor: torch.Tensor) -> torch.Tensor
probe.snapshots() -> list[TensorSnapshot]
probe.clear_snapshots() -> None
probe.status() -> TensorProbeStatus
probe.assert_ok() -> None
probe.watch_grad(
    tensor: torch.Tensor,
    *,
    strict: bool = False,
) -> torch.utils.hooks.RemovableHandle | None
probe.close() -> None
```

`snapshots()` exposes latest `RecordTensor` slots as CPU tensors. Call it
only after graph replay and synchronization. Capture installs copy nodes but
does not execute them.

`clear_snapshots()` zeroes retained host storage. It does not remove graph
nodes or release the probe.

`status()` returns:

```python
@dataclass(frozen=True)
class TensorProbeStatus:
    ok: bool
    message: str
    replay_index: int
    invocation_index: int
```

`assert_ok()` raises `TensorMismatchError` when the sticky compare status is
not OK.

`watch_grad()` registers an autograd hook. The hook probes the gradient for its
side effect and returns the original gradient. The method returns a removable
hook handle. If `tensor.requires_grad` is false, it returns `None`; with
`strict=True`, it raises `RuntimeError`.

`close()` releases native resources. Do not close a probe while a graph that
captured it may still replay.

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

`PrintTensor` prints from a native CUDA host callback. `RecordTensor` stores
latest host snapshots without requiring a host callback when used alone.

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

The tensor is a CPU view of retained latest-record storage. Clone it when a
replay-by-replay time series must survive later replays.

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
TensorBoard itself. The caller owns synchronization and writer lifecycle.

## Memory Debug

### Stable Imports

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
metadata, compact `groups: Mapping[GroupKey, MemoryStats]`, and warnings. Pool
and scope mappings are cached derivations of that state.

```python
point.groups
point.pools
point.totals  # all/default/private MemoryStats
point.raw_snapshot() -> Mapping | Sequence
point.descriptor() -> dict
```

For persisted runs, `raw_snapshot()` lazily reads and caches the point's gzip
JSON payload.

### MemoryRun

```python
MemoryRun.load(bundle_dir, *, cache_snapshots=True) -> MemoryRun
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
cohort retains point-by-
point active/requested bytes, block counts, stream IDs, a unique-instance size
histogram, allocation births, release stacks, and point/event peaks.

Release evidence is explicit:

- `event_exact` means a matching `free_requested` event exists in the marker-
  delimited interval;
- `snapshot_inferred` means an observed block disappeared without exact event
  evidence;
- `still_active_bytes` counts observed instances active at the end point.

The application must enable full allocator history before the allocations of
interest to obtain exact release timing and free call stacks. With events
disabled or unavailable, snapshot-inferred results remain usable. Snapshot-
only matching cannot detect a free-and-reallocate cycle that reuses the same
address and shape entirely between two points.

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
2. Private pools match only through a one-to-one `pool_mapping`.
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

Event attribution applies to the two same-run growth comparisons. Start/end
cross-run comparisons never compare events.
Lifetime attribution follows the same rule: each growth range owns its cohort
report, while start/end cross-run comparisons do not.

### Multi-Rank Run Groups

```python
MemoryRunGroup.load(group_dir, *, cache_snapshots=False) -> MemoryRunGroup
MemoryRunGroup.from_runs(runs) -> MemoryRunGroup
group[rank] -> MemoryRun
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
`GroupPhaseComparison` own rendering:

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

Shared options are `--stacks`, `--events`, `--lifetimes`,
`--on-missing {warn,error}`, `--stack-depth`, `--limit`, and
`--only-changed`. The dedicated `lifetimes` command always groups by
allocation stack, enables events by default, and uses stack depth 4;
`--no-events` requests snapshot-only inference.
Cross-run event and lifetime requests are rejected. Pool IDs use comma-
separated components, for example `--pool-map 0,1=0,4`.

## Experimental Advanced API

```python
from torch_cudagraph_debug.memory_debug import advanced
```

The module exposes raw normalization, `MemoryStats` state aggregation, stack
grouping, event-window, and byte-formatting helpers. `summarize_snapshot()` and
`summarize_segments()` return `Mapping[GroupKey, MemoryStats]`. These helpers
may change in a minor release; the stable facade follows semantic versioning.

## Errors

- `CudaGraphDebugError`: package-wide base.
- `TensorDebugError`: tensor domain base.
- `TensorMismatchError`: sticky tensor compare mismatch.
- `MemoryDebugError`: memory domain base.
- `MemoryHistoryError`: requested history unavailable or incomplete.
- `MemoryBundleError`: malformed, unsupported, or unreadable bundle.
- `MemoryOwnershipError`: point used with a run that does not own it.
