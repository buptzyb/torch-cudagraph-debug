# Memory Debug Guide

`memory_debug` captures CUDA allocator state and analyzes default and
non-default pools, including CUDA Graph private pools. `MemoryProbe.snapshot()`
returns a standalone `MemoryProbeSnapshot` for direct two-point comparison.
`MemoryRecorder` produces a `MemoryRun` whose labeled points support
persistence, timelines, phases, and run groups. Probe snapshots and Recorder
points contain the same ownerless `MemoryObservation` leaves; both workflows
reuse the private memory collector. This guide covers collection,
allocator-history ownership, comparisons, lifetimes, reports, bundles, and
multi-rank workflows. For exact signatures, see the
[API reference](api.md#memory-debug); for runnable programs, follow the
[Memory Debug examples](../examples/memory_debug/README.md).

## Choose An Analysis

| Question | API | Allocator history for the base result |
|---|---|---|
| What changed between two quick snapshots? | `probe.compare()` or `compare_snapshots()` | Not required |
| What changed between two points in one run? | `run.compare()` | Not required |
| How did allocator state evolve across all points? | `run.timeline()` | Not required |
| Which allocations were born, released, or remained active? | `run.lifetimes()` | Complete event history required |
| How do endpoints from independent runs differ? | `compare_points()` | Not required; state history enables stack attribution |
| How does candidate phase change differ from baseline phase change? | `compare_phases()` | Not required; attribution is optional on each same-run change leg |
| Which ranks differ or have the worst change? | `MemoryRunGroup.summary()` and `compare_run_group_phases()` | Not required; phase attribution is optional |

Allocator history is not required for pool, stream, segment, block, or
fragmentation state. Stack attribution uses available live-block frames, while
event and lifetime analyses require the corresponding complete history.
Neither the probe nor the recorder
enables history on the application's behalf.

## Core Usage

Import the supported public memory API from `memory_debug`:

```python
from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryDisplayOptions,
    MemoryLifetimeSelection,
    MemoryPoolKey,
    MemoryLifetimeOptions,
    MemoryProbe,
    MemoryRecorder,
    MemoryRun,
    MemoryRunGroup,
    compare_run_group_phases,
    compare_phases,
    compare_points,
    compare_snapshots,
)
```

Both workflows collect `torch.cuda.memory._snapshot()` and discover every pool
present on selected devices; users do not pass pool handles. `devices=None`
binds the current device lazily, an explicit device or sequence selects known
devices, and `devices="all"` selects every visible device. Within each device,
`pool[0,0]` is the default pool and other IDs represent private pools such as
CUDA Graph pools.

The control-flow and containment relationships are distinct:

```text
MemoryProbe --returns--> MemoryProbeSnapshot
MemoryRecorder --produces--> MemoryRun --contains--> MemoryPoint
MemoryProbeSnapshot and MemoryPoint --contain--> MemoryObservation
```

Each snapshot or point contains one ordered observation for every
`(device_index, pool_id, stream)` key. `pool_stats` aggregates by
`MemoryPoolKey(device_index, pool_id)`, and `allocator_scope_stats` provides
`all`, `default`, and `private` totals across selected devices. Ownership and
workflow metadata live on the snapshot or point, not on an observation.

## Interpreting CUDA Graph Private-Pool Inactive Memory

`MemoryStats` separates four base allocator layers plus three derived
metrics:

- `reserved_bytes` is segment capacity retained by the pool.
- `active_bytes` is block space that is allocated or still awaiting a
  stream-safe free.
- `allocated_bytes` is active block space still owned by live allocations.
- `requested_bytes` is the unrounded request size represented by active blocks.
- `awaiting_free_bytes = active_bytes - allocated_bytes` is no longer owned by
  live allocations but cannot yet be reused.
- `inactive_bytes = reserved_bytes - active_bytes` is currently reusable under
  that pool's allocator rules.
- `expandable_inactive_bytes` is the inactive capacity residing in expandable
  segments. The allocator may return completely unused pages from that
  capacity under memory pressure, but the metric is not an exact count of
  currently releasable pages. Inactive capacity in regular segments returns
  to the driver only when a whole unsplit segment is released. With
  `expandable_segments:True`, `largest_inactive_block_bytes` also loses its
  fragmentation-verdict power: an allocation that fits no hole can still be
  served by growing the segment in place.
  The [expandable-segment example](../examples/memory_debug/probe/expandable_segments.py)
  shows inactive mapped capacity before `empty_cache()` and the exact range
  removed when the allocator unmaps an interior hole.
- `internal_fragmentation_bytes = active_bytes - requested_bytes` is allocator
  rounding inside active or awaiting-free blocks.

Comparison and timeline text and HTML render one decomposition tree per device:

```text
device[0]
  CUDA total: 44.39 GiB -> 44.39 GiB (0 B)
  CUDA used: 434.19 MiB -> 540.19 MiB (+106.00 MiB)
    residual: 434.19 MiB -> 522.19 MiB (+88.00 MiB)
    allocator:
      reserved: 0 B -> 18.00 MiB (+18.00 MiB)
      allocated: ..., active: ..., requested: ...
      diagnostics: ...
      pool[0,0] (default)
        reserved: ...
        stream[336943840]
          ...
```

`CUDA used = residual + allocator reserved`; allocator reserved is the sum of
its pools, and each pool is the sum of its streams. The residual can be
positive or negative;
[Device-Wide CUDA Runtime Memory](#device-wide-cuda-runtime-memory) explains
its measurement and scope. Interior allocator, pool, and stream nodes put
`reserved:` first, followed by the remaining core metrics and sparse structural
diagnostics. The `diagnostics` line is intentionally sparse: text and HTML
surface structural metrics only when they explain an absolute state or change;
JSON and CSV retain all metrics. Points without a CUDA Runtime sample attach
`allocator:` directly to the device.

Cross-run device roots retain both endpoint indices, for example
`device[0] -> device[1] [mapped]`. Match tags appear only when the identity
decision carries information. The `depth` option (`"device"`, `"pool"`, or
`"stream"`, default `"stream"`) truncates text and HTML. One bottom-up filter
implements `include_unchanged=False`: a changed descendant keeps its complete
parent path. CSV remains flat but keeps those parent context rows; JSON and
in-memory results always remain complete.

For a CUDA Graph private pool, inactive does not mean that the memory is
available to the default pool or has been returned to the CUDA driver. Capture
records kernel arguments, including device addresses. Replay submits those
recorded kernels without rerunning the Python allocation and deallocation calls
that constructed the captured workload, so the graph must retain its address
space across replays. PyTorch keeps the private pool alive until the graph and
the tensors created during capture go out of scope; see
[Graph memory management](https://docs.pytorch.org/docs/main/notes/cuda.html#graph-memory-management).

A graph may therefore reach a high reserved-memory watermark during capture,
then show low active memory after transient activations or workspaces are freed.
The difference appears as inactive private-pool capacity. This is not by itself
evidence of a tensor leak, but it is still unavailable to unrelated default-pool
allocations. Investigate capture-time peaks, graph count and pool sharing,
segment layout, and graph lifetime before deciding whether the reservation is
necessary. The
[private-pool inactive example](../examples/memory_debug/probe/private_pool_inactive.py)
demonstrates this state without allocator history.

## Quick Two-Point Comparison

Use `MemoryProbe` when labels, persistence, and multi-point analysis are not
needed:

```python
import torch

probe = MemoryProbe("capture")
before = probe.snapshot()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_buffer = torch.empty((4096, 4096), device="cuda")

after = probe.snapshot()
comparison = probe.compare(before, after)

print(after.allocator_scope_stats["private"])
print(comparison.to_text())
```

`snapshot()` returns one complete allocator state. The probe's `synchronize=`
constructor argument sets the default synchronization target, and each
`snapshot(synchronize=...)` call may override it — the same bool/stream/device
contract as `record_point()`. `MemoryProbeSnapshot.raw_snapshot()` keeps the
complete in-memory raw snapshot available for low-level local inspection.
`probe.compare()` validates
that both snapshots belong to that probe and are in increasing index order.
Top-level `compare_snapshots(reference, candidate, device_mapping=...,
pool_mapping=...)` also compares independent probes, which is useful for two
independently instrumented workloads. Device pairing is explicit first, then
inferred from cross-device pool mappings, then matched by an unclaimed equal
index. Default pools follow resolved device pairs; private pools stay unmatched
unless `pool_mapping` explicitly pairs them, exactly like cross-run
`compare_points`.

Structured pool and stream rows carry `MatchKind`; device rows carry
`DeviceMatchKind`. Pool values are `same_probe`, `same_run`, `default`,
`mapped`, `reference_only`, or `candidate_only`. Device values replace
`default` with `same_index` and add `pool_mapping`. Text and HTML show tags
only when
the decision matters: `[mapped]`, `[pool_mapping]`, `[reference_only]`, or
`[candidate_only]`. Owner-local and automatic same-index/default matches stay
untagged.
Same-probe comparisons may request marker-delimited allocator events and
lifetimes when the application enabled allocator history before the interval.
The embedded lifetime result reports `source_kind="probe"`; Run-based lifetime
analysis reports `source_kind="run"`, without adapting either workflow into
the other.
Independent-probe comparisons support state and allocation stacks, but not
events or lifetimes. Use a Recorder for a durable multi-point investigation.

## Complete Run Workflow

```python
import torch

recorder = MemoryRecorder(
    name="capture",
    rank=0,
    group_id="job-20260630",
    world_size=4,
    run_metadata={
        "source_revision": "abc123",
        "scenario": "cuda-graph",
    },
    bundle_dir="rank0.tcgd-memory",
)
recorder.record_point("before_capture")

pool = torch.cuda.graph_pool_handle()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, pool=pool):
    tmp_a = torch.empty((4096, 4096), device="cuda")
    recorder.record_point("during_capture")
    tmp_b = torch.empty((2048, 4096), device="cuda")

recorder.record_point("after_capture")
graph.replay()
recorder.record_point("after_replay")
run = recorder.finish()
```

`bundle_dir` must be an absent or empty directory; a nonempty target raises
`FileExistsError`, so pick a fresh directory per run. The manifest is
rewritten after every `record_point()`, so a crashed run leaves a loadable
`complete=False` bundle.

The manifest records rank/group identity, user `run_metadata`, and automatic
runtime provenance such as package, Python, PyTorch, CUDA, and initialized GPU
properties. Device provenance is collected only after a real allocator
snapshot; creating a recorder does not initialize CUDA for metadata alone.

`record_point()` synchronizes every selected device by default outside capture.
Pass a CUDA stream to synchronize only the workload stream, a CUDA device to
request device-wide synchronization explicitly, or `False` when the application
owns the ordering. During CUDA stream capture, requested synchronization is
skipped because it is capture-illegal, and the point records a warning.
The point-in-time allocator state remains usable, but an event interval touching
a point whose boundary marker could not be recorded fails with
`MemoryHistoryBoundaryError` when event or lifetime evidence is requested
(`boundary_unavailable` is the per-device status recorded at collection).

`MemoryRecorder` only collects data. `finish()` returns an immutable
`MemoryRun`, and `preview()` returns a view of the collected points during
collection (the terminal result afterward); analysis belongs to the run and
result objects:

```python
comparison = run.compare("before_capture", "after_capture")
print(comparison.to_text())

timeline = run.timeline()
timeline.write("reports/timeline")
```

The context-manager form finishes automatically:

```python
with MemoryRecorder(name="capture") as recorder:
    recorder.record_point("start")
    # workload
    recorder.record_point("end")

run = recorder.result
```

A normal exit sets `run.complete=True`. If the body raises, the recorder keeps
the points already collected, sets `finished_at`, writes a terminal
`complete=False` manifest when persistence is enabled, and blocks later
collection. The application exception still propagates, and
`recorder.result` remains available for postmortem analysis.

## Allocator History Is Application-Owned

Neither workflow calls `torch.cuda.memory._record_memory_history()`. Pool,
stream, segment, block, and fragmentation state works without history.
Attribution requires the application to enable the corresponding history before
the allocations of interest:

```python
torch.cuda.memory._record_memory_history(
    enabled="state",
    context="state",
    stacks="python",
    clear_history=True,
)
```

Use `enabled="all", context="all"` when historical allocator events are also
needed. Then request attribution explicitly:

```python
options = MemoryAttributionOptions(
    stacks=True,
    events=True,
    display=MemoryDisplayOptions(stack_depth=2, limit=20),
)
comparison = run.compare("start", "end", attribution=options)
```

Event and lifetime requests fail with typed errors instead of degrading:
`MemoryHistoryDisabledError` when required history evidence is unavailable on
an analyzed device, `MemoryHistoryBoundaryError` when either endpoint could not
record its metadata boundary, and `MemoryHistoryTruncatedError` when a required
start marker is missing from the trace. The tool cannot distinguish bounded-ring
overwrite from history enabled after that boundary. Recorder intervals use the
ending snapshot's trace end because its own marker is normally absent, so a
complete interval assumes history remained enabled between the points. Stack
attribution returns
exact framed rows plus an `<unattributed>` bucket when coverage is partial; it
raises `MemoryHistoryDisabledError` only when nonempty active state has zero
frame coverage. Invalid boundary order or complete history that cannot be
reconciled with allocator states raises `MemoryReconciliationError`
unconditionally. Possible causes include overlapping marker-bearing collection,
allocator activity during the non-atomic marker/snapshot window, corrupted
input, or an implementation defect.

Every successful comparison exposes `attribution_status`. `requested`
distinguishes an analysis the caller asked for, and `available`/`complete`
describe the evidence that backed it. `allocation stack coverage` is attributed
active block bytes divided by total active block bytes at each endpoint; an
endpoint with zero active bytes reports full coverage. In a
lifetime report, `instance stack coverage` is the sum of sizes for tracked
allocation generations with frames divided by the sum for all tracked
generations; it is not a point-in-time memory peak.

A block's `frames` are the allocation call stack for memory still active in a
snapshot. Entries under `device_traces` are historical allocator events, and
their `frames` are event call stacks. These are separate data sources.
Recorder bundles remove cumulative `device_traces` from point state and
preserve only each adjacent interval's raw event mappings.

Each allocator-event row reports pool-attribution confidence:

- `reported`: the raw event supplied its pool ID;
- `matched`: the event address resolves uniquely through endpoint segment
  ranges;
- `ambiguous`: endpoint ranges associate the address with conflicting pools;
- `unknown`: no address was available or no endpoint range contains it;
- `not_applicable`: the event describes the device rather than an allocator
  block (`oom`, whose payload is the device's free-byte count, not an
  address), so pool attribution is never attempted.

Rendered rows use `pool[n/a]` for `not_applicable`; `pool[unknown]` means
the pool could apply but available evidence could not determine it.

Confidence describes pool attribution, not event-history completeness; the
comparison's `attribution_status.events` carries the latter.

The normalized raw action vocabulary is `alloc`, `free_requested`,
`free_completed`, `segment_alloc`, `segment_free`, `segment_map`,
`segment_unmap`, `snapshot`, and `oom`. Unknown future actions remain visible
in event reports and produce a lifetime warning instead of aborting analysis.
PyTorch `memory_viz` may display a synthetic action named `free` when it
collapses adjacent `free_requested` and `free_completed` entries; `free` is not
a raw `_snapshot()` trace action.

## Analysis Modes

The basic operations remain timeline and two-point comparison:

```python
run = MemoryRun.load("rank0.tcgd-memory")

timeline = run.timeline()
change = run.compare("before_capture", "after_capture")
phase_range = run.between("before_capture", "after_capture")
```

The default timeline is manifest-only: it computes absolute state and deltas
without decompressing allocator-state files, and
`timeline.point_comparisons` is empty.
Passing `MemoryAttributionOptions(stacks=True)` loads each required allocator
state, and `events=True` additionally loads each ending point's event evidence.
When events and lifetimes are requested together, one loaded event window backs
both analyses.
The timeline then attaches attributed adjacent comparisons.
The first point has `delta=None`; each later point is compared with its immediate
predecessor. Pools that disappear remain visible with zero state and a negative
delta.

Every point and comparison also exposes allocator-wide `all`, `default`, and
`private` totals. These totals include unmatched private pools, so a cross-run
headline does not silently become a default-pool-only number. Pool and
device/pool/stream rows remain available for detailed inspection.

### Device-Wide CUDA Runtime Memory

Allocator snapshots cover only the caching allocator of the recording process.
Each snapshot and point also attempts `torch.cuda.mem_get_info` for every
selected device and stores the result in `device_memory`. A sample records
CUDA-visible `free_bytes` and `total_bytes`;
`used_bytes = total_bytes - free_bytes`. Comparisons and timelines also derive

```text
cuda_allocator_residual_bytes = used_bytes - allocator_reserved_bytes
```

The residual can be positive or negative. `_snapshot()` and `mem_get_info()` are
consecutive, not atomic, and CUDA used includes the process context, NCCL
buffers, library workspaces, raw `cudaMalloc`, every other process sharing the
GPU, and concurrent changes between the two measurements. The value is evidence
not explained by this process allocator's reserved state, not ownership
attribution. `allocator_reserved_bytes` remains process-local; CUDA used, free,
total, and residual values are device-global.

Reports expose `delta_total_bytes` as well as used and free deltas. When
CUDA-visible total capacity changes, used growth is not allocation growth alone:
`delta_used_bytes = delta_total_bytes - delta_free_bytes`. A missing endpoint
sample renders as `n/a` and produces no fabricated delta. Sample availability
at only one endpoint still makes the device comparison changed, and devices
whose pools exist without samples still render their allocator subtree.

### Allocation Cohort Lifetimes

Allocation cohort lifetimes are an optional focused same-run analysis, not a
replacement for the timeline and comparison modes above. To answer "what was
live here, and when did it go away?", anchor the analysis at the point of
interest:

```python
lifetimes = run.lifetimes(
    MemoryLifetimeSelection.active_at("before_capture"),
    through="after_replay",
    options=MemoryLifetimeOptions(
        display=MemoryDisplayOptions(stack_depth=4, limit=20),
    ),
)
print(lifetimes.to_text())
lifetimes.write("reports/lifetimes")
```

The anchor keeps only allocation generations that are not reusable at that
point. Cohorts use device, pool, and the complete allocation call stack as their
identity; allocations without stack frames additionally include their block and
requested sizes in the identity so unrelated unattributed sizes are not merged.
Streams, allocation sizes, requested sizes, and terminal outcomes remain
available as detail. `stack_depth` shortens rendered stacks only and
`limit` restricts text/HTML presentation only. JSON, CSV, and the in-memory
`cohorts` tuple always retain the complete result.

A generation can occupy three relevant states:

- **owner active**: the application still owns the allocation;
- **awaiting free**: `free_requested` occurred, but stream-ordered work still
  prevents allocator reuse;
- **free completed**: `free_completed` occurred and the block can be reused by
  the allocator.

The cohort summary separates point sampling from event replay:

- `snapshot_peak` is the largest cohort `active_bytes` value in any recorded
  point snapshot.
- `snapshot_blocks` is the largest cohort block count in any recorded point
  snapshot.
- `owner_event_peak` is the largest owner-active byte total reconstructed from
  the range-start state and allocator events.
- `unreusable_event_peak` is the largest reconstructed owner-active plus
  awaiting-free byte total.

An allocation born and freed between adjacent points can therefore have zero
`snapshot_peak` and `snapshot_blocks` but nonzero event peaks.

Free request and completion are reported independently, each with its own
stack table. Transitions inside the analyzed range are backed by allocator
events. When a block is already awaiting free at the first point, its earlier
request is represented once with `origin="range_boundary"` rather than assigned
to a guessed interval. `free_completed` means allocator-reusable; it does not
mean the segment was returned to CUDA or that pool `reserved_bytes` decreased.

Lifetime analysis requires complete allocator event history for the analyzed
range and reconciles the event stream against every allocator state. An `alloc`
event first identifies a provisional generation by device, address, and
requested size; its first snapshot may add allocator rounding and missing pool,
stream, or stack metadata. Later snapshots must preserve that confirmed
generation's rounded size, requested size, known pool and stream, and nonempty
allocation stack until a `free_completed`/`alloc` sequence witnesses address
reuse. Duplicate active addresses or duplicate `free_requested` transitions are
also contradictions. The raised `MemoryReconciliationError` summarizes each
reason and device with the total count and up to three example addresses. The
report describes allocator blocks, not Python tensor names or object ownership.

Synchronizing the recorded stream completes the CUDA work, but the caching
allocator may not emit `free_completed` until a later allocator operation polls
its pending events. A snapshot taken before that poll can still legitimately
show `active_awaiting_free`.

To answer a different question, "what was allocated in this interval?", use
the half-open `(start, end]` birth selection:

```python
born = run.lifetimes(
    MemoryLifetimeSelection.born_between("before_capture", "after_capture"),
    through="after_replay",
    options=MemoryLifetimeOptions(),
)
```

This mode retains every generation born in the interval — including
transients that were created and freed between the points and are active in
neither endpoint snapshot. It
reports birth, free-request, and free-completion stacks, plus event-derived
owner-active and allocator-unreusable peaks.

When a transient allocation event carries its own pool ID (PyTorch 2.12+,
pytorch/pytorch#177717), that reported value is used directly. Otherwise its
pool is anchored temporally to the birth interval's endpoint snapshots: an
endpoint testifies only when no covering segment churn separates it from the
birth. Expandable anchoring treats all mapped ranges with the same (device,
stream, pool, segment type) key as one reservation-liveness witness. This
matches normal allocator behavior but is a heuristic: extreme fragmentation
can create multiple reservations with the same key, and snapshots expose no
reservation ID. A transient whose unreported-pool era touches neither
endpoint reports `pool[unknown]`; endpoints that disagree without covering
churn, or a birth address invisible at an endpoint without a covering segment
event, raise `MemoryReconciliationError`.

Set `lifetimes=True` in `MemoryAttributionOptions` to embed the same cohort
summary in a same-run comparison, a timeline, or both same-run change legs of a
four-point phase comparison. This consumes event history internally but does not
add allocator-event tables unless `events=True` is also requested.

Same-run comparison can use allocator addresses for lifecycle observations such
as new segments and blocks becoming active or inactive.
The comparison reports `lifecycle_confidence="exact"` only when both states
retain every segment and active-block address. Missing addresses produce an
`approximate` result and warning because equal-size entries are matched by
multiplicity. Independent runs report lifecycle as `unavailable` rather than
claiming address identity across processes.

### Cross-Run Comparison

For independent runs, use `compare_points()`:

```python
baseline = MemoryRun.load("baseline-rank0.tcgd-memory")
candidate = MemoryRun.load("candidate-rank0.tcgd-memory")

end_gap = compare_points(
    baseline["forward_end"],
    candidate["capture_end"],
    pool_mapping={MemoryPoolKey(0, (0, 1)): MemoryPoolKey(1, (0, 4))},
    device_mapping={0: 1},  # Omit when both runs use the same device index.
    attribution=MemoryAttributionOptions(stacks=True),
)
```

Points from the same run raise `MemoryOwnershipError` here — same-run analysis
belongs to `MemoryRun.compare`. A foreign snapshot passed to `probe.compare()`
raises the same error.

Cross-run identity is conservative and deterministic:

1. An explicit one-to-one `device_mapping` pairs reference and candidate devices.
2. A cross-device `pool_mapping` infers the same device pair when no explicit
   mapping claimed either device.
3. Remaining unclaimed devices with the same index pair automatically.
4. Remaining devices are reference-only or candidate-only.
5. Default pools pair within every resolved device pair. Private pools require
   an explicit one-to-one `pool_mapping`; equal raw IDs are not identity.
6. Stream IDs remain process-local, so stream observations are one-sided.
7. Address lifecycle, allocator events, and lifetimes are unavailable across runs.

An explicit device map and every pool map must agree; conflicting or many-to-one
mappings raise `ValueError`.

### Four-Point Phase Comparison

The optional four-point helper separates start-state differences from phase
change:

```python
phase = compare_phases(
    baseline.between("forward_start", "forward_end"),
    candidate.between("capture_start", "capture_end"),
    pool_mapping={MemoryPoolKey(0, (0, 1)): MemoryPoolKey(0, (0, 4))},
)
print(phase.to_text())
```

For every matched pool and metric it checks:

```text
end_gap = start_gap + candidate_change - baseline_change
```

The same four-point equation is emitted for `all`, `default`, and `private`
totals independently of private-pool ID matching.

Pool and device mappings are validated against the union of each run's phase
endpoints. A pool or device may therefore be absent at phase start or end; the
missing endpoint contributes zero allocator state. Every device phase row
retains both `baseline_device_index` and `candidate_device_index`.

## Multi-Rank Groups

Place direct per-rank bundles under one directory:

```text
baseline/
  rank-00000.tcgd-memory/
  rank-00001.tcgd-memory/
```

Then validate and summarize them as one group:

```python
baseline_group = MemoryRunGroup.load("baseline")
candidate_group = MemoryRunGroup.load("candidate")

summary = baseline_group.summary()
summary.write("reports/baseline-group")

group_phase = compare_run_group_phases(
    baseline_group,
    candidate_group,
    baseline_start="forward_start",
    baseline_end="forward_end",
    candidate_start="capture_start",
    candidate_end="capture_end",
    pool_mappings={
        0: {MemoryPoolKey(0, (0, 1)): MemoryPoolKey(1, (0, 4))},
    },
    device_mappings={0: {0: 1}},
)
group_phase.write("reports/group-phase")
```

Groups require non-null unique ranks, one run name, and matching point-label
sequences; an incomplete rank whose labels are a strict prefix of the longest
sequence (a crashed rank) is accepted with the incomplete-bundle warning. A
run without a rank, a rank at or above the declared world size, a complete
run with fewer points or a non-prefix sequence,
and conflicting non-null group IDs or world sizes are errors; missing
group IDs or world sizes, declared-but-missing ranks,
incomplete bundles, provenance differences, and metadata differences are
warnings. `compare_run_group_phases()` skips a common rank, with a warning,
when an incomplete run stopped before a requested phase endpoint. A missing
endpoint in a complete run is an error, and the comparison fails when no common
rank contains the full phase.

`rank` and `world_size` default from the `RANK`/`WORLD_SIZE`
environment variables or initialized `torch.distributed`, so torchrun processes
usually need no explicit identity arguments. `group.missing_ranks` and
`group.complete` expose rank coverage programmatically. GPU memory is never
summed across ranks. Group-summary text and HTML show allocated, reserved,
active, and requested headline metrics; JSON and CSV retain every memory
metric, its minimum and maximum rank, and the spread. Group-phase text reports
the minimum, maximum, owning ranks, and spread for `end_gap` and
`change_gap = candidate_change - baseline_change` for every phase metric. Its
JSON and CSV also retain cross-rank extrema for all five equation components.
`pool_mappings` and `device_mappings` are keyed by rank because allocator and
CUDA device identity is resolved independently for each rank.
Group loading defaults to `cache_snapshots=False`; use `True` only when repeated
allocator-state or event-payload access is worth the additional host memory.

## Reports And Bundles

Every result renders through the same family of methods — `to_text()`,
`to_dict()`, `to_html()`, and `write(output_dir)` — with display options
(`include_unchanged`, `limit`, `stack_depth`, and, for state comparisons and
timelines, the `depth` tree cutoff exposed on the CLI as `--depth`) that vary
slightly by result type. Lifetime analyses have no unchanged-row filter, and
run-group summaries take no display options at all — their `write()` accepts
only `overwrite`. See the
[API reference](api.md#memory-debug) for the exact per-result signatures.

Allocation-stack and allocator-event identity always uses complete
normalized stacks. Public attribution models expose those stacks as
`stack_frames`. `stack_key` is a location-only convenience string; text and
HTML append optional FX node metadata to the corresponding frame. `to_dict()`
and report JSON store frames as arrays, while flat CSV rows store the same data
as canonical JSON in `stack_frames_json`. `limit` and `stack_depth` affect text
and HTML only; in-memory results, JSON, and CSV retain every attribution row.
Limited HTML tables state exactly how many rows or cohorts are shown. Invalid
`limit` or `stack_depth` values raise before `write()` creates an output
directory.
Cohort charts apply `limit` to the selected cohort identities but retain every
point for those cohorts, so lines are never bent by point filtering. Structured
results and CSV continue to retain every cohort.

`write()` always creates `report.txt`, `report.json`, and `report.html`.
Report files are written atomically. `write()` rejects a nonempty directory
unless `overwrite=True`.
Overwrite removes known tcgd report artifacts from the prior report while
preserving unrelated files in the directory.

Pool-oriented results also create `allocator_scopes.csv`, `pools.csv`, and
`observations.csv`. State comparisons, timelines, and phase comparisons add
`devices.csv` whenever any device carries a sample or pools; phase reports
additionally add `device_decomposition.csv`. Attribution can add
`allocation_stack_comparisons.csv` or `events.csv`; phase reports also add
`pool_decomposition.csv` and `allocator_scope_decomposition.csv`. Lifetime
reports add `cohorts.csv`, `cohort_points.csv`, `size_histograms.csv`,
`size_outcomes.csv`, and optional transition-stack CSVs. Group summaries add
`rank_points.csv`, optional `rank_devices.csv`, and `point_aggregates.csv`.
Group phase reports add `rank_decomposition.csv`,
`rank_pool_decomposition.csv`, optional `rank_device_decomposition.csv`, and
`phase_aggregates.csv`. Attributed
group-phase reports additionally export full rank/component
`allocation_stack_comparisons.csv` and `events.csv`. Timeline HTML includes
allocated, reserved, active, requested, and optional cohort charts.
`include_unchanged=False` filters zero-change rows from text, HTML, and CSV;
JSON always retains the complete result.

Bundles use the `torch-cudagraph-debug/memory-run` schema: a `manifest.json`
beside per-point `states/NNNN.json.gz` and per-interval
`events/NNNN-NNNN.json.gz` payloads, each authenticated with SHA-256. Loading a
run reads only the manifest and compact summaries; `point.allocator_state()`
verifies and loads state lazily, and `run.validate_payloads()` bypasses lazy
caches to reread every persisted payload. `MemoryRun.load()` defaults to
`cache_snapshots=True`, while the CLI and `MemoryRunGroup.load()` use
bounded-memory loading that retains only current comparison payloads and
compact indexes. Malformed or unreadable bundles raise `MemoryBundleError`.
The format is JSON-only; use one bundle per process/rank and one writer per
bundle. The [Bundle Format](api.md#bundle-format) reference documents the
exact manifest fields and payload layout.

## CLI

The `tcgd-memory` entry point mirrors the Python analysis modes:

```bash
tcgd-memory summary rank0.tcgd-memory

tcgd-memory allocation-lifetimes rank0.tcgd-memory \
  --active-at before_capture --through after_replay \
  --output reports/lifetimes

tcgd-memory allocation-lifetimes rank0.tcgd-memory \
  --born-between before_capture after_capture \
  --through after_replay --output reports/born

tcgd-memory timeline rank0.tcgd-memory \
  --output reports/timeline

tcgd-memory compare-points rank0.tcgd-memory \
  --reference-point before_capture --candidate-point after_capture \
  --stacks --output reports/capture

tcgd-memory compare-points baseline.tcgd-memory candidate.tcgd-memory \
  --reference-point forward_end --candidate-point capture_end \
  --pool-map 0:0,1=0:0,4 --output reports/end-gap

tcgd-memory compare-phases baseline.tcgd-memory candidate.tcgd-memory \
  --baseline-start forward_start --baseline-end forward_end \
  --candidate-start capture_start --candidate-end capture_end \
  --pool-map 0:0,1=0:0,4 --output reports/phase

tcgd-memory group-summary baseline \
  --output reports/baseline-group

tcgd-memory compare-run-group-phases baseline candidate \
  --baseline-start forward_start --baseline-end forward_end \
  --candidate-start capture_start --candidate-end capture_end \
  --pool-map 0@0:0,1=0:0,4 \
  --pool-map 1@0:0,1=0:0,4 \
  --output reports/group-phase
```

Omitting the candidate bundle from `compare-points` compares two ordered points
in the reference run; `--pool-map` and `--device-map` are valid only across
independent runs. Group phases use rank-qualified forms such as
`--device-map 3@0=1`.
`timeline`, `compare-points`, `compare-phases`, and
`compare-run-group-phases` accept `--stacks`, `--events`, `--lifetimes`,
`--stack-depth`, `--limit`, and `--only-changed`. Lifetime analysis always uses
complete event history internally; `--events` independently controls event-table
output. `allocation-lifetimes` instead accepts `--stack-depth` and
`--limit`; event evidence is always required.
`summary` writes to standard output. All report commands print text and write
files only when `--output` is supplied. Reusing a nonempty output directory
requires `--overwrite`. Cross-run event and lifetime requests are rejected
with `MemoryDebugError` because allocator addresses and histories have no
cross-run identity.

## Advanced Helpers

Low-level snapshot parsers and attribution helpers are available under
`torch_cudagraph_debug.memory_debug.advanced`. They are experimental and may
change in a minor release. Their snapshot summaries (`summarize_snapshot`,
`summarize_segments`) use `Mapping[MemoryObservationKey, MemoryStats]`.

## Operational Constraints

- PyTorch allocator history remains bounded by `max_entries` during collection.
  If one point interval exceeds that ring or history starts after its first
  boundary, state remains usable but event and lifetime queries crossing that
  interval raise `MemoryHistoryTruncatedError`. Enable
  `_record_memory_history()` before the first analyzed point and keep it
  enabled: PyTorch does not expose a continuity signal that lets the recorder
  detect a mid-interval stop. Increase `max_entries` or record points more
  frequently when the ring is too small.
- Lifetime analysis is per process and rank and requires complete marker-bounded
  event history. `born_between` retains transient allocations that are active at
  neither endpoint snapshot.
- Boundary markers use the process-global allocator metadata
  (`torch.cuda.memory._set_memory_metadata`). Multiple Probe and Recorder
  objects may coexist, but marker-bearing `snapshot()` and `record_point()`
  calls must not overlap across threads or collectors. The application must not
  call `_set_memory_metadata` while either operation is taking a snapshot:
  interleaved markers misclassify event windows and can leave a stale marker in
  the global metadata.
- The metadata APIs are private to PyTorch and may be absent from a given
  build. Collection then records a warning and the point's boundary is not
  marked; state analysis stays usable, while event or lifetime queries crossing
  that interval raise `MemoryHistoryBoundaryError`.
- Marker placement is not atomic with the snapshot. Allocator events issued by
  other threads in the instant between taking a snapshot and restoring the
  metadata can fall outside both adjacent event windows; net-visible effects
  then surface as `MemoryReconciliationError`, and balanced transient churn
  from that instant is not attributable. Quiesce concurrent allocation around
  `record_point` when event or lifetime analysis matters.
- Device-wide CUDA Runtime sampling (`torch.cuda.mem_get_info`) is attempted
  during CUDA Graph capture as well as normal execution. Collection does not
  synchronize inside capture. A query failure omits only that device and adds a
  warning; allocator state from the same point remains usable. The reading is
  consecutive with the allocator snapshot, not atomic, so concurrent activity
  can move device-wide values between the two calls.

## Further Reading

- [Memory Debug API reference](api.md#memory-debug)
- [Memory Debug examples](../examples/memory_debug/README.md)
- [Complete examples index](../examples/README.md)
