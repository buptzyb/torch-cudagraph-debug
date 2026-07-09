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
event and lifetime analysis require the corresponding complete history.
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
- `internal_fragmentation_bytes = active_bytes - requested_bytes` is allocator
  rounding inside active or awaiting-free blocks.

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

`snapshot()` returns one complete allocator state. `probe.compare()` validates
that both snapshots belong to that probe and are in increasing index order.
Top-level `compare_snapshots(reference, candidate, pool_mapping=...)` also
compares independent probes, which is useful for two independently
instrumented workloads in one process (for example an eager baseline against
a CUDA Graph run). Private pools stay unmatched across independent probes
unless `pool_mapping` explicitly pairs them, exactly like cross-run
`compare_points`.

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
`MemoryRun`; analysis belongs to the run and result objects:

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
describe the evidence that backed it.

A block's `frames` are the allocation call stack for memory still active in a
snapshot. Entries under `device_traces` are historical allocator events, and
their `frames` are event call stacks. These are separate data sources.
Recorder bundles remove cumulative `device_traces` from point state and
preserve only each adjacent interval's raw event mappings.

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

Allocation cohort lifetimes are an optional focused same-run analysis, not a
replacement for those modes. To answer "what was live here, and when did it
go away?", anchor the analysis at the point of interest:

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

Free request and completion are reported independently, each with its own
stack table. Transitions inside the analyzed range are backed by allocator
events. When a block is already awaiting free at the first point, its earlier
request is represented once with `origin="range_boundary"` rather than assigned
to a guessed interval. `free_completed` means allocator-reusable; it does not
mean the segment was returned to CUDA or that pool `reserved_bytes` decreased.

Lifetime analysis requires complete allocator event history for the analyzed
range and reconciles the event stream against every allocator state. When event
replay finds contradictions, the raised `MemoryReconciliationError` summarizes
each reason and device with the total count and up to three example
addresses. The report describes
allocator blocks, not Python tensor names or object ownership.

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

For independent runs, use `compare_points()`:

```python
baseline = MemoryRun.load("baseline-rank0.tcgd-memory")
candidate = MemoryRun.load("candidate-rank0.tcgd-memory")

end_gap = compare_points(
    baseline["forward_end"],
    candidate["capture_end"],
    pool_mapping={MemoryPoolKey(0, (0, 1)): MemoryPoolKey(0, (0, 4))},
    attribution=MemoryAttributionOptions(stacks=True),
)
```

Cross-run rules are intentionally conservative:

- Default pools on the same device match automatically.
- Private pools remain reference-only/candidate-only even when raw IDs happen
  to be equal, unless `pool_mapping` explicitly pairs `MemoryPoolKey` objects.
- Stream IDs are process-local CUDA handles with no stable cross-run identity,
  so pool/stream observations remain reference-only or candidate-only.
- Address lifecycle and allocator events are unavailable across runs.

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

A private-pool mapping is validated against the union of each run's phase
endpoints. A pool may therefore be absent at phase start or end; the missing
endpoint contributes zero state, preserving phases that create or destroy the
pool.

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
        0: {MemoryPoolKey(0, (0, 1)): MemoryPoolKey(0, (0, 4))},
    },
)
group_phase.write("reports/group-phase")
```

Groups require non-null unique ranks, one run name, and matching point-label
sequences. A run without a rank, a rank at or above the declared world size,
and conflicting non-null group IDs or world sizes are errors; missing
group IDs or world sizes, declared-but-missing ranks,
incomplete bundles, provenance differences, and metadata differences are
warnings. `rank` and `world_size` default from the `RANK`/`WORLD_SIZE`
environment variables or initialized `torch.distributed`, so torchrun processes
usually need no explicit identity arguments. `group.missing_ranks` and
`group.complete` expose rank coverage programmatically. Reports preserve
per-rank values and show min, max, spread, and the worst rank; GPU memory is deliberately not summed across ranks.
`pool_mappings` is keyed by rank because private-pool identity is rank-local.
Group loading defaults to `cache_snapshots=False`; use `True` only when repeated
allocator-state or event-payload access is worth the additional host memory.

## Reports And Bundles

State comparisons, timelines, phase comparisons, and group phase comparisons
provide:

- `to_text(include_unchanged=True, limit=None, stack_depth=None)`
- `to_dict()`
- `to_html(include_unchanged=True, limit=None, stack_depth=None)`
- `write(output_dir, include_unchanged=True, limit=None, stack_depth=None, overwrite=False)`

Lifetime analyses have no unchanged-row filter and provide `to_text(limit=None,
stack_depth=None)`, `to_dict()`, `to_html(limit=None, stack_depth=None)`, and
`write(output_dir, limit=None, stack_depth=None, overwrite=False)`. Run-group
summaries provide the corresponding parameter-free render methods plus
`overwrite` on `write()`.

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
`observations.csv`. Attribution can add `allocation_stack_comparisons.csv` or
`events.csv`; phase reports add `pool_decomposition.csv` and
`allocator_scope_decomposition.csv`. Lifetime reports add `cohorts.csv`,
`cohort_points.csv`, `size_histograms.csv`, `size_outcomes.csv`, and, when
present, `birth_stacks.csv`, `free_request_stacks.csv`, and
`free_completion_stacks.csv`. Group reports add either `rank_points.csv`
and `point_aggregates.csv`, or `rank_decomposition.csv`,
`rank_pool_decomposition.csv`, and `phase_aggregates.csv`. Attributed
group-phase reports additionally export full rank/component
`allocation_stack_comparisons.csv` and `events.csv`. Timeline HTML includes
allocated, reserved, active, requested, and optional cohort charts.
`include_unchanged=False` filters zero-change rows from text, HTML, and CSV;
JSON always retains the complete result.

Bundles use the `torch-cudagraph-debug/memory-run` schema: `manifest.json`, one
`states/NNNN.json.gz` allocator-state file per point, and one
`events/NNNN-NNNN.json.gz` event-evidence file per adjacent interval. Point zero
has no event file. State files omit cumulative `device_traces`; event files keep
the raw mappings for only their interval. Manifest, point, and observation fields
are canonical; derived awaiting-free, inactive, and fragmentation values are not
stored. Each point manifest authenticates its state and event gzip payload with
SHA-256. Loading a run still reads only the manifest and compact summaries.
`point.allocator_state()` loads the immutable state lazily, verifies its digest,
and rejects disagreement between recomputed state summaries and the manifest.
`run.validate_payloads()` bypasses lazy caches and rereads every persisted state
and event payload. Ordinary access uses separate state and event caches when
the run was loaded with `cache_snapshots=True`, the `MemoryRun.load()` default.
The CLI and `MemoryRunGroup.load()` use bounded-memory loading, retaining only
current comparison payloads and compact indexes. The format is JSON-only. Use
one bundle per process/rank and one writer per bundle.

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
in the reference run; `--pool-map` is valid only across independent runs.

`timeline`, `compare-points`, `compare-phases`, and
`compare-run-group-phases` accept `--stacks`, `--events`, `--lifetimes`,
`--stack-depth`, `--limit`, and `--only-changed`. Lifetime analysis always uses
complete event history internally; `--events` independently controls event-table
output. `allocation-lifetimes` instead accepts `--stack-depth` and
`--limit`; event evidence is always required.
`summary` writes to standard output. All report commands print text and write
files only when `--output` is supplied. Reusing a nonempty output directory
requires `--overwrite`. Cross-run event and lifetime requests are rejected
because allocator addresses and histories have no cross-run identity.

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
  detect a mid-interval stop. Increase `max_entries` or record real points more
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
- Marker placement is not atomic with the snapshot. Allocator events issued by
  other threads in the instant between taking a snapshot and restoring the
  metadata can fall outside both adjacent event windows; net-visible effects
  then surface as `MemoryReconciliationError`, and balanced transient churn
  from that instant is not attributable. Quiesce concurrent allocation around
  `record_point` when event or lifetime analysis matters.

## Further Reading

- [Memory Debug API reference](api.md#memory-debug)
- [Memory Debug examples](../examples/memory_debug/README.md)
- [Complete examples index](../examples/README.md)
