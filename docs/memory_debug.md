# Memory Debug Guide

`memory_debug` records CUDA allocator snapshots at explicit points and
analyzes default and non-default pools, including CUDA Graph private pools.
This guide covers collection,
allocator-history ownership, comparisons, lifetimes, reports, bundles, and
multi-rank workflows. For exact signatures, see the
[API reference](api.md#memory-debug); for runnable programs, follow the
[Memory Debug examples](../examples/memory_debug/README.md).

## Choose An Analysis

| Question | API | Allocator history for the base result |
|---|---|---|
| What changed between two points in one run? | `run.compare()` | Not required |
| How did allocator state evolve across all points? | `run.timeline()` | Not required |
| Which allocations were born, released, or remained active? | `run.lifetimes()` | Not required for snapshot inference; full history enables exact event evidence |
| How do endpoints from independent runs differ? | `compare_points()` | Not required; state history enables stack attribution |
| Did a candidate phase grow more than a baseline phase? | `compare_phases()` | Not required; attribution is optional on each same-run growth leg |
| Which ranks differ or have the worst growth? | `MemoryRunGroup.summary()` and `compare_group_phases()` | Not required; phase attribution is optional |

Allocator history enriches an analysis; it is not required for pool, stream,
segment, block, or fragmentation state. The recorder never enables history on
the application's behalf.

## Core Usage

Import the supported public memory API from `memory_debug`:

```python
from torch_cudagraph_debug.memory_debug import (
    AttributionOptions,
    MemoryRecorder,
    MemoryRun,
    MemoryRunGroup,
    compare_group_phases,
    compare_phases,
    compare_points,
)
```

The recorder always calls `torch.cuda.memory._snapshot()`. It discovers every
pool present in the snapshot; users do not pass pool handles. `pool[0,0]` is
the default allocator pool, and other `segment_pool_id` values are private
pools such as CUDA Graph pools.

## Record A Run

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
recorder.mark("before_capture")

pool = torch.cuda.graph_pool_handle()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, pool=pool):
    tmp_a = torch.empty((4096, 4096), device="cuda")
    recorder.mark("during_capture")
    tmp_b = torch.empty((2048, 4096), device="cuda")

recorder.mark("after_capture")
graph.replay()
recorder.mark("after_replay")
run = recorder.finish()
```

The manifest records rank/group identity, user `run_metadata`, and automatic
runtime provenance such as package, Python, PyTorch, CUDA, and initialized GPU
properties. Device provenance is collected only after a real allocator
snapshot; creating a recorder does not initialize CUDA for metadata alone.

`mark()` synchronizes by default outside capture. During CUDA stream capture,
it skips synchronization because synchronization is capture-illegal, then
takes the allocator snapshot immediately.

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
    recorder.mark("start")
    # workload
    recorder.mark("end")

run = recorder.result
```

## Allocator History Is Application-Owned

The recorder never calls `torch.cuda.memory._record_memory_history()`. Pool,
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
options = AttributionOptions(
    stacks=True,
    events=True,
    on_missing="warn",
    stack_depth=2,
    limit=20,
)
comparison = run.compare("start", "end", attribution=options)
```

`on_missing="warn"` preserves state results and adds warnings.
`on_missing="error"` raises `MemoryHistoryError`.

A block's `frames` are the allocation call stack for memory still active in a
snapshot. Entries under `device_traces` are historical allocator events, and
their `frames` are event call stacks. These are separate data sources.

## Analysis Modes

The basic operations remain timeline and two-point comparison:

```python
run = MemoryRun.load("rank0.tcgd-memory")

timeline = run.timeline()
growth = run.compare("before_capture", "after_capture")
phase_range = run.between("before_capture", "after_capture")
```

The default timeline is manifest-only: it computes absolute state and deltas
without decompressing raw snapshots, and `timeline.adjacent` is empty. Passing
`AttributionOptions(stacks=True)` or `events=True` streams one raw snapshot per
point and attaches attributed adjacent comparisons.

Every point and comparison also exposes allocator-wide `all`, `default`, and
`private` totals. These totals include unmatched private pools, so a cross-run
headline does not silently become a default-pool-only number. Pool and
pool/stream rows remain available for drill-down.

Allocation cohort lifetimes are an optional same-run drill-down, not a
replacement for those modes. To answer "what was live here, and when did it
go away?", anchor the analysis at the point of interest:

```python
lifetimes = run.lifetimes(
    "before_capture",
    through="after_replay",
    attribution=AttributionOptions(
        events=True,
        on_missing="warn",
        stack_depth=4,
    ),
)
print(lifetimes.to_text())
lifetimes.write("reports/lifetimes")
```

The anchor keeps only allocation instances active at that point. Cohorts are
grouped by device, pool, and allocation call stack, with streams and allocation
size histograms retained as detail. Each release is classified as:

- `event_exact`: a matching `free_requested` event was found between point
  markers;
- `snapshot_inferred`: the block disappeared between snapshots without an
  exact event;
- still active at the final point.

Full allocator history is required for exact release timing and free call
stacks. Without it, snapshot-only lifetime analysis still works but cannot
distinguish an unobserved free-and-reallocate cycle that reuses the same address
and shape. The report describes allocator blocks, not Python tensor names or
object ownership.

To answer a different question, "what was allocated in this interval?", use
the half-open `(start, end]` birth selection:

```python
born = run.lifetimes(
    born_between=("before_capture", "after_capture"),
    through="after_replay",
)
```

With full event history, this mode retains allocations that were both created
and freed between the two points, even when they are active in neither endpoint
snapshot. It reports allocation and release stacks plus an event-derived live
byte peak. With events disabled or incomplete, it falls back to
snapshot-inferred births and warns that transient allocations may be missing.

Set `lifetimes=True` in `AttributionOptions` to embed the same top-cohort
summary in a same-run comparison, a timeline, or both same-run growth legs of a
four-point phase comparison.

Same-run comparison can use allocator addresses for lifecycle observations such
as new segments and blocks becoming active or inactive.

For independent runs, use `compare_points()`:

```python
baseline = MemoryRun.load("baseline-rank0.tcgd-memory")
candidate = MemoryRun.load("candidate-rank0.tcgd-memory")

end_delta = compare_points(
    baseline["forward_end"],
    candidate["capture_end"],
    pool_mapping={(0, 1): (0, 4)},
    attribution=AttributionOptions(stacks=True),
)
```

Cross-run rules are intentionally conservative:

- `(0,0)` matches `(0,0)` automatically.
- Private pools remain before-only/after-only even when their raw IDs happen to
  be equal, unless `pool_mapping` explicitly pairs them.
- Stream IDs are never matched across runs. Pool/stream observations remain
  before-only or after-only.
- Address lifecycle and allocator events are unavailable across runs.

The optional four-point helper separates start-state differences from phase
growth:

```python
phase = compare_phases(
    baseline.between("forward_start", "forward_end"),
    candidate.between("capture_start", "capture_end"),
    pool_mapping={(0, 1): (0, 4)},
)
print(phase.to_text())
```

For every matched pool and metric it checks:

```text
end delta = start delta + candidate growth - baseline growth
```

The same four-point equation is emitted for `all`, `default`, and `private`
totals independently of private-pool ID matching.

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

group_phase = compare_group_phases(
    baseline_group,
    candidate_group,
    baseline_start="forward_start",
    baseline_end="forward_end",
    candidate_start="capture_start",
    candidate_end="capture_end",
)
group_phase.write("reports/group-phase")
```

Groups require unique ranks, matching point-label sequences, and consistent
non-null group IDs/world sizes. Missing ranks and provenance mismatches are
warnings. Reports preserve per-rank values and show min, max, spread, and the
worst rank; GPU memory is deliberately not summed across ranks. Group loading
defaults to `cache_snapshots=False` so decompressed payloads do not accumulate
across ranks and points. Pass `cache_snapshots=True` only when repeated raw
snapshot access is worth the additional host memory.

## Reports And Bundles

Every timeline, comparison, and phase result owns:

- `to_text(include_unchanged=True)`
- `to_dict()`
- `to_html(include_unchanged=True)`
- `write(output_dir, include_unchanged=True)`

`write()` creates `report.txt`, `report.json`, `report.html`, `totals.csv`,
`pools.csv`, and `pool_streams.csv`. Attribution adds
`allocation_stacks.csv` or `events.csv`; phase reports add `phase.csv` and
`phase_totals.csv`.
Lifetime reports add `cohorts.csv`, `cohort_points.csv`,
`size_histograms.csv`, and, when present, `birth_stacks.csv` and
`release_stacks.csv`. Group reports add `rank_points.csv` plus
`point_summary.csv`, or `rank_phase.csv` plus `phase_summary.csv`.
Timeline HTML includes allocated, reserved, and optional cohort charts.
`include_unchanged=False` filters zero-change rows from text, HTML, and CSV;
JSON always retains the complete result.

Bundles use the `torch-cudagraph-debug/memory-run` schema: one
`manifest.json` plus one gzip JSON snapshot per point. Manifest, point, and
group fields are canonical; derived inactive/fragmentation values are not
stored. Loading a run reads only the manifest and compact summaries;
`point.raw_snapshot()` loads and caches the full snapshot lazily when the run
was loaded with `cache_snapshots=True`, the `MemoryRun.load()` default. The
CLI and `MemoryRunGroup.load()` use bounded-memory loading, retaining only
current comparison payloads and compact indexes. The format is JSON-only. Use
one bundle per process/rank and one writer per bundle.

## CLI

The `tcgd-memory` entry point mirrors the Python analysis modes:

```bash
tcgd-memory lifetimes rank0.tcgd-memory \
  --at before_capture --through after_replay \
  --output reports/lifetimes

tcgd-memory lifetimes rank0.tcgd-memory \
  --born-between before_capture after_capture \
  --through after_replay --output reports/born

tcgd-memory timeline rank0.tcgd-memory \
  --output reports/timeline

tcgd-memory compare rank0.tcgd-memory \
  --before before_capture --after after_capture \
  --stacks --output reports/capture

tcgd-memory compare-runs baseline.tcgd-memory candidate.tcgd-memory \
  --before forward_end --after capture_end \
  --pool-map 0,1=0,4 --output reports/end-delta

tcgd-memory compare-phases baseline.tcgd-memory candidate.tcgd-memory \
  --baseline-start forward_start --baseline-end forward_end \
  --candidate-start capture_start --candidate-end capture_end \
  --pool-map 0,1=0,4 --output reports/phase

tcgd-memory summarize-group baseline \
  --output reports/baseline-group

tcgd-memory compare-group-phases baseline candidate \
  --baseline-start forward_start --baseline-end forward_end \
  --candidate-start capture_start --candidate-end capture_end \
  --output reports/group-phase
```

Every command except `summarize-group` supports `--stacks`, `--events`,
`--lifetimes`, `--on-missing`, `--stack-depth`, `--limit`, and
`--only-changed`; `summarize-group` accepts only its group directory and
`--output`. `lifetimes` always groups cohorts by allocation stack and enables
allocator events by default; pass `--no-events` for snapshot-only inference. Cross-run
event and lifetime requests are rejected
because allocator addresses and histories have no cross-run identity.

Low-level snapshot parsers and attribution helpers are available under
`torch_cudagraph_debug.memory_debug.advanced`. They are experimental and may
change in a minor release. Snapshot summaries use
`Mapping[GroupKey, MemoryStats]`.

## Operational Constraints

- Full allocator event history is cumulative up to PyTorch's `max_entries`;
  frequent memory points can produce large bundles.
- Lifetime analysis is per process and rank. Event-backed `born_between` can
  retain transient allocations between points; snapshot-only fallback cannot
  observe allocations that are active at no recorded point.

## Further Reading

- [Memory Debug API reference](api.md#memory-debug)
- [Memory Debug examples](../examples/memory_debug/README.md)
- [Complete examples index](../examples/README.md)
