# Memory Debug Examples

Read the [Memory Debug guide](../../docs/memory_debug.md) for collection,
history, attribution, and report semantics.

`quickstart.py` uses one `MemoryProbe` for standalone snapshots and immediate
two-point comparison. `snapshot_comparison.py` compares endpoints captured by
independent probes. The remaining examples use `MemoryRecorder` to collect labeled
points into immutable runs for persistence, timeline, lifetime, phase, and
multi-rank analysis.

## Allocator History

The application owns `torch.cuda.memory._record_memory_history()` and must
choose its overhead explicitly:

| Example | History mode | Information used |
|---|---|---|
| `quickstart.py` | disabled | pool/stream state and same-run lifecycle |
| `snapshot_comparison.py` | disabled | cross-probe pool/stream state |
| `timeline_and_reports.py` | state | live-block allocation stacks |
| `attribution_modes.py` | disabled, then all | warn/error policy, snapshot inference, and exact events |
| `allocation_lifetimes.py` | all | allocation/free events and exact lifetime transitions |
| `compare_runs_and_phases.py` | disabled | compact state and explicit private-pool mapping |
| `distributed_run_groups.py` | disabled | rank-local state and cross-rank extrema |

History must be enabled before allocations whose stacks or events matter. The
Probe and Recorder never enable or disable it on the application's behalf.

`attribution_modes.py` is the policy guide. Its first bundle shows that state
comparison still works without history, then contrasts `on_missing="warn"`
with `on_missing="error"`. It also shows the explicit limitation of
snapshot-inferred births and releases: allocations that start and end between
points are invisible. Its second bundle enables full history before the
workload and requests stacks, events, and embedded lifetimes from ordinary
`compare()` and `timeline()` calls.

## Output And Reuse

Scripts that persist data require `--output-dir` and refuse to reuse an existing
path. A bundle is a `*.tcgd-memory` directory. Reports contain text, JSON, CSV,
and HTML generated from the same result object.

Run the attribution policy comparison with:

```bash
python examples/memory_debug/attribution_modes.py \
  --output-dir /tmp/tcgd-attribution
```

`compare_runs_and_phases.py` writes `pool-map.txt` after discovering the private
pool created by each scenario. The Python API and CLI examples both consume
that explicit mapping; identical raw private-pool IDs are never assumed to be
the same across runs. The example seeds each mapped pool before `phase_start`
because phase comparison applies the mapping at both start and end points.

Use `distributed_run_groups.py` through `torchrun` on a shared filesystem. Each rank
writes one direct child bundle. Group reports retain per-rank values and show
min/max/spread; they never sum per-GPU memory.

## CLI

`cli_workflows.sh single OUTPUT_ROOT` generates single-rank bundles and runs
`summary`, `timeline`, `allocation-lifetimes`, `compare-points`, and `compare-phases`.

`cli_workflows.sh distributed OUTPUT_ROOT` launches the group producer with
`NPROC_PER_NODE` processes and runs `summarize-run-group` for both scenarios plus
`compare-run-group-phases`.
