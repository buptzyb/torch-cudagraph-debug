# Memory Debug Examples

Memory examples are separated by workflow. `probe/` answers local two-point
questions without bundles. `recorder/` preserves named allocator boundaries for
timelines, attribution, lifetime, cross-run, and distributed analysis. `cli/`
shows shell automation over persisted bundles.

## Probe Workflow

`probe/quickstart.py` records ordinary, capture-time, and post-replay allocator
state with one `MemoryProbe`. It discovers all pools present in
`torch.cuda.memory._snapshot()` and performs an immediate same-probe comparison.

`probe/private_pool_inactive.py` releases a transient capture allocation, then
shows why the graph private pool remains reserved even though most of its
capacity is inactive.

`probe/snapshot_comparison.py` compares endpoints from independent probes. Use
this pattern when the two snapshots do not share one Probe lifecycle.

None of the Probe examples require allocator history.

## Recorder And Run Workflow

Allocator history belongs to the application. Each example enables only the
mode required by its analysis and disables it during cleanup:

| Example | History mode | Analysis demonstrated |
|---|---|---|
| `timeline_and_reports.py` | state | live allocation stacks, capture points, persistence, all report formats |
| `history_requirements.py` | disabled | state comparison plus typed stack, event, and lifetime failures |
| `stack_and_event_attribution.py` | all | allocation stacks, exact events, embedded lifetimes |
| `allocation_lifetimes.py` | all | active-at and born-between cohorts with exact free transitions |
| `compare_runs_and_phases.py` | disabled | cross-run point/phase comparison and explicit private-pool mapping |
| `distributed_run_groups.py` | disabled | rank-local bundles, group extrema, group phase comparison |

Complete event history must cover intervals used for event or lifetime analysis.
An unavailable point boundary, disabled history, and a boundary overwritten by the
PyTorch ring buffer fail with distinct typed errors.
Stack attribution reports partial frame coverage and fails only when nonempty
active state has no frames. Probe and Recorder never enable or disable allocator
history on the application's behalf.

Persistent examples require an absent output directory. They print the absolute
bundle and report paths they produce. `compare_runs_and_phases.py` also writes an
explicit private-pool map; raw private-pool IDs are never assumed to match across
runs.

```bash
python examples/memory_debug/recorder/timeline_and_reports.py \
  --output-dir /tmp/tcgd-timeline
python examples/memory_debug/recorder/history_requirements.py \
  --output-dir /tmp/tcgd-history
python examples/memory_debug/recorder/stack_and_event_attribution.py \
  --output-dir /tmp/tcgd-stack-events
python examples/memory_debug/recorder/allocation_lifetimes.py \
  --output-dir /tmp/tcgd-lifetimes
python examples/memory_debug/recorder/compare_runs_and_phases.py \
  --output-dir /tmp/tcgd-runs
```

Run the distributed example through `torchrun` on a shared filesystem:

```bash
torchrun --standalone --nproc-per-node=2 \
  examples/memory_debug/recorder/distributed_run_groups.py \
  --output-dir /tmp/tcgd-groups
```

## CLI Workflow

`single` covers timeline, summary, lifetime, point, and phase commands.
`distributed` covers both run-group summaries and group phase comparison.

```bash
bash examples/memory_debug/cli/workflows.sh single /tmp/tcgd-memory-cli
NPROC_PER_NODE=2 bash examples/memory_debug/cli/workflows.sh \
  distributed /tmp/tcgd-memory-cli-distributed
```
