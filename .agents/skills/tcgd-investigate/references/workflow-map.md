# Workflow Map

Use this map after the user question is precise. Start from the linked example
and adapt only the points, tensors, or workload integration needed by the case.

## Tensor Debug

| Question | Workflow | Public API | Starting example |
|---|---|---|---|
| What value did a hidden tensor have on the latest replay? | Probe | `TensorProbe`, `RecordAction`, `snapshot()` | `examples/tensor_debug/probe/quickstart.py` |
| Do two local executions produce the same tensor? | Probe | `compare_snapshots()` | `examples/tensor_debug/probe/snapshot_comparison.py` |
| Which replay first changed? | Probe or Recorder | `TensorProbe.compare()` or `compare_point_series()` | `examples/tensor_debug/probe/replay_comparison.py`, `examples/tensor_debug/recorder/replay_series.py` |
| What gradient reached an activation or parameter? | Probe or Recorder | `watch_grad()` | `examples/tensor_debug/probe/gradients.py`, `examples/tensor_debug/recorder/forward_backward.py` |
| Does a complete eager execution match CUDA Graph execution? | Recorder | `TensorRecorder`, `compare_points()`, `compare_runs()` | `examples/tensor_debug/recorder/eager_vs_cuda_graph.py` |
| How do corresponding tensor ranks compare? | Recorder | `TensorRunGroup`, `compare_run_groups()` | `examples/tensor_debug/recorder/distributed_run_groups.py` |
| How should persisted tensor bundles be inspected in automation? | CLI | `tcgd-tensor` | `examples/tensor_debug/cli/workflows.sh` |

Use `RecordAction` as the default action. Add `PrintAction` or `CheckAction`
only when online output or online expected-value validation is the actual
requirement.

## Memory Debug

| Question | Workflow | Public API | Starting example |
|---|---|---|---|
| Which allocator pools and streams changed between two points? | Probe | `MemoryProbe.snapshot()`, `MemoryProbe.compare()` | `examples/memory_debug/probe/quickstart.py` |
| Why is a CUDA Graph private pool mostly inactive but still reserved? | Probe | `MemoryStats.inactive_bytes` | `examples/memory_debug/probe/private_pool_inactive.py` |
| How do independently collected endpoints differ? | Probe | `compare_snapshots()` | `examples/memory_debug/probe/snapshot_comparison.py` |
| Where in a phase did memory grow? | Recorder | `MemoryRecorder`, `MemoryRun.timeline()` | `examples/memory_debug/recorder/timeline_and_reports.py` |
| Which stacks or allocator events caused growth? | Recorder plus application-owned history | `MemoryAttributionOptions` | `examples/memory_debug/recorder/stack_and_event_attribution.py` |
| Which allocations survived or were born in a range? | Recorder plus application-owned history | `MemoryRun.lifetimes()`, `MemoryLifetimeSelection` | `examples/memory_debug/recorder/allocation_lifetimes.py` |
| How do matched baseline and candidate phases differ? | Recorder | `compare_phases()` | `examples/memory_debug/recorder/compare_runs_and_phases.py` |
| How do corresponding ranks differ? | Recorder | `MemoryRunGroup`, `compare_run_group_phases()` | `examples/memory_debug/recorder/distributed_run_groups.py` |
| How should persisted memory bundles be inspected in automation? | CLI | `tcgd-memory` | `examples/memory_debug/cli/workflows.sh` |

Memory collection discovers every allocator pool represented by the PyTorch
snapshot. Never require application code to pass private-pool handles.

## Choosing Measurement Points

- For a two-point delta, place endpoints immediately around the suspected
  phase.
- For eager versus CUDA Graph, align semantic work: for example, compare after
  all eager forward layers with the point after an aligned graph replay has
  completed the corresponding forward work. Capture constructs the replay; it
  is not itself the candidate execution to compare.
- For capture growth, separate pre-capture state, capture construction, graph
  replay, backward, and optimizer initialization when the workload contains
  those phases.
- Add timeline points only where they can distinguish competing explanations.
  More points are not automatically better.

## History Decision

| Required result | Allocator history |
|---|---|
| Allocated, reserved, active, requested, awaiting-free, inactive, pool, stream, fragmentation, expandable segments | Not required |
| Live-allocation stack attribution | Required before allocations of interest |
| Allocator event attribution | Required before events of interest |
| Event-backed birth and release lifetimes | Required before the analyzed range |

If history was not enabled in time, preserve the state-only result and state
which attribution questions remain unanswered.
