# Examples

Examples follow the same two workflows as the public API:

- `Probe` examples provide quick, local inspection without bundles.
- `Recorder -> Run` examples preserve named points for reports and offline analysis.
- `CLI` examples generate their own bundles and exercise the complete command surface.

Every script uses a deterministic workload, asserts the behavior it demonstrates,
and exits nonzero when that behavior is unavailable. Install
`torch-cudagraph-debug` in a CUDA-enabled PyTorch environment before running
them. Persistent outputs belong outside the source checkout; commands below use
`/tmp`.

## Tensor Debug

Read the [Tensor Debug example guide](tensor_debug/README.md) for lifecycle and
output details.

### Probe Workflow

| Order | User question | Example | Primary API |
|---:|---|---|---|
| 1 | What did sequential hidden states contain on the latest replay? | [Quickstart](tensor_debug/probe/quickstart.py) | `TensorProbe`, `RecordAction`, `snapshot()` |
| 2 | Do eager and graph execution produce the same hidden value? | [Snapshot comparison](tensor_debug/probe/snapshot_comparison.py) | `compare_snapshots()` |
| 3 | Which values changed between two replays of one graph? | [Replay comparison](tensor_debug/probe/replay_comparison.py) | `TensorProbe.compare()` |
| 4 | How do online print, record, and expected-value checks behave? | [Probe actions](tensor_debug/probe/actions.py) | `PrintAction`, `RecordAction`, `CheckAction` |
| 5 | What are an activation gradient and a final parameter gradient? | [Gradients](tensor_debug/probe/gradients.py) | `TensorProbe.watch_grad()` |
| 6 | When does a probe run, and how are non-contiguous tensors handled? | [Capture modes](tensor_debug/probe/capture_modes.py) | `when`, `non_contiguous` |
| 7 | Where should a probe be placed in a real module? | [Module integration](tensor_debug/probe/module_integration.py) | `TensorProbe` in `torch.nn.Module` |

### Recorder And Run Workflow

| Order | User question | Example | Primary API |
|---:|---|---|---|
| 1 | Does a complete eager run match a CUDA Graph run? | [Eager vs CUDA Graph](tensor_debug/recorder/eager_vs_cuda_graph.py) | `TensorRecorder`, `compare_points()`, `compare_runs()` |
| 2 | How are forward activations and backward gradients persisted together? | [Forward and backward](tensor_debug/recorder/forward_backward.py) | `observe()`, `watch_grad()`, `snapshot_run()` |
| 3 | On which replay did drift first appear? | [Replay series](tensor_debug/recorder/replay_series.py) | summary/full payloads, `compare_point_series()` |

### CLI Workflow

| User question | Example | Commands covered |
|---|---|---|
| How are persisted tensor runs inspected in automation? | [CLI workflows](tensor_debug/cli/workflows.sh) | all four `tcgd-tensor` commands |

## Memory Debug

Read the [Memory Debug example guide](memory_debug/README.md) before enabling
allocator history in a long-running process.

### Probe Workflow

| Order | User question | Example | Primary API |
|---:|---|---|---|
| 1 | Which pools and streams grew around graph capture and replay? | [Quickstart](memory_debug/probe/quickstart.py) | `MemoryProbe`, `snapshot()`, `compare()` |
| 2 | How are independently collected allocator endpoints compared? | [Snapshot comparison](memory_debug/probe/snapshot_comparison.py) | `compare_snapshots()` |

### Recorder And Run Workflow

| Order | User question | Example | History mode |
|---:|---|---|---|
| 1 | How is a capture timeline persisted, loaded, and exported? | [Timeline and reports](memory_debug/recorder/timeline_and_reports.py) | state |
| 2 | What remains available when allocator history is disabled? | [History requirements](memory_debug/recorder/history_requirements.py) | disabled |
| 3 | Which stack and allocator events caused growth? | [Stack and event attribution](memory_debug/recorder/stack_and_event_attribution.py) | all |
| 4 | Which allocations survived or were born between named points? | [Allocation lifetimes](memory_debug/recorder/allocation_lifetimes.py) | all |
| 5 | How do baseline and candidate phases differ across private pools? | [Compare runs and phases](memory_debug/recorder/compare_runs_and_phases.py) | disabled |
| 6 | How are per-rank runs summarized without summing GPU memory? | [Distributed run groups](memory_debug/recorder/distributed_run_groups.py) | disabled, 2+ GPUs |

### CLI Workflow

| User question | Example | Commands covered |
|---|---|---|
| How are memory bundles analyzed from shell automation? | [CLI workflows](memory_debug/cli/workflows.sh) | all seven `tcgd-memory` commands |

## Integrations

| User question | Example | Extra dependency |
|---|---|---|
| How are synchronized probe values exported to TensorBoard? | [TensorBoard export](integrations/tensorboard_export.py) | `tensorboard` |

Read the [integration notes](integrations/README.md) for output ownership.

The experimental `memory_debug.advanced` module intentionally has no release
example. Its low-level parsers are documented in `docs/api.md`; examples teach
the supported user workflows and CLI.
