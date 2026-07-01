# Examples

The examples are organized by debugging workflow rather than by individual
classes. Install `torch-cudagraph-debug` into a CUDA-enabled PyTorch
environment before running them. Every script contains assertions and exits
nonzero when the demonstrated behavior is unavailable.

Generated bundles, reports, and TensorBoard logs belong outside the source
checkout. The commands below use `/tmp` for that reason.

## Tensor Debug Learning Path

| Order | Example | Main capability | Requirement |
|---:|---|---|---|
| 1 | [Tensor quickstart](tensor_debug/quickstart.py) | `PrintTensor`, capture, replay, synchronization | 1 GPU |
| 2 | [Record and compare](tensor_debug/record_and_compare.py) | snapshots, typed status, successful and failed comparisons | 1 GPU |
| 3 | [Multiple invocations](tensor_debug/multiple_invocations.py) | one probe with ordered capture slots | 1 GPU |
| 4 | [Gradient probes](tensor_debug/gradient_probes.py) | activation and parameter gradient hooks | 1 GPU |
| 5 | [Probe modes](tensor_debug/probe_modes.py) | capture-only, always-active, non-contiguous error/copy | 1 GPU |
| 6 | [Module integration](tensor_debug/module_integration.py) | configurable probe inside `torch.nn.Module` | 1 GPU |

Read [Tensor Debug examples](tensor_debug/README.md) for lifecycle rules and
expected output.

## Memory Debug Learning Path

| Order | Example | Main capability | Requirement |
|---:|---|---|---|
| 1 | [Memory quickstart](memory_debug/quickstart.py) | capture points, all/default/private totals, same-run comparison | 1 GPU |
| 2 | [Timeline and reports](memory_debug/timeline_and_reports.py) | persistence, reload, stack attribution, text/JSON/CSV/HTML | 1 GPU |
| 3 | [Attribution modes](memory_debug/attribution_modes.py) | missing-history policy, snapshot inference, full-history attribution | 1 GPU |
| 4 | [Allocation lifetimes](memory_debug/allocation_lifetimes.py) | active-at and born-between cohorts with exact events | 1 GPU |
| 5 | [Compare runs and phases](memory_debug/compare_runs_and_phases.py) | cross-run comparison, private-pool mapping, four-point phase | 1 GPU |
| 6 | [Distributed groups](memory_debug/distributed_groups.py) | rank-local bundles, group summary, group phase | 2+ GPUs |
| 7 | [CLI workflows](memory_debug/cli_workflows.sh) | all seven `tcgd-memory` commands | 1 or 2+ GPUs |

Read [Memory Debug examples](memory_debug/README.md) before enabling allocator
history in a long-running process.

## Integrations

| Example | Main capability | Extra dependency |
|---|---|---|
| [TensorBoard export](integrations/tensorboard_export.py) | export synchronized `TensorSnapshot` records | `tensorboard` |

Read the [integration notes](integrations/README.md) for dependency and output
ownership.

## Quick Commands

Run these commands from the repository root:

```bash
python examples/tensor_debug/quickstart.py
python examples/tensor_debug/record_and_compare.py
python examples/memory_debug/quickstart.py
python examples/memory_debug/timeline_and_reports.py \
  --output-dir /tmp/tcgd-timeline
python examples/memory_debug/attribution_modes.py \
  --output-dir /tmp/tcgd-attribution
python examples/memory_debug/allocation_lifetimes.py \
  --output-dir /tmp/tcgd-lifetimes
python examples/memory_debug/compare_runs_and_phases.py \
  --output-dir /tmp/tcgd-runs

torchrun --standalone --nproc-per-node=2 \
  examples/memory_debug/distributed_groups.py \
  --output-dir /tmp/tcgd-groups

bash examples/memory_debug/cli_workflows.sh \
  single /tmp/tcgd-cli-single
NPROC_PER_NODE=2 bash examples/memory_debug/cli_workflows.sh \
  distributed /tmp/tcgd-cli-distributed
```

The CLI workflow creates its own input bundles. `single` exercises timeline,
comparison, lifetime, cross-run, and phase commands. `distributed` exercises
both group summary commands and group phase comparison.

The experimental `memory_debug.advanced` module intentionally has no release
example. Its low-level parsers are documented in `docs/api.md`; the examples
above teach the stable API and CLI.
