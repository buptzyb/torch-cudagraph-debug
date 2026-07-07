# Tensor Debug Examples

Tensor examples are separated by workflow. Start with `probe/quickstart.py` for
one-process inspection. Move to `recorder/` when observations must survive the
process, carry named points, or be compared across executions. Use `cli/` for
automation over existing bundles.

All examples require Linux, CUDA, a CUDA-enabled PyTorch build, and the compiled
native extension.

## Probe Workflow

Run the Probe examples in this order:

1. `quickstart.py` records two sequential hidden tensors from one replay.
2. `snapshot_comparison.py` compares independent eager and CUDA Graph probes.
3. `replay_comparison.py` compares two ordered snapshots owned by one probe.
4. `actions.py` combines Record, Print, and Check on a changing static input.
5. `gradients.py` records activation and parameter gradients.
6. `capture_modes.py` covers eager/capture policy and non-contiguous tensors.
7. `module_integration.py` places a probe at a real module boundary.

The normal lifecycle remains visible in every example: create before capture,
capture, replay, query with the replay stream, stop replaying, then close.
`PrintAction` and `CheckAction` use CUDA host callbacks and are demonstrated for
targeted diagnosis; `RecordAction` remains the recommended default.

`actions.py` intentionally triggers and catches one `TensorCheckError`.
`capture_modes.py` intentionally catches the default non-contiguous-input error.
Both scripts exit successfully after verifying those diagnostics.

## Recorder And Run Workflow

`recorder/eager_vs_cuda_graph.py` records semantically aligned `forward` points,
loads both bundles, calls point and run comparison, and writes text, JSON, CSV,
and HTML reports.

`recorder/forward_backward.py` captures one training step and persists the
activation, loss, activation gradient, and final weight gradient in one point.
It also shows an incomplete `preview()` before `finish()`.

`recorder/replay_series.py` compares two graph replays with one eager reference.
Summary-only inputs demonstrate limited payload storage; the full output keeps
the intentional second-replay mismatch conclusive.

`recorder/distributed_run_groups.py` records eager and CUDA Graph bundles on
each rank, loads both `TensorRunGroup` objects, and writes rank-preserving
summary and comparison reports.

Recorder examples require an absent output directory because bundle writers do
not overwrite nonempty bundles:

```bash
python examples/tensor_debug/recorder/eager_vs_cuda_graph.py \
  --output-dir /tmp/tcgd-tensor-eager-vs-cg
python examples/tensor_debug/recorder/forward_backward.py \
  --output-dir /tmp/tcgd-tensor-forward-backward
python examples/tensor_debug/recorder/replay_series.py \
  --output-dir /tmp/tcgd-tensor-series
torchrun --standalone --nproc-per-node=2 \
  examples/tensor_debug/recorder/distributed_run_groups.py \
  --output-dir /tmp/tcgd-tensor-groups
```

## CLI Workflow

The CLI workflow creates matching eager/graph runs and an intentionally drifting
replay series, then executes every `tcgd-tensor` command. Exit status `1` from
the series comparison is expected and checked by the script.

```bash
bash examples/tensor_debug/cli/workflows.sh /tmp/tcgd-tensor-cli
```
