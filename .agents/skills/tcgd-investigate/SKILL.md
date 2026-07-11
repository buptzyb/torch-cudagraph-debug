---
name: tcgd-investigate
description: Investigate real PyTorch CUDA Graph tensor mismatches, replay drift, NaNs, gradients, allocator growth, pool behavior, or eager-versus-graph differences with torch-cudagraph-debug. Use when debugging CUDA Graph accuracy or GPU memory issues in an application workload, conducting a fresh investigation, or analyzing persisted bundles. Do not use for implementing the debug library itself or for generic CUDA questions that do not require its public APIs.
---

# TCGD Investigation

Own one tensor or memory investigation from reproduction through a saved,
evidence-backed report. Start with the public `torch_cudagraph_debug` API and
escalate only when the remaining question cannot be answered by the package.

## Inputs

Establish these values from the request or discover them from the environment:

- the precise question and success criterion;
- the workload command, source revision, and package version;
- the baseline and candidate variants, if the question is comparative;
- the GPU environment and resource-management workflow;
- the output root.

If the user does not provide an output root, create
`/tmp/tcgd-investigation-YYYYMMDD-HHMMSS`. Do not reuse an existing directory.
Do not stop for values that can be discovered from the workload, repository, or
runtime.

Read [the workflow map](references/workflow-map.md) before choosing APIs. Use
[the report template](assets/report-template.md) for the final deliverable.

## 1. Define The Measurement

Turn the request into one explicit comparison or timeline before editing the
workload.

- Choose Tensor Debug for values, gradients, NaNs, replay drift, or numerical
  parity.
- Choose Memory Debug for allocator state, pool growth, fragmentation,
  allocation lifetime, or stack and event attribution.
- Use a Probe for an immediate local question or direct two-snapshot
  comparison.
- Use a Recorder for named points, persistence, a timeline, multiple ranks,
  multiple processes, or cross-run analysis.

Name points by semantic lifecycle boundary, such as `before_forward` and
`after_forward`, rather than generic names such as `start` and `final`. For
baseline-versus-candidate work, measure equivalent boundaries in both runs.

Start with the smallest measurement that answers the question:

- two matched endpoints for a simple delta;
- a short timeline when the growth location is unknown;
- matched start and end points in both runs for phase decomposition;
- a replay series only when the first divergence matters.

## 2. Create A Fresh Evidence Root

Use this artifact layout:

```text
<investigation-root>/
├── metadata.json
├── runs/
│   ├── <baseline>/
│   │   ├── command.txt
│   │   ├── stdout.log
│   │   ├── stderr.log
│   │   └── bundles/
│   └── <candidate>/
│       ├── command.txt
│       ├── stdout.log
│       ├── stderr.log
│       └── bundles/
└── REPORT.md
```

Record at least:

- UTC start time;
- workload and debug-package Git revisions;
- Python, PyTorch, CUDA, driver, and GPU identity;
- exact commands and relevant environment variables;
- random seeds and input-shape parameters;
- warm-up policy, repetition count, and variant execution order;
- whether each run used no instrumentation, tool instrumentation, or both;
- the requested question and selected measurement points.

Do not copy old bundles into a fresh evidence root. Existing bundles may be
analyzed only when the user explicitly asks for offline analysis; record the
mode as existing-artifact analysis in `metadata.json`.

## 3. Acquire A Suitable GPU Environment

Check for a local CUDA device and for suitable active allocations exposed by
the current environment. If a GPU resource-management skill is available, use
it and follow its allocation, reuse, and release policy. Request a new resource
only when no suitable active resource exists.

Keep baseline and candidate on equivalent hardware and software. Treat
instrumentation as a possible observer effect, especially for timing, memory,
or race-sensitive failures. When practical, run a paired no-tool/tool check on
the same node with all other inputs fixed. If that check is unnecessary or too
expensive, state that the observer effect was not measured; do not silently
assume it is absent.

## 4. Instrument With Public APIs

### Tensor Debug

- Begin with `TensorProbe` and `RecordAction` for local inspection.
- Give each semantic tensor a stable name. Compare observations by
  `(name, invocation_index)`; do not use capture order as cross-run identity.
- Use `watch_grad()` when the question concerns activation or parameter
  gradients.
- Use `TensorRecorder(execution="eager")` and
  `TensorRecorder(execution="cuda_graph")` for persisted eager-versus-graph
  work.
- Use `TensorRunGroup` and `compare_run_groups()` when rank-local tensor bundles
  must be compared without collapsing rank identity.
- Query snapshots only after the relevant replay is ordered. Pass the replay
  stream to synchronization-aware query APIs when it is available.

### Memory Debug

- Use `MemoryProbe.snapshot()` for direct state endpoints and
  `MemoryRecorder.record_point()` for named, persisted boundaries.
- Let the package discover allocator pools. Do not ask the user to provide
  pool handles.
- State comparison works without allocator history.
- If stack, allocator-event, or lifetime attribution is required, have the
  application enable `_record_memory_history()` before the allocations of
  interest. Do not make the package own this global setting.
- If required history is absent, either continue with an explicitly state-only
  result or rerun with application-owned history. Never infer missing stacks or
  events.

Keep instrumentation focused on the tensors and lifecycle phases needed for
the question. Preserve the package's unedited text or CLI output in the run
logs.

## 5. Run Controlled Variants

Predeclare the variant order. One matched baseline/candidate pair is sufficient
for a deterministic correctness question. For noisy timing or memory questions,
use repeated counterbalanced `AB/BA` (or randomized) order after equivalent
warm-up so drift and cache state are not confounded with the candidate. Record
the order and repetition number for every run. Keep code revision, model inputs,
seeds, precision, device count, and launch settings fixed except for the
intended variant.

For each run:

1. Save the exact command before execution.
2. Capture stdout and stderr without hiding live failures.
3. Verify that every requested point or observation was collected.
4. Save bundles and package-generated reports under that run directory.
5. Record whether the run completed and whether values such as loss remained
   finite.

Do not substitute a later successful run for a failed run. Preserve both and
explain the difference.

## 6. Analyze With The Package First

Use public comparison, timeline, lifetime, run-group, and report APIs or the
`tcgd-tensor` and `tcgd-memory` CLIs before writing custom parsers.

- Preserve absolute values and deltas for memory results.
- Report unchanged pools and streams when they establish the comparison
  baseline; use filtered output only for a deliberate focused view.
- For tensor results, report matched, mismatched, missing, and inconclusive
  observations and identify the first replay or point that diverged when the
  data permits it.
- For distributed memory results, keep rank-local values visible. Do not sum
  unrelated per-GPU allocator totals into a misleading global allocation.
- Treat high reserved and low active memory in a CUDA Graph private pool as
  retained pool capacity first, not as a demonstrated tensor leak. Inactive
  blocks remain unavailable to the default pool while the graph retains its
  replay-safe addresses; inspect capture peaks, sharing, fragmentation, and
  graph lifetime before assigning cause.

Copy the exact public output into the `Tool Evidence` section of `REPORT.md` or
link to the saved log containing it.

## 7. Escalate Transparently

Only after public analysis leaves a concrete unresolved question may you use:

1. raw bundle fields already persisted by the package;
2. `torch.cuda.memory._snapshot()` or another raw PyTorch API;
3. a narrowly scoped ad hoc parser;
4. PyTorch or application source inspection for semantic interpretation.

Before each escalation, state the missing fact and why the public result cannot
provide it. Save supplemental commands and output separately. If raw evidence
reveals information the package should expose, record it under `Tool Gaps`
rather than presenting it as an existing package capability.

## 8. Report And Verify

Complete `REPORT.md` with these evidence classes kept separate:

- **Tool Evidence**: direct public API or CLI output and bundle-derived reports.
- **Supplemental Analysis**: raw APIs, custom scripts, or source inspection.
- **Inference**: conclusions that combine evidence with interpretation.
- **Tool Gaps**: facts that required leaving the public workflow.

Every conclusion must identify the measured points or observations that support
it. Include absolute values before deltas, distinguish allocator reservation
from active use, and distinguish correlation from a demonstrated cause.

Before finishing, verify that all paths in the report are absolute, commands
are reproducible, required artifacts exist, and no claimed output came from an
unrecorded manual transformation. State whether an instrumentation observer
effect was measured and whether the number/order of runs supports the requested
claim.
