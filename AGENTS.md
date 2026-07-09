# Repository Instructions

## Project Scope

`torch-cudagraph-debug` diagnoses PyTorch CUDA Graph tensor values and CUDA
allocator memory behavior. The package has two public domains:

- `tensor_debug` for tensor values, gradients, replay drift, and eager versus
  CUDA Graph comparisons.
- `memory_debug` for allocator state, timelines, allocation lifetimes, stack
  attribution, and cross-run comparisons.

Communicate in the user's language when practical. Keep code, comments,
documentation, reports, and persisted metadata in English.

## Tool-First Investigation Contract

When investigating tensor or memory behavior in an application:

1. Reproduce the question with the public `torch_cudagraph_debug` Probe or
   Recorder workflow before using raw PyTorch allocator APIs or ad hoc parsers.
2. Use a Probe for local inspection or a direct two-snapshot comparison. Use a
   Recorder when the investigation needs named points, persistence, timelines,
   multiple processes, or cross-run analysis.
3. Run fresh workload variants unless the user explicitly asks to analyze an
   existing bundle. Do not silently mix artifacts from earlier investigations.
4. Preserve the exact public API or CLI output, commands, environment, package
   version, source revision, logs, and bundle paths used for the conclusion.
5. Align comparisons by semantic execution boundary. A convenient `final`
   point is not a valid substitute for matching the same lifecycle phase.
6. Treat allocator history as application-owned. The application must enable
   `torch.cuda.memory._record_memory_history()` before allocations of interest
   when stack, event, or lifetime attribution is required. The debug package
   must not enable it implicitly.
7. Use `torch.cuda.memory._snapshot()`, custom scripts, or source inspection
   only when the public report cannot answer the remaining question. State why
   the escalation was necessary.
8. Separate final reporting into public tool evidence, supplemental analysis,
   inference, and product gaps. Never describe supplemental raw-snapshot work
   as output produced by this package.

Use the `tcgd-investigate` skill for the complete investigation procedure. Use
the `tcgd-debugger` custom agent when the investigation benefits from an
isolated agent that owns reproduction, collection, analysis, and reporting end to end.

## Development Rules

- Read `docs/architecture.md` and the relevant domain guide before changing
  package behavior.
- Keep Probe and Recorder as sibling public workflows over private collectors.
- Keep generated investigation artifacts outside the source checkout, under a
  user-provided directory or `/tmp` by default.
- Fix validation failures at their source. Do not bypass checks, suppress test
  failures, or use destructive Git commands to make a run pass.
- Run focused tests while developing and the full test suite before declaring
  a change ready. GPU behavior must be verified on a CUDA-enabled system.
- Do not commit, amend, push, or rewrite history unless the user explicitly
  requests that Git operation.

## GPU Resources

Use the execution environment's existing GPU resource-management workflow when
one is available. Reuse a suitable active allocation before requesting a new
one, obey that workflow's lifecycle policy, and do not claim GPU validation
from CPU-only or skipped tests. Repository instructions must remain portable
and must not embed commands for a private cluster.
