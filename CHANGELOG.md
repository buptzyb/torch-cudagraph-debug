# Changelog

## v0.2.0 - 2026-07-09

This release rebuilds the tensor-debug domain and introduces the
memory-debug domain.

### Breaking Changes

- The v0.1.0 tensor-debug API is replaced wholesale, with no deprecation
  shims: `CudaGraphTensorProbe` → `TensorProbe`, `TensorPrint` →
  `PrintAction`, `TensorRecord` → `RecordAction`, `TensorCompare` →
  `CheckAction`, `TensorSnapshot` → `TensorProbeSnapshot`,
  `probe.assert_ok()` → `probe.assert_check_ok()`, `attach_grad()` →
  `watch_grad()`, the `mode=` keyword → `when=`, and
  `export_records_to_tensorboard()` → `export_snapshots_to_tensorboard()`.
  Every v0.1.0 import of these names fails on upgrade.

### Fixed

- Tensor comparison and online checks preserve exact integer values and
  differences across the full `int64` range, and apply `equal_nan` without
  discarding signed-zero or NaN-payload distinctions required by exact mode.
  Online checks apply tolerance only to finite values, so infinities require an
  exact same-sign match.
- Tensor probes publish eager and captured slots transactionally, preserve
  invocation order after rejected calls, reclaim retired staging safely, remove
  registered gradient hooks on close, and reject close or query operations that
  could invalidate an active capture or in-flight copy. Capture-only probes use
  their configured device's capture state before validating inputs, so CPU and
  cross-device tensors cannot be silently ignored during active capture.
- Persisted tensor summaries use numerically stable variance, and TensorBoard
  export preserves float64 scalar precision, supports summary-only observations,
  and isolates cached tensors from writer-side mutation.
- Memory lifetime replay distinguishes provisional event-born allocations from
  snapshot-confirmed generations and rejects unwitnessed identity drift,
  duplicate active addresses, and duplicate free requests. Device comparisons
  also treat one-sided CUDA sample availability as a change without fabricating
  a delta.
- `setup.py` treats an invocation as metadata-only only when no requested
  command needs the native extension; `python setup.py sdist bdist_wheel`
  previously produced a pure-Python wheel without `_C` and
  `clean build_ext --inplace` silently skipped the rebuild.

### Added

- `memory_debug`, a Python-only CUDA allocator analysis domain built around
  low-ceremony `MemoryProbe` snapshots plus complete `MemoryRecorder`,
  `MemoryRun`, and `MemoryPoint` workflows.
- Ownerless `MemoryObservation` leaves shared by standalone snapshots and
  recorded points, with explicit ownership on their containing objects.
- Standalone snapshot, same-run point, cross-run point, timeline, and four-point
  phase comparison modes with allocator-wide `all`, `default`, and `private`
  totals. Same-identity lifecycle deltas match expandable segments by mapped
  address ranges, so in-place growth, shrink, and hole punching report only
  the mapped or unmapped bytes instead of whole-segment churn. Snapshot and
  Point comparisons are sibling result types and direct
  Probe analysis never constructs a synthetic Point or Run.
- Automatic discovery of every `segment_pool_id` across explicit single-device,
  device-set, and all-visible-device collection, with device-aware pool and raw
  `(device, pool, stream)` state, absolute values, signed deltas, provenance,
  and same-run lifecycle observations.
- Conservative cross-run matching with automatic default-pool pairing,
  explicit one-to-one private-pool mappings, and visible unmatched pools.
- Optional live-allocation stack attribution with explicit partial coverage,
  plus strict marker-delimited allocator-event attribution and typed unavailable,
  boundary, truncated, and reconciliation failures. OOM trace entries keep the
  device free-byte count in `device_free_bytes` instead of `addr` and report
  pool-attribution confidence `not_applicable`.
- Allocation cohort lifetime analysis requiring complete allocator event history,
  with address-reuse generation splitting,
  owner-active versus awaiting-free point states, event-backed birth,
  free-request, and free-completion stacks, size-by-terminal-state outcomes,
  transient generations, owner/unreusable event peaks, stable full-stack cohort
  identity, lossless structured output, explicit transition origin, and `run`
  or `probe` source metadata.
- Strict allocator-state invariants plus explicit awaiting-free, inactive,
  fragmentation, segment, block, and expandable-segment metrics, including
  `expandable_inactive_bytes` to distinguish inactive capacity in expandable
  segments from inactive capacity in native segments.
- Device-wide CUDA Runtime memory sampling: snapshots and points attempt one
  `torch.cuda.mem_get_info` reading per selected device, including points inside
  CUDA Graph capture. Comparisons, timelines, phases, and run groups report
  CUDA used/free/total, current-process allocator reserved, and device-global
  `cuda_allocator_residual_bytes`; nonzero total-capacity deltas remain explicit.
  Per-device query failures preserve allocator state and add a warning. Reports
  add device sections and device-specific CSV artifacts.
- Explicit one-to-one cross-run CUDA device mapping, device pairs inferred from
  pool mappings, unclaimed same-index fallback, paired endpoint indices in
  phase and rank reports, and `--device-map` CLI forms.
- Rank-local provenance and application-owned metadata plus `MemoryRunGroup`
  summary and rank-paired phase analysis without summing memory across GPUs.
- Canonical gzip JSON run bundles with atomic writes, exact manifest fields,
  point-owned `states/NNNN.json.gz` allocator state, adjacent
  `events/NNNN-NNNN.json.gz` evidence, independent lazy caches, strict JSON
  validation, and path traversal protection.
- Manifest-only default timelines and unattributed cross-run comparisons;
  attributed analyses load only required allocator states and point-owned event
  chunks while retaining bounded compact indexes.
- Result-owned text, nested JSON, flattened CSV, and standalone HTML reports.
  Comparison and timeline text and HTML share one device tree:
  `CUDA used = residual + allocator reserved`, allocator = sum of pools, and
  pool = sum of streams. Interior nodes lead with `reserved:` and sparse
  diagnostics. `depth` (`device`/`pool`/`stream`, exposed as `--depth` on
  `timeline` and `compare-points`) truncates text and HTML only.
  `include_unchanged=False` uses one bottom-up pruning pass for text, HTML, and
  flat CSV parent-context rows; JSON remains complete.
- Consistent allocated, reserved, active, and requested metrics at allocator,
  device/pool, and device/pool/stream scope, with structural and
  fragmentation details shown as diagnostics when they explain a change.
- The `tcgd-memory` CLI with the `summary`, `timeline`,
  `allocation-lifetimes`, `compare-points`, `compare-phases`,
  `group-summary`, and `compare-run-group-phases` commands.
- Public experimental allocator helpers under `memory_debug.advanced` using
  one `Mapping[MemoryObservationKey, MemoryStats]` state representation.
- `TensorProbe` with `PrintAction`, `RecordAction`, `CheckAction`, immutable
  `TensorCheckStatus`, and one aggregate `TensorProbeSnapshot` per queried replay.
- Capture-only and always-active tensor probing, gradient hook handles,
  invocation-indexed expected values, sticky check status, and explicit
  non-contiguous copy mode.
- A per-probe CUDA `int64` replay counter shared by record, print, check, and
  status results, with 1-based graph replay indices, explicit device selection,
  bool/stream/device query synchronization, query-time transfer for
  callback-free recording, and callback-counter staging reuse.
- `TensorRecorder` with explicit eager and CUDA Graph execution modes, named
  observations, ordered points, gradient observation hooks, and one shared
  native recording session across every logical probe in a graph.
- Immutable `TensorRun`, `TensorPoint`, and `TensorObservation` models with
  strict content-addressed bundles, full or summary payloads, SHA-256
  verification, numerical summaries, deduplication, and lazy CPU-only loading.
- Sibling standalone snapshot and point results plus run and replay-series
  comparisons with allclose or exact policies, strict or promoted dtypes,
  first-divergence and worst-error reporting, and explicit
  match/mismatch/inconclusive results.
- Rank-preserving `TensorRunGroup` summaries and comparisons with missing-rank,
  completeness, provenance, metadata, and point-label validation. Both group
  domains accept a crashed rank's incomplete strict-prefix bundle with a
  warning, and comparisons report an incomplete run's missing crash tail as
  `inconclusive` — differences on shared points still win as `mismatch`.
- Private domain collectors shared by each domain's Probe and Recorder
  workflows without introducing a synthetic cross-domain base class.
- Text, JSON, CSV, and standalone HTML tensor reports plus the `tcgd-tensor`
  `summary`, `compare-points`, `compare-runs`, `compare-point-series`,
  `group-summary`, and `compare-run-groups` CLI.
- TensorBoard export through `export_snapshots_to_tensorboard`.
- Categorized, runnable Tensor Debug, Memory Debug, CLI, distributed, and
  TensorBoard examples with quickstarts and complete synthetic workflows.
- Concise root quick starts with dedicated Tensor Debug and Memory Debug guides.
- Repository-scoped `tcgd-investigate` skill and `tcgd-debugger` custom agent for
  Codex and Claude Code, with one shared tool-first investigation workflow.
- Private-pool inactive-memory guidance and a runnable Probe example that
  distinguishes retained CUDA Graph capacity from active tensor memory.

## v0.1.0 - 2026-05-12

First public 0.1.0 release of `torch-cudagraph-debug`.

### Added

- `CudaGraphTensorProbe`, a CUDA Graph tensor probe that returns its input tensor
  unchanged while capturing debug-side device-to-host copies and host callbacks.
- `TensorPrint` for compact native CPU-side printing from replay snapshots.
- `TensorRecord` for callback-free latest CPU snapshots per logical probe slot,
  exposed as `TensorSnapshot` objects in Python.
- Per-invocation pinned host staging for all actions, so callback-backed probes
  do not reuse one shared host buffer across logical slots.
- `TensorCompare` for replay-time comparison against per-invocation CPU tensor
  or NumPy ground truth lists, with sticky mismatch reporting via
  `probe.assert_ok()`.
- `invocation_index` on recorded snapshots and compare status so callers can
  distinguish repeated calls of the same probe within one replay.
- Single-capture probe ownership: one probe may be called multiple times inside
  one capture, slots may have different tensor metadata, and another graph
  capture must use a new probe.
- `CudaGraphTensorProbe.attach_grad()` for activation, output, and parameter
  gradient probing through PyTorch autograd hooks.
- `mode="capture"` as the default warmup-transparent mode and `mode="always"`
  as an explicit eager/debug escape hatch.
- `non_contiguous="copy"` for opt-in debug-only contiguous copies of
  non-contiguous inputs.
- `torch_cudagraph_debug.tensor_debug.postprocess.export_records_to_tensorboard()`
  for scalar and optional histogram summaries from recorded snapshots.
- Examples for basic tensor debugging, record/compare workflows, module-internal
  hidden tensor probes, gradient probe patterns, Python-side replay snapshot
  collection, and TensorBoard export.

### Compatibility Notes

- Linux CUDA environments only.
- Source builds only; prebuilt wheels are intentionally not provided for v0.1.
- Build against the CUDA-enabled PyTorch installation in the target runtime with
  `pip install --no-build-isolation`.
- Source-tree imports and all-disabled probes can run without the native
  extension, but enabled probes require a native extension built against
  CUDA-enabled PyTorch.
- Probe nodes are inline graph dependencies and can introduce large GPU bubbles;
  they are intended for correctness debugging, not performance measurement.
