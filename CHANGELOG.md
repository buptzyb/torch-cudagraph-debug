# Changelog

## v0.2.0 - 2026-06-28

### Added

- `memory_debug`, a Python-only CUDA allocator analysis domain built around
  `MemoryRecorder`, immutable `MemoryRun` and `MemoryPoint` objects, and
  explicit point ownership.
- Timeline, same-run two-point, independent-run two-point, and four-point phase
  comparison modes with allocator-wide `all`, `default`, and `private`
  totals.
- Automatic discovery of every `segment_pool_id` with pool and raw
  `(pool, stream)` state, absolute values, signed deltas, and same-run
  lifecycle observations.
- Conservative cross-run matching with automatic default-pool pairing,
  explicit one-to-one private-pool mappings, and visible unmatched pools.
- Optional live-allocation stack and marker-delimited allocator-event
  attribution with configurable warning or error policies.
- Allocation cohort lifetime analysis with address-reuse generation splitting,
  size histograms, event-backed birth and release stacks, transient
  generations, live-byte peaks, and snapshot-inferred confidence.
- Rank-local provenance and application-owned metadata plus `MemoryRunGroup`
  summary and rank-paired phase analysis without summing memory across GPUs.
- Canonical gzip JSON run bundles with atomic writes, exact manifest fields,
  lazy snapshot loading, strict JSON validation, and path traversal protection.
- Manifest-only default timelines and un-attributed cross-run comparisons;
  attributed analyses stream each raw point once and retain bounded compact
  indexes.
- Result-owned text, nested JSON, flattened CSV, and standalone HTML reports.
  `include_unchanged=False` consistently filters text, HTML, and CSV while
  JSON remains complete.
- The `tcgd-memory` CLI with lifetime, timeline, two-run, phase, and
  multi-rank group commands.
- Public experimental allocator helpers under `memory_debug.advanced` using
  one `Mapping[GroupKey, MemoryStats]` state representation.
- `TensorProbe` with `PrintTensor`, `RecordTensor`, and `CompareTensor`
  actions, immutable `TensorProbeStatus`, and typed tensor snapshots.
- Capture-only and always-active tensor probing, gradient hook handles,
  invocation-indexed expected values, sticky comparison status, and explicit
  non-contiguous copy mode.
- TensorBoard export through `export_snapshots_to_tensorboard`.
- Categorized, runnable Tensor Debug, Memory Debug, CLI, distributed, and
  TensorBoard examples with quickstarts and complete synthetic workflows.

## v0.1.0 - 2026-05-12

Initial tensor-debug release for Linux CUDA source builds.
