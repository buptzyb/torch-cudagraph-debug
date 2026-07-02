# Changelog

## v0.2.0 - Unreleased

### Added

- `memory_debug`, a Python-only CUDA allocator analysis domain built around
  low-ceremony `MemoryProbe` snapshots plus complete `MemoryRecorder`,
  `MemoryRun`, and `MemoryPoint` workflows.
- Ownerless `MemoryObservation` leaves shared by standalone snapshots and
  recorded points, with explicit ownership on their containing objects.
- Standalone snapshot, same-run point, cross-run point, timeline, and four-point
  phase comparison modes with allocator-wide `all`, `default`, and `private`
  totals. Snapshot and Point comparisons are sibling result types and direct
  Probe analysis never constructs a synthetic Point or Run.
- Automatic discovery of every `segment_pool_id` with pool and raw
  `(pool, stream)` state, absolute values, signed deltas, and same-run
  lifecycle observations.
- Conservative cross-run matching with automatic default-pool pairing,
  explicit one-to-one private-pool mappings, and visible unmatched pools.
- Optional live-allocation stack and marker-delimited allocator-event
  attribution with configurable warning or error policies.
- Allocation cohort lifetime analysis with address-reuse generation splitting,
  size histograms, event-backed birth and release stacks, transient
  generations, live-byte peaks, snapshot-inferred confidence, and explicit
  `run` or `probe` source metadata.
- Rank-local provenance and application-owned metadata plus `MemoryRunGroup`
  summary and rank-paired phase analysis without summing memory across GPUs.
- Canonical gzip JSON run bundles with atomic writes, exact manifest fields,
  lazy snapshot loading, strict JSON validation, and path traversal protection.
- Manifest-only default timelines and unattributed cross-run comparisons;
  attributed analyses stream each raw point once and retain bounded compact
  indexes.
- Result-owned text, nested JSON, flattened CSV, and standalone HTML reports.
  `include_unchanged=False` consistently filters text, HTML, and CSV while
  JSON remains complete.
- The `tcgd-memory` CLI with summary, timeline, allocation-lifetime,
  point-comparison, phase-comparison, and multi-rank run-group commands.
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
- Sibling standalone snapshot and point results plus same-label run and
  replay-series comparisons with allclose or exact policies, strict or promoted
  dtypes, first-divergence and worst-error reporting, and explicit
  match/mismatch/inconclusive results.
- Private domain collectors shared by each domain's Probe and Recorder
  workflows without introducing a synthetic cross-domain base class.
- Text, JSON, CSV, and standalone HTML tensor reports plus the `tcgd-tensor`
  `summary`, `compare-points`, `compare-runs`, and `compare-point-series` CLI.
- TensorBoard export through `export_snapshots_to_tensorboard`.
- Categorized, runnable Tensor Debug, Memory Debug, CLI, distributed, and
  TensorBoard examples with quickstarts and complete synthetic workflows.
- Concise root quick starts with dedicated Tensor Debug and Memory Debug guides.

### Changed

- `TensorProbe.close()` and `TensorRecorder.close()` now accept the same
  bool/stream/device synchronization target as tensor result queries. Closing
  an enabled probe is rejected during CUDA Graph capture.
- Eager `when="always"` probes lock one CUDA stream, allow eager work before
  capture, and reject eager work after capture establishes graph ownership.

### Fixed

- Reclaim eager callback payloads, non-contiguous eager source temporaries, and
  replaced pinned staging instead of retaining them for the full Probe
  lifetime. Unsynchronized close now reports pending eager callbacks.
- Preserve exceptional TensorRecorder and MemoryRecorder sessions as terminal,
  loadable `complete=False` runs instead of marking partial data complete.

## v0.1.0 - 2026-05-12

Initial tensor-debug release for Linux CUDA source builds.
