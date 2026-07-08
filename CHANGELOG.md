# Changelog

## v0.2.0 - Unreleased

### Fixed

- Allocation lifetime reconciliation now matches event-born instances to
  their allocator-rounded snapshot blocks by address with request/rounded
  size compatibility. Previously any allocation whose request size was not
  already allocator-rounded was double counted (a fabricated free plus a
  duplicate birth) whenever it survived a point boundary.
- A missing start marker no longer silently replays the entire history ring
  buffer as event evidence; the affected interval is excluded and reported.
- `event_owner_peak_bytes` no longer subtracts the free request of a block
  that was already awaiting free at the range start. Such blocks never enter
  the owner-active running sum, so the unbalanced subtraction underreported
  cohort owner peaks (down to zero) whenever a range began with pending
  cross-stream frees.
- Snapshot normalization reports absent identity and size fields (`device`,
  `segment_pool_id`, segment and block sizes, block `state`) through the
  point warnings channel, aggregated per field, instead of substituting
  defaults silently. Present-but-invalid fields still raise.
- `became_inactive_bytes` now mirrors the newly-active rule: a
  reference-active block whose `(address, size)` key is gone in the
  candidate (freed and coalesced, re-split, or in a freed segment) counts as
  became-inactive. Previously only in-place `inactive` transitions counted,
  so steady-state churn read like a leak.
- Timeline HTML charts group polylines by the pool key actually present in
  timeline rows; previously every pool collapsed into one unlabeled line.
- A truncated or corrupted gzip snapshot payload raises `MemoryBundleError`
  instead of leaking a raw `EOFError` or `zlib.error` through the CLI.

### Changed

- Allocation lifetime analysis requires complete allocator event history.
  Snapshot-only lifetime analysis and all snapshot-inferred transitions were
  removed; every reported transition is event-backed, and requests that
  predate the analysis range carry a `range_boundary` origin.
- Missing or incomplete event and lifetime evidence now raises typed errors:
  `MemoryHistoryDisabledError` (history never recorded) and
  `MemoryHistoryTruncatedError` (marker overwritten in the ring buffer). Stack
  attribution retains exact partial results with explicit coverage and raises
  only when nonempty active state has zero frame coverage. The `on_missing`
  policy, `MissingPolicy`, and the CLI `--on-missing`/`--no-events` flags were
  removed; lifetime evidence and event-table output are independent requests.
- Complete event history that cannot be reconciled with the snapshots raises
  `MemoryReconciliationError` unconditionally, aggregating every
  contradiction with per-device counts and example addresses.
- `AllocationCohort` replaces the paired `event_exact_*` /
  `snapshot_inferred_*` counters with merged `born_*`, `free_requested_*`,
  and `free_completed_*` totals; transition `confidence` was replaced by
  `origin` (`event` or `range_boundary`).

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
- Automatic discovery of every `segment_pool_id` across explicit single-device,
  device-set, and all-visible-device collection, with device-aware pool and raw
  `(device, pool, stream)` state, absolute values, signed deltas, provenance,
  and same-run lifecycle observations.
- Conservative cross-run matching with automatic default-pool pairing,
  explicit one-to-one private-pool mappings, and visible unmatched pools.
- Optional live-allocation stack attribution with explicit partial coverage,
  plus strict marker-delimited allocator-event attribution.
- Allocation cohort lifetime analysis with address-reuse generation splitting,
  owner-active versus awaiting-free point states, event-backed birth,
  free-request, and free-completion stacks, size-by-terminal-state outcomes,
  transient generations, owner/unreusable event peaks, stable full-stack cohort
  identity, lossless structured output, explicit transition origin, and `run`
  or `probe` source metadata.
- Strict allocator-state invariants plus explicit awaiting-free, inactive,
  fragmentation, segment, block, and expandable-segment metrics.
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
- Consistent allocated, reserved, active, and requested metrics at allocator,
  device/pool, and device/pool/stream scope, with structural and fragmentation details shown
  as diagnostics when they explain a change.
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
- Sibling standalone snapshot and point results plus run and replay-series
  comparisons with allclose or exact policies, strict or promoted dtypes,
  first-divergence and worst-error reporting, and explicit
  match/mismatch/inconclusive results.
- Rank-preserving `TensorRunGroup` summaries and comparisons with missing-rank,
  completeness, provenance, metadata, and point-label validation.
- Private domain collectors shared by each domain's Probe and Recorder
  workflows without introducing a synthetic cross-domain base class.
- Text, JSON, CSV, and standalone HTML tensor reports plus the `tcgd-tensor`
  `summary`, `compare-points`, `compare-runs`, `compare-point-series`,
  `group-summary`, and `compare-run-groups` CLI.
- TensorBoard export through `export_snapshots_to_tensorboard`.
- Categorized, runnable Tensor Debug, Memory Debug, CLI, distributed, and
  TensorBoard examples with quickstarts and complete synthetic workflows.
- Concise root quick starts with dedicated Tensor Debug and Memory Debug guides.
- Repository-scoped `tcgd-case-study` skill and `tcgd-debugger` custom agent for
  Codex and Claude Code, with one shared tool-first investigation workflow.
- Private-pool inactive-memory guidance and a runnable Probe example that
  distinguishes retained CUDA Graph capacity from active tensor memory.

### Changed

- Allocation-stack and allocator-event attribution now always use complete
  normalized stack identity and retain every structured row. `stack_depth` and
  `limit` affect text and HTML presentation only across comparisons, timelines,
  phase reports, and run-group phase reports. Advanced structured helpers no
  longer accept lossy depth or row-limit parameters.
- Allocation-stack, allocator-event, and lifetime models now expose complete
  structured frames. JSON stores frame arrays, CSV stores canonical
  `stack_frames_json`, and text/HTML render optional FX metadata without
  changing the location-only `stack_key` convenience value.
- `TensorProbe.close()` and `TensorRecorder.close()` now accept the same
  bool/stream/device synchronization target as tensor result queries. Closing
  an enabled probe is rejected during CUDA Graph capture.
- Eager `when="always"` probes lock one CUDA stream, allow eager work before
  capture, and reject eager work after capture establishes graph ownership.
- Direct allocation-lifetime analysis uses `MemoryLifetimeOptions`; embedded
  comparison and timeline attribution continues to use
  `MemoryAttributionOptions`.

### Fixed

- Reject malformed manifests, non-finite JSON, invalid scalar coercions,
  inconsistent ownership, invalid allocator frame fields, and partial recorder
  writes instead of accepting ambiguous persisted state. Missing per-device
  trace slots remain unavailable history rather than malformed data.
- Verify content-addressed tensor payloads even when two observations advertise
  the same digest, and return defensive tensor copies from public observations.
- Render all four core allocator metrics consistently for device/pool/stream rows and
  timeline charts, preserve stream-only stack attribution when pool aggregates
  cancel, clean stale optional report artifacts, and return absolute report paths.
- Report HTML now states shown and total counts for every limited attribution
  table instead of truncating silently. Invalid display limits and stack depths
  are rejected before rendering or creating report output directories.
- Preserve allocation identity by device, address, and generation so distinct
  same-sized blocks are not merged during lifetime analysis.
- Aggregate lifetime replay reconciliation warnings by reason and device while
  preserving total event counts and up to three example addresses, instead of
  emitting one warning per allocator event.
- Reclaim eager callback payloads, non-contiguous eager source temporaries, and
  replaced pinned staging instead of retaining them for the full Probe
  lifetime. Unsynchronized close now reports pending eager callbacks.
- Preserve exceptional TensorRecorder and MemoryRecorder sessions as terminal,
  loadable `complete=False` runs instead of marking partial data complete.

## v0.1.0 - 2026-05-12

Initial tensor-debug release for Linux CUDA source builds.
