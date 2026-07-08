# Changelog

## v0.2.0 - Unreleased

### Fixed

- Tensor comparison and native `CheckAction` now preserve exact integer
  differences and values across the full `int64` range instead of converting
  operands to `float64` before subtraction or formatting.
- `TensorRecorder` commits a capture invocation only after the private
  collector accepts it, and capture-external calls determine scope from the
  recorder device before validating the input tensor.
- Memory bundles now authenticate state and event payloads with SHA-256 and
  validate raw state summaries against manifest observations on lazy access;
  `MemoryRun.validate_payloads()` performs an explicit full-bundle check that
  rereads persisted payloads instead of trusting prior lazy-load caches.
- Strict persisted models reject duplicate JSON keys, overflowing numeric
  coercions, impossible tensor summaries, non-string identities, and invalid
  Run completion or timestamp order before those states can be written.
- Block-address inference now resumes from every explicit block address.
  Same-identity memory comparisons report address lifecycle confidence as
  `exact` or `approximate`; independent comparisons report `unavailable`.
- Timeline cohort charts honor the display cohort limit while retaining every
  selected cohort's complete point series and every cohort in JSON and CSV.

- Structural segment and block sizes (`total_size`, `allocated_size`,
  `active_size`, block `size`) are required: absent fields raise instead of
  defaulting to invariant-breaking zeros. Absent `requested_size` falls back
  to the active/block size with an aggregated warning, so fragmentation is
  no longer fabricated as 100%.
- `devices="all"` with a snapshot provider no longer binds an empty device
  set when the first snapshot precedes any allocation; device resolution is
  retried like the default selection.
- Lifecycle deltas compare segment and active-block identities as
  multisets, so address-less entries count with multiplicity instead of
  collapsing onto one key and undercounting.
- Allocator events at addresses whose segment changed pools between the
  compared snapshots are attributed with confidence `ambiguous` (event
  tables) or an `unknown` cohort pool (lifetime transients) instead of
  being silently binned into the newer pool as `matched`. Surviving
  instances still take their pool from the matching snapshot block.
- Cohort transition rows are re-stamped with the allocator-rounded block
  size at reconciliation, so `births`/`free_requests` rows agree with
  `born_*`/`free_*` byte totals; transients keep the requested basis
  throughout.
- Timeline HTML charts always plot the full per-point series;
  `include_unchanged=False` no longer bends polylines across filtered
  points or hides single-change series.
- Allocation lifetime reconciliation now matches event-born instances to
  their allocator-rounded snapshot blocks by address with request/rounded
  size compatibility. Previously any allocation whose request size was not
  already allocator-rounded was double counted (a fabricated free plus a
  duplicate birth) whenever it survived a point boundary.
- A missing start marker no longer silently replays the entire history ring
  buffer as event evidence; the affected interval is excluded and reported.
- The advanced `extract_event_window` helpers no longer classify a window as
  complete when an explicitly requested end marker is absent from the trace;
  the window is reported truncated. An explicitly open window
  (`end_marker=None`) still extends through the end of the trace, and
  recorder point evidence keeps treating the ending snapshot's trace end as
  the interval boundary (the end marker enters history as the snapshot is
  taken and is never visible in its own trace). A missing start-boundary
  message lists ring-buffer overwrite and history enabled late as possible
  causes; it cannot distinguish them. Truncated errors recommend enabling
  `_record_memory_history()` before the first analyzed point.
- Memory device selections are normalized to ascending order at binding.
  Previously an unsorted explicit selection such as `devices=(1, 0)` was
  honored verbatim, and the recorder wrote bundles whose device order
  `MemoryRun.load()` rejects.
- A selected device that stayed completely idle across an interval (no
  segments at either endpoint and no trace entries) no longer fails event
  and lifetime analysis with `MemoryHistoryDisabledError`; evidence is
  demanded only from devices with observed activity.
- Snapshot-provider collection with the default or `"all"` device selection
  adopts devices that first appear after binding and warns that earlier
  captures do not cover them, instead of silently dropping every later
  allocation on those devices.
- Segments without a `device` field that are excluded because device 0 is
  not selected are now disclosed with a warning instead of being dropped
  silently before normalization could report the missing field.
- `TensorProbe.close()` and `TensorRecorder.close()` remove every gradient
  hook registered through `watch_grad`; previously a surviving hook fired
  against the closed object inside a later backward pass.
- `event_owner_peak_bytes` no longer subtracts the free request of a block
  that was already awaiting free at the range start. Such blocks never enter
  the owner-active running sum, so the unbalanced subtraction underreported
  cohort owner peaks (down to zero) whenever a range began with pending
  cross-stream frees.
- Snapshot normalization reports absent defaultable fields (`device`,
  `segment_pool_id`, segment/block `requested_size`, and block `state`) through
  the point warnings channel, aggregated per field, instead of substituting
  defaults silently. Structural sizes remain required, and present-but-invalid
  fields still raise.
- `became_inactive_bytes` now mirrors the newly-active rule: a
  reference-active block whose `(address, size)` key is gone in the
  candidate (freed and coalesced, re-split, or in a freed segment) counts as
  became-inactive. Previously only in-place `inactive` transitions counted,
  so steady-state churn read like a leak.
- Timeline HTML charts group polylines by the pool key actually present in
  timeline rows; previously every pool collapsed into one unlabeled line.
- A truncated or corrupted gzip allocator-state or event payload raises
  `MemoryBundleError` instead of leaking a raw `EOFError`, `zlib.error`, or
  `UnicodeDecodeError` through the CLI.
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
- The `tcgd-memory` CLI can be invoked with `python -m
  torch_cudagraph_debug.memory_debug.cli`; previously module invocation was a
  silent no-op.
- The test suite runs on the declared Python 3.10 floor (only the single
  agent-asset test that parses TOML skips where `tomllib` is unavailable;
  the remaining asset gates keep running instead of aborting collection).
- Persisted tensor summaries compute `std` with a chunked Welford merge. The
  one-pass sum-of-squares formula catastrophically cancelled when the mean
  dominated the spread (mean 1e12 with unit spread reported std ~11585).
- TensorBoard export reads scalars from the persisted float64 observation
  summary instead of recomputing them from a float32 cast (which overflowed
  large values to inf and lost int64 precision), hands writers a detached
  histogram copy so writer-side mutation cannot corrupt cached observations,
  and exports scalars for summary-only observations instead of raising
  mid-export.
- Equal-digest comparisons no longer fast-path bit-identical NaNs to a match
  under allclose with `equal_nan=False`; same-dtype exact comparison with
  `equal_nan=True` exempts both-NaN positions from the bitwise check while
  preserving signed-zero and NaN-payload distinctions everywhere else.
- `compare_runs` merges an explicit `point_mapping` with identical-label
  auto-alignment: explicit entries take precedence, unmapped labels align by
  identical label, and a candidate label claimed twice across the merged
  mapping raises `ValueError`. Previously any mapping disabled auto-alignment
  and forced unmapped identical labels into a mismatch.
- `MemoryProbeSnapshot` preserves whether each endpoint metadata boundary was
  actually recorded. Event and lifetime requests now raise
  `MemoryHistoryBoundaryError` for a failed Probe boundary instead of
  misclassifying it as truncated history.
- Combined event and lifetime analysis reuses each loaded interval event
  payload within compare, phase, and timeline workflows. With snapshot caching
  disabled, the same gzip payload is no longer decompressed twice in one
  analysis.
- Serialized tensor comparison options include `layout_policy`; reports no
  longer omit the policy that determined whether source-stride differences
  were mismatches.
- Tensor run groups accept an incomplete rank whose point labels are a strict
  prefix of the longest rank sequence (a crashed rank) with the existing
  incomplete-bundle warning instead of raising; a complete rank with fewer
  points or any non-prefix sequence still raises.
- A tensor probe enqueue that fails validation or host-side slot preparation no longer
  consumes its invocation and slot order: keyed checks stay satisfiable on
  retry, positional checks bind the intended expected entry, and a failed first
  captured call no longer loses the replay-counter capture. Host-side slot
  preparation is transactional: eager insert and replacement failures restore
  the prior name layout and values, while capture failures restore the prior
  layout and order. A
  failure after CUDA command submission begins now makes the probe explicitly
  unusable instead of exposing partially published state.
- Eager (`when="always"`) probes keep one slot per observation name:
  repeated names sample in place and new names append slots, so `snapshot()`
  returns the latest value of every observed name. Previously every eager
  call shared one slot and a second distinct name silently overwrote the
  first. A later capture replaces the eager slot layout with its own and
  retires the eager staging.
- In-place eager re-samples are counted per name and disclosed as
  `TensorProbeSnapshot.eager_overwrites`; snapshot comparisons warn when an
  eager sample with a nonzero count is aligned against multi-invocation
  observations, identify which side needs complete collection, and reject
  malformed overwrite metadata instead of silently pairing or collapsing
  non-corresponding occurrences.
- Record-only `snapshot(synchronize=False)` during CUDA graph capture is
  rejected instead of issuing a blocking counter read that invalidates the
  capture; retired-staging reclaim is likewise deferred during capture and
  while eager work is pending.
- Unsynchronized `close()` verifies pending eager device work through a CUDA
  event and is rejected while copies or callbacks are provably in flight
  (record-only probes previously freed staging a running D2H copy still
  targeted). Replays cannot be verified from inside the probe;
  closing after capture asserts both that no replay is in flight and that no
  graph containing the probe can replay again. `synchronize=False` additionally
  asserts that required synchronization already occurred.

### Changed

- Allocation lifetime analysis requires complete allocator event history.
  Snapshot-only lifetime analysis and inferred in-range transitions were
  removed. Transitions inside the analyzed range are event-backed; a free
  request already in effect at the start is represented once with a
  `range_boundary` origin.
- Missing or incomplete event and lifetime evidence now raises typed errors:
  `MemoryHistoryDisabledError` (history never recorded),
  `MemoryHistoryBoundaryError` (point boundary unavailable), and
  `MemoryHistoryTruncatedError` (a required marker is missing from the trace).
  Ring-buffer overwrite and history enabled late are not distinguishable.
  Stack attribution retains exact partial results with explicit coverage and
  raises only when nonempty active state has zero frame coverage. The
  `on_missing` policy, `MissingPolicy`, and the CLI
  `--on-missing`/`--no-events` flags were removed; lifetime evidence and
  event-table output are independent requests.
- Complete event history that cannot be reconciled with the snapshots raises
  `MemoryReconciliationError` unconditionally, aggregating every
  contradiction with per-device counts and example addresses.
- `AllocationCohort` replaces the paired `event_exact_*` /
  `snapshot_inferred_*` counters with merged `born_*`, `free_requested_*`,
  and `free_completed_*` totals; transition `confidence` was replaced by
  `origin` (`event` or `range_boundary`).
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
  point-owned `states/NNNN.json.gz` allocator state, adjacent
  `events/NNNN-NNNN.json.gz` evidence, independent lazy caches, strict JSON
  validation, and path traversal protection.
- Manifest-only default timelines and unattributed cross-run comparisons;
  attributed analyses load only required allocator states and point-owned event
  chunks while retaining bounded compact indexes.
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

## v0.1.0 - 2026-05-12

Initial tensor-debug release for Linux CUDA source builds.
