# GitHub Release Checklist

## Metadata And Documentation

- Confirm `pyproject.toml` and `torch_cudagraph_debug.__version__` match the
  intended tag.
- Confirm package URLs, Apache-2.0 license metadata, README install command, and
  changelog are current.
- Confirm the API reference describes the exact supported public facades and
  serialized formats shipped in this repository.
- Compare every user-visible text/HTML label with its renderer. Define its
  source, formula, sampling scope, and relationship to structured fields; mark
  intentionally sparse text sections and dynamic example values explicitly.
- Confirm the root README stays limited to concise quick starts and links to
  the dedicated Tensor Debug and Memory Debug guides.
- Confirm tensor lifecycle, single-capture ownership, eager single-stream and
  post-capture rules, unified replay and invocation indexing, device matching,
  callback overhead, non-contiguous copy cost, and synchronization-aware close
  are documented.
- Confirm tensor run point boundaries, eager/CUDA Graph observation alignment,
  full/summary payload semantics, clean versus incomplete terminal runs,
  three-state comparison, raw blob format, and CPU-only offline loading are
  documented.
- Confirm memory ownership, history policy, cross-run matching, JSON bundle
  authentication and validation, lifecycle confidence, display-limit semantics,
  and one-bundle-per-rank rule are documented.
- Confirm stack/event JSON contains structured `stack_frames`, CSV contains
  parseable `stack_frames_json`, FX metadata is readable in text/HTML, and
  limited HTML tables state shown and total row counts.

## Local Gate

Install the development tools from `requirements-dev.txt` before running
the gate (Ruff is pinned — use the repository's version; the others are
floors). Run every gate on Python 3.11 or newer: the suite's one
`tomllib`-dependent test skips on 3.10 and `TCGD_FAIL_ON_SKIP=1` turns
that skip into a failure.

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --upgrade "setuptools>=77.0.3" wheel
python -m compileall -q src tests examples
bash -n examples/tensor_debug/cli/workflows.sh
bash -n examples/memory_debug/cli/workflows.sh
python -m ruff check src tests examples
python -m ruff check --select I src tests examples
python -m ruff format --check src tests examples
python -m pytest -q
rm -rf dist
python -m build --sdist --no-isolation
python -m twine check dist/*
git diff --check   # uncommitted whitespace/conflict markers only
```

Inspect the sdist and confirm it contains package sources, C++ and CUDA
sources (including `replay_counter.cu`), tests, examples, and public docs, with
no runtime output or cache directories.

## GPU Gate

Source builds require CUDA-enabled PyTorch, a compatible CUDA development
toolkit, and a C++17 compiler. The complete GPU gate — including this first
suite run — requires a node with at least two GPUs: `TCGD_FAIL_ON_SKIP=1`
turns every skip into a failure, and parts of the suite skip below two
devices. Start in the source checkout and keep the same shell:

```bash
TCGD_REPO_ROOT="$(pwd)"
TCGD_RUN_ROOT="$(mktemp -d /tmp/tcgd-gpu-gate.XXXXXX)"

python -m pip install -r requirements-dev.txt
python -m pip install --upgrade "setuptools>=77.0.3" wheel
rm -rf dist
python -m build --sdist --wheel --no-isolation
TCGD_SDIST="$(find dist -maxdepth 1 -name 'torch_cudagraph_debug-*.tar.gz' -print -quit)"
python -m pip install --no-build-isolation --no-deps "${TCGD_SDIST}"
cd "${TCGD_RUN_ROOT}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TCGD_TEST_INSTALLED=1 \
  TCGD_FAIL_ON_SKIP=1 \
  python -m pytest -s -q "${TCGD_REPO_ROOT}/tests"
```

The `rm -rf dist` guarantees the freshly built sdist is the one installed
and gated — a stale artifact in `dist/` would otherwise be picked first.
The `cd` plus `TCGD_TEST_INSTALLED=1` ensure tests and examples import the
installed package and compiled native extension rather than source-tree
artifacts (`TCGD_TEST_INSTALLED=1` keeps `src/` off `sys.path`, so a
silently failed install fails the suite instead of falling back).
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` keeps globally installed plugins from changing
collection or execution. `-s` avoids environment-specific file-descriptor capture;
tests that need Python output capture use explicit in-memory fixtures.

Tensor coverage must include:

- capture-only and always-active probes;
- print, record, check, and sticky typed check status;
- single and repeated invocation expected values;
- gradient hook handles;
- supported dense dtypes and zero-element tensors;
- default non-contiguous rejection and explicit copy mode;
- single-capture ownership rejection;
- callback and side-stream staging behavior;
- bounded eager callback payload ownership, eager non-contiguous source
  release, retired pinned-staging reclamation, and pending-callback close
  protection;
- eager single-stream ownership, one latest-value slot per observation name,
  first-use positional order, eager-before-capture support, post-capture eager
  rejection, and close rejection during capture;
- host-side enqueue rollback for validation, slot reservation, staging, and
  payload failures; explicit unusable state after partial CUDA submission;
- 1-based replay advancement, one shared index across repeated invocations,
  queued replay visibility, and retained snapshot indices;
- callback-free query-time counter transfer, callback-counter staging reuse,
  bool/stream/device query synchronization, print cadence, exact check
  failure indices, explicit device selection, and device mismatch errors;
- same-probe and cross-probe aggregate snapshot comparison;
- eager repeated named observations and CUDA Graph logical-name slot mapping;
- one shared recorder session and replay counter across many named observations;
- full and summary bundles, content-addressed deduplication, every supported
  dtype, scalars, empty tensors, lazy loading, corruption rejection, and
  payload digest verification;
- exact full-range integer diagnostics in Python comparison and native checks;
- point, run, and point-series comparison, first divergence, worst errors,
  strict and promoted dtypes, three-state summary results, reports, and the
  `tcgd-tensor` CLI;
- multi-rank `TensorRunGroup` summaries and comparisons, point-label mappings,
  completeness handling, and direct text/JSON/HTML/report output.

Memory coverage must include:

- state snapshots with history disabled;
- allocation frames with state history enabled;
- marker-delimited events with full history enabled;
- capture-time `record_point()` without synchronization;
- default and graph-private pool discovery;
- start/during/end capture points;
- replay-stable state;
- same-probe and cross-probe standalone snapshot comparison;
- device-aware identities for one, multiple, and all visible devices;
- allocated, reserved, active, requested, awaiting-free, inactive,
  fragmentation, segment, block, expandable reserved, and expandable inactive
  metrics;
- gzip JSON persistence and `MemoryRun.load()` round trip;
- payload digest verification, state/manifest cross-validation, and explicit
  cache-independent full-bundle validation;
- exact, approximate, and unavailable address-lifecycle confidence;
- bounded cohort charts that retain complete structured timeline data.

Run every supported single-GPU example from the installed package:

```bash
EXAMPLE_ROOT="${TCGD_RUN_ROOT}/examples"
mkdir -p "${EXAMPLE_ROOT}"

python "${TCGD_REPO_ROOT}/examples/tensor_debug/probe/quickstart.py"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/probe/snapshot_comparison.py"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/probe/replay_comparison.py"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/probe/actions.py"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/probe/gradients.py"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/probe/capture_modes.py"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/probe/module_integration.py"

python "${TCGD_REPO_ROOT}/examples/tensor_debug/recorder/eager_vs_cuda_graph.py" \
  --output-dir "${EXAMPLE_ROOT}/tensor-eager-vs-cg"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/recorder/forward_backward.py" \
  --output-dir "${EXAMPLE_ROOT}/tensor-forward-backward"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/recorder/replay_series.py" \
  --output-dir "${EXAMPLE_ROOT}/tensor-series"
bash "${TCGD_REPO_ROOT}/examples/tensor_debug/cli/workflows.sh" \
  "${EXAMPLE_ROOT}/tensor-cli"

python "${TCGD_REPO_ROOT}/examples/memory_debug/probe/quickstart.py"
python "${TCGD_REPO_ROOT}/examples/memory_debug/probe/private_pool_inactive.py"
python "${TCGD_REPO_ROOT}/examples/memory_debug/probe/snapshot_comparison.py"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python "${TCGD_REPO_ROOT}/examples/memory_debug/probe/expandable_segments.py"
python "${TCGD_REPO_ROOT}/examples/memory_debug/recorder/timeline_and_reports.py" \
  --output-dir "${EXAMPLE_ROOT}/timeline"
python "${TCGD_REPO_ROOT}/examples/memory_debug/recorder/history_requirements.py" \
  --output-dir "${EXAMPLE_ROOT}/history"
python "${TCGD_REPO_ROOT}/examples/memory_debug/recorder/stack_and_event_attribution.py" \
  --output-dir "${EXAMPLE_ROOT}/stack-events"
python "${TCGD_REPO_ROOT}/examples/memory_debug/recorder/allocation_lifetimes.py" \
  --output-dir "${EXAMPLE_ROOT}/lifetimes"
python "${TCGD_REPO_ROOT}/examples/memory_debug/recorder/compare_runs_and_phases.py" \
  --output-dir "${EXAMPLE_ROOT}/runs"
bash "${TCGD_REPO_ROOT}/examples/memory_debug/cli/workflows.sh" \
  single "${EXAMPLE_ROOT}/cli-single"
```

With the optional TensorBoard dependency installed, also run:

```bash
python "${TCGD_REPO_ROOT}/examples/integrations/tensorboard_export.py" \
  --logdir "${EXAMPLE_ROOT}/tensorboard"
```

Still on the two-GPU node, rerun the complete installed-package suite with
zero skips — the examples above ran real workloads against the installed
package, and this rerun confirms none of them corrupted the environment —
then cover rank-local run groups and the remaining CLI commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TCGD_TEST_INSTALLED=1 \
  TCGD_FAIL_ON_SKIP=1 python -m pytest -s -q "${TCGD_REPO_ROOT}/tests"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TCGD_TEST_INSTALLED=1 TCGD_FAIL_ON_SKIP=1 \
  python -m pytest -s -q \
  "${TCGD_REPO_ROOT}/tests/memory_debug/test_gpu_smoke.py::test_graph_pool_capture_and_json_bundle_round_trip"

python -m torch.distributed.run --standalone --nproc-per-node=2 \
  "${TCGD_REPO_ROOT}/examples/tensor_debug/recorder/distributed_run_groups.py" \
  --output-dir "${EXAMPLE_ROOT}/tensor-groups"

python -m torch.distributed.run --standalone --nproc-per-node=2 \
  "${TCGD_REPO_ROOT}/examples/memory_debug/recorder/distributed_run_groups.py" \
  --output-dir "${EXAMPLE_ROOT}/groups"
NPROC_PER_NODE=2 bash \
  "${TCGD_REPO_ROOT}/examples/memory_debug/cli/workflows.sh" \
  distributed "${EXAMPLE_ROOT}/cli-distributed"
```

## PyTorch Compatibility Gate

Memory Debug depends on private PyTorch allocator interfaces and schemas:
`torch.cuda.memory._snapshot()`, allocator metadata markers, block `frames`, and
`device_traces`. For every newly supported PyTorch minor or NVIDIA container
release, repeat the exact-sdist installed-package GPU gate with
`TCGD_FAIL_ON_SKIP=1`. Confirm state-only snapshots with history disabled,
allocation frames with state history, marker-delimited events with full
history, graph-private pool IDs, and capture-time snapshot behavior. Treat a
schema or warning-policy change as a compatibility issue to fix or document;
do not accept a skipped test as coverage.

## Public Ref Gate

Before tagging, install the exact public GitHub commit or branch in a clean CUDA
environment and repeat the GPU gate's test and example steps — skip the
gate's own build/install steps, which would overwrite this public-ref
install with a locally built sdist and hide public-ref-only failures
(files missing from the pushed ref):

```bash
python -m pip install --upgrade "setuptools>=77.0.3" wheel
REF=<commit-or-branch>
python -m pip install --no-build-isolation \
  "git+https://github.com/buptzyb/torch-cudagraph-debug.git@${REF}"
```

## Publish

After every gate passes:

```bash
git tag -a v0.2.0 -m "torch-cudagraph-debug v0.2.0"
git push origin v0.2.0
```

Verify installation from the tag in a clean CUDA-enabled environment:

```bash
python -m pip install --no-build-isolation \
  "git+https://github.com/buptzyb/torch-cudagraph-debug.git@v0.2.0"
```

The release remains source-only; users build against their installed
CUDA-enabled PyTorch.
