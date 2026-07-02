# GitHub Release Checklist

## Metadata And Documentation

- Confirm `pyproject.toml` and `torch_cudagraph_debug.__version__` match the
  intended tag.
- Confirm package URLs, Apache-2.0 license metadata, README install command, and
  changelog are current.
- Confirm the API reference describes the exact supported public facades and
  serialized formats shipped in this repository.
- Confirm the root README stays limited to concise quick starts and links to
  the dedicated Tensor Debug and Memory Debug guides.
- Confirm tensor lifecycle, single-capture ownership, unified replay and
  invocation indexing, device matching, callback overhead, and non-contiguous
  copy cost are documented.
- Confirm tensor run point boundaries, eager/CG observation alignment,
  full/summary payload semantics, three-state comparison, raw blob format, and
  CPU-only offline loading are documented.
- Confirm memory ownership, history policy, cross-run matching, JSON bundle
  format, and one-bundle-per-rank rule are documented.

## Local Gate

```bash
python -m py_compile $(find src tests examples -name '*.py')
bash -n examples/memory_debug/cli_workflows.sh
python -m ruff check src tests examples
python -m pytest -q tests/test_terminology.py
python -m pytest -q
python -m build --sdist --no-isolation
python -m twine check dist/*
git diff --check
```

Inspect the sdist and confirm it contains package sources, C++ and CUDA
sources (including `replay_counter.cu`), tests, examples, and public docs, with
no runtime output or cache directories.

## GPU Gate

Source builds require CUDA-enabled PyTorch, a compatible CUDA development
toolkit, and a C++17 compiler. Start in the source checkout and keep the same
shell for the complete GPU gate:

```bash
TCGD_REPO_ROOT="$(pwd)"
TCGD_RUN_ROOT="$(mktemp -d /tmp/tcgd-gpu-gate.XXXXXX)"

python -m pip install --upgrade "setuptools>=77.0.3" wheel
python -m build --sdist --no-isolation
TCGD_SDIST="$(find dist -maxdepth 1 -name 'torch_cudagraph_debug-*.tar.gz' -print -quit)"
python -m pip install --no-build-isolation --no-deps "${TCGD_SDIST}"
cd "${TCGD_RUN_ROOT}"
TCGD_TEST_INSTALLED=1 python -m pytest -q "${TCGD_REPO_ROOT}/tests"
```

The `cd` ensures tests and examples import the installed package and compiled
native extension rather than source-tree artifacts.

Tensor coverage must include:

- capture-only and always-active probes;
- print, record, check, and sticky typed check status;
- single and repeated invocation expected values;
- gradient hook handles;
- supported dense dtypes and zero-element tensors;
- default non-contiguous rejection and explicit copy mode;
- single-capture ownership rejection;
- callback and side-stream staging behavior;
- 1-based replay advancement, one shared index across repeated invocations,
  queued replay visibility, and retained snapshot indices;
- callback-free query-time counter transfer, callback-counter staging reuse,
  bool/stream/device query synchronization, print cadence, exact check
  failure indices, explicit device selection, and device mismatch errors.
- same-Probe and cross-Probe aggregate snapshot comparison;

- eager repeated named observations and CUDA Graph logical-name slot mapping;
- one shared recorder session and replay counter across many named observations;
- full and summary bundles, content-addressed deduplication, every supported
  dtype, scalars, empty tensors, lazy loading, corruption rejection, and
  payload digest verification;
- point, run, and point-series comparison, first divergence, worst errors,
  strict and promoted dtypes, three-state summary results, reports, and the
  `tcgd-tensor` CLI.

Memory coverage must include:

- state snapshots with history disabled;
- allocation frames with state history enabled;
- marker-delimited events with full history enabled;
- capture-time `record_point()` without synchronization;
- default and graph-private pool discovery;
- start/during/end capture points;
- replay-stable state;
- same-Probe and cross-Probe standalone snapshot comparison;
- gzip JSON persistence and `MemoryRun.load()` round trip.

Run every supported single-GPU example from the installed package:

```bash
EXAMPLE_ROOT="${TCGD_RUN_ROOT}/examples"
mkdir -p "${EXAMPLE_ROOT}"

python "${TCGD_REPO_ROOT}/examples/tensor_debug/quickstart.py"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/snapshot_comparison.py"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/record_and_check.py"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/multiple_invocations.py"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/gradient_probes.py"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/probe_modes.py"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/module_integration.py"

python "${TCGD_REPO_ROOT}/examples/tensor_debug/eager_vs_cuda_graph.py" \
  --output-dir "${EXAMPLE_ROOT}/tensor-eager-vs-cg"
python "${TCGD_REPO_ROOT}/examples/tensor_debug/replay_series.py" \
  --output-dir "${EXAMPLE_ROOT}/tensor-series"
tcgd-tensor summary "${EXAMPLE_ROOT}/tensor-eager-vs-cg/eager.tcgd-tensor"
tcgd-tensor compare-points \
  "${EXAMPLE_ROOT}/tensor-eager-vs-cg/eager.tcgd-tensor" \
  "${EXAMPLE_ROOT}/tensor-eager-vs-cg/cuda-graph.tcgd-tensor" \
  --reference-point forward --candidate-point replay-1 \
  --output "${EXAMPLE_ROOT}/tensor-cli-report"

python "${TCGD_REPO_ROOT}/examples/memory_debug/quickstart.py"
python "${TCGD_REPO_ROOT}/examples/memory_debug/snapshot_comparison.py"
python "${TCGD_REPO_ROOT}/examples/memory_debug/timeline_and_reports.py" \
  --output-dir "${EXAMPLE_ROOT}/timeline"
python "${TCGD_REPO_ROOT}/examples/memory_debug/attribution_modes.py" \
  --output-dir "${EXAMPLE_ROOT}/attribution"
python "${TCGD_REPO_ROOT}/examples/memory_debug/allocation_lifetimes.py" \
  --output-dir "${EXAMPLE_ROOT}/lifetimes"
python "${TCGD_REPO_ROOT}/examples/memory_debug/compare_runs_and_phases.py" \
  --output-dir "${EXAMPLE_ROOT}/runs"
bash "${TCGD_REPO_ROOT}/examples/memory_debug/cli_workflows.sh" \
  single "${EXAMPLE_ROOT}/cli-single"
```

With the optional TensorBoard dependency installed, also run:

```bash
python "${TCGD_REPO_ROOT}/examples/integrations/tensorboard_export.py" \
  --logdir "${EXAMPLE_ROOT}/tensorboard"
```

On a node with at least two GPUs, rerun the complete installed-package suite with
zero skips, then cover rank-local run groups and the remaining CLI commands:

```bash
TCGD_TEST_INSTALLED=1 TCGD_FAIL_ON_SKIP=1 python -m pytest -q "${TCGD_REPO_ROOT}/tests"

python -m torch.distributed.run --standalone --nproc-per-node=2 \
  "${TCGD_REPO_ROOT}/examples/memory_debug/distributed_run_groups.py" \
  --output-dir "${EXAMPLE_ROOT}/groups"
NPROC_PER_NODE=2 bash \
  "${TCGD_REPO_ROOT}/examples/memory_debug/cli_workflows.sh" \
  distributed "${EXAMPLE_ROOT}/cli-distributed"
```

## Public Ref Gate

Before tagging, install the exact public GitHub commit or branch in a clean CUDA
environment and repeat the GPU gate:

```bash
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
