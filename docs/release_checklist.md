# GitHub Release Checklist

## Metadata And Documentation

- Confirm `pyproject.toml` and `torch_cudagraph_debug.__version__` match the
  intended tag.
- Confirm package URLs, Apache-2.0 license metadata, README install command, and
  changelog are current.
- Confirm the API reference and architecture spec describe the exact stable
  facades and serialized formats.
- Confirm tensor lifecycle, single-capture ownership, invocation indexing,
  callback overhead, and non-contiguous copy cost are documented.
- Confirm memory ownership, history policy, cross-run matching, JSON bundle
  format, and one-bundle-per-rank rule are documented.

## Local Gate

```bash
python -m py_compile $(find src tests examples -name '*.py')
pytest -q
python -m build --sdist --no-isolation
twine check dist/*
git diff --check
```

Inspect the sdist and confirm it contains package sources, C++ sources, tests,
examples, and public docs, with no runtime output or cache directories.

## GPU Gate

Install from source in the target CUDA-enabled PyTorch container:

```bash
pip install --no-build-isolation --no-deps .
```

Run tests outside the source checkout so imports resolve to the installed
package and compiled native extension.

Tensor coverage must include:

- capture-only and always-active probes;
- print, snapshot, compare, and sticky typed status;
- single and repeated invocation expected values;
- gradient hook handles;
- supported dense dtypes and zero-element tensors;
- default non-contiguous rejection and explicit copy mode;
- single-capture ownership rejection;
- callback and side-stream staging behavior.

Memory coverage must include:

- state snapshots with history disabled;
- allocation frames with state history enabled;
- marker-delimited events with full history enabled;
- capture-time `mark()` without synchronization;
- default and graph-private pool discovery;
- before/during/after capture points;
- replay-stable state;
- gzip JSON persistence and `MemoryRun.load()` round trip.

Run the user-facing examples that are relevant to the release:

```bash
python examples/grad_probe_patterns.py
python examples/multiple_invocations_record_compare.py
python examples/tensorboard_export_records.py
python examples/memory_debug_basic.py
python examples/memory_debug_timeline.py
```

## Public Ref Gate

Before tagging, install the exact public GitHub commit or branch in a clean CUDA
environment and repeat the GPU gate:

```bash
REF=<commit-or-branch>
pip install --no-build-isolation \
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
pip install --no-build-isolation \
  "git+https://github.com/buptzyb/torch-cudagraph-debug.git@v0.2.0"
```

The release remains source-only; users build against their installed
CUDA-enabled PyTorch.
