# GitHub Release Checklist

This checklist is for publishing `torch-cudagraph-debug` from GitHub. Users
install from Git tags or commits with `pip install --no-build-isolation`.

## Before Publishing

- Confirm `pyproject.toml` version matches the intended release.
- Confirm `pyproject.toml` `project.urls` points at the public repository,
  issue tracker, and documentation.
- Confirm `LICENSE` is present and `license = "Apache-2.0"` is accepted by
  `setuptools>=77.0.3`.
- Confirm README installation instructions mention `pip install --no-build-isolation`.
- Confirm README install examples reference the intended release tag.
- Confirm `CHANGELOG.md` summarizes the release API surface and compatibility
  notes.
- Confirm CUDA graph lifecycle, single-capture probe topology, multi-invocation
  indexing, host-callback overhead, and non-contiguous memory policy are
  documented.
- Run local checks:

```bash
python -m py_compile $(find src tests examples -name '*.py')
python -m build --sdist --no-isolation
git diff --check
```

## GPU Gate

The GPU gate must install the package from source in a CUDA-enabled PyTorch
environment and pass the full pytest suite, including CUDA graph capture/replay
tests for:

- contiguous print/record/compare;
- non-contiguous default error;
- non-contiguous explicit copy;
- compare mismatch reporting;
- multi-invocation latest `TensorRecord` and Python-side offline compare;
- per-invocation `TensorCompare` expected lists;
- different tensor metadata across invocations in one capture;
- rejection when one probe is reused in another graph capture;
- callback-only per-invocation staging, including side-stream capture;
- zero-element tensors;
- supported dense dtype coverage.

Before publishing, also verify the public GitHub install path from the exact
branch or tag that will be released. Install from GitHub, then check out the
same ref and run tests from outside the source tree so imports resolve to the
installed package, including the compiled native extension:

```bash
REF=v0.1.0
pip install --no-build-isolation \
  "git+https://github.com/buptzyb/torch-cudagraph-debug.git@${REF}"
git clone https://github.com/buptzyb/torch-cudagraph-debug.git
cd torch-cudagraph-debug
git checkout "${REF}"
CHECKOUT=$PWD
rm -rf /tmp/tcgd-tests
cp -r tests /tmp/tcgd-tests
cd /tmp
pytest -q /tmp/tcgd-tests
python "${CHECKOUT}/examples/grad_probe_patterns.py"
python "${CHECKOUT}/examples/multiple_invocations_record_compare.py"
python "${CHECKOUT}/examples/tensorboard_export_records.py"
```

Running `pytest` from the source checkout can import `src/torch_cudagraph_debug`
instead of the installed package and make CUDA-native tests skip, so do not use
that as the release GPU gate.

## Build Artifacts

Build source distribution locally:

```bash
python -m build --sdist --no-isolation
twine check dist/*
```

Do not publish prebuilt wheels for v0.1.0. Users should build against their
local CUDA-enabled PyTorch installation.

## Publish

Create and push a signed or annotated tag after the GitHub install gate passes:

```bash
git tag -a v0.1.0 -m "torch-cudagraph-debug v0.1.0"
git push origin v0.1.0
```

Verify installation from the tag in a CUDA-enabled PyTorch environment:

```bash
pip install --no-build-isolation \
  "git+https://github.com/buptzyb/torch-cudagraph-debug.git@v0.1.0"
```
