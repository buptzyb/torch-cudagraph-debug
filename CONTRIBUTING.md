# Contributing

Thanks for improving `torch-cudagraph-debug`.

## Development Setup

Install a CUDA-enabled PyTorch build, a compatible CUDA development toolkit,
and a C++17 compiler first. Source installation builds the native tensor
extension for the whole package. Then install from the source checkout:

```bash
python -m pip install --upgrade pip "setuptools>=77.0.3" wheel
python -m pip install pytest build twine ruff
python -m pip install --no-build-isolation -e .
```

CPU-only environments can run Python-level tests. CUDA graph behavior requires a
CUDA-enabled PyTorch runtime and a GPU.

## Checks

Run local checks before opening a pull request:

```bash
python -m py_compile $(find src tests examples -name '*.py')
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m pytest -q tests
python -m build --sdist --no-isolation
python -m twine check dist/*
```

GPU tests are marked with `pytest.mark.gpu` but are included in the default test
suite; they skip automatically when CUDA or the native extension is unavailable.

## Native Extension Notes

The host callback path must not call Python or CUDA APIs. Keep callback work to
native CPU operations on already-allocated host memory. If a change needs CUDA
work, enqueue it before the host callback so it becomes part of graph capture.

## Release Notes

This project publishes source distributions first. Do not add prebuilt CUDA
wheels unless the release process also covers PyTorch, CUDA, Python, and platform
compatibility for those wheels.

