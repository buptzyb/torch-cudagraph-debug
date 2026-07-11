from pathlib import Path
import os
import sys

from setuptools import setup


def is_metadata_command() -> bool:
    """True only when no requested command needs the native extension.

    distutils runs every command of one invocation with the same
    ``ext_modules``, so a build-style command anywhere in argv must win:
    ``setup.py sdist bdist_wheel`` would otherwise silently produce a pure
    wheel without ``_C``, and ``clean build_ext --inplace`` a no-op rebuild.
    """

    build_commands = {
        "build",
        "build_ext",
        "build_py",
        "bdist",
        "bdist_wheel",
        "bdist_egg",
        "install",
        "develop",
        "editable_wheel",
    }
    metadata_commands = {
        "egg_info",
        "dist_info",
        "sdist",
        "clean",
        "--name",
        "--version",
    }
    args = sys.argv[1:]
    if any(arg in build_commands for arg in args):
        return False
    return any(arg in metadata_commands for arg in args)


def get_extensions():
    if is_metadata_command():
        return [], {}

    if os.environ.get("TCGD_NO_TENSOR_COLLECTION") == "1":
        return [], {}

    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception as exc:  # pragma: no cover - exercised during package build only.
        raise RuntimeError(
            "torch-cudagraph-debug requires PyTorch at build time. Install a CUDA-enabled "
            "PyTorch first, then install this package with build isolation disabled "
            "(`pip install --no-build-isolation torch-cudagraph-debug`). To install "
            "without tensor collection instead, set TCGD_NO_TENSOR_COLLECTION=1 during "
            "installation: memory collection and analysis stay fully available, and "
            "tensor bundles remain loadable and comparable offline."
        ) from exc

    if not torch.cuda._is_compiled():
        raise RuntimeError(
            "torch-cudagraph-debug must be built against a CUDA-enabled PyTorch "
            "installation for tensor collection. Set TCGD_NO_TENSOR_COLLECTION=1 "
            "during installation to skip the compiled extension: memory collection "
            "and analysis stay fully available, and tensor bundles remain loadable "
            "and comparable offline."
        )

    root = Path(__file__).parent
    csrc = root / "src" / "torch_cudagraph_debug" / "csrc"
    sources = [
        csrc / "bindings.cpp",
        csrc / "tensor_debug" / "probe_context.cpp",
        csrc / "tensor_debug" / "replay_counter.cu",
        csrc / "tensor_debug" / "tensor_format.cpp",
        csrc / "tensor_debug" / "check.cpp",
    ]

    extension = CUDAExtension(
        name="torch_cudagraph_debug._C",
        sources=[str(path.relative_to(root)) for path in sources],
        include_dirs=[str(csrc)],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", "-Wall"],
            "nvcc": ["-O3", "-std=c++17"],
        },
    )
    return [extension], {"build_ext": BuildExtension}


ext_modules, cmdclass = get_extensions()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
