from pathlib import Path
import sys

from setuptools import setup


def is_metadata_command() -> bool:
    metadata_commands = {
        "egg_info",
        "dist_info",
        "sdist",
        "clean",
        "--name",
        "--version",
    }
    return any(arg in metadata_commands for arg in sys.argv[1:])


def get_extensions():
    if is_metadata_command():
        return [], {}

    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception as exc:  # pragma: no cover - exercised during package build only.
        raise RuntimeError(
            "torch-cudagraph-debug requires PyTorch at build time. Install a CUDA-enabled "
            "PyTorch first, then install this package with build isolation disabled if needed."
        ) from exc

    if not torch.cuda._is_compiled():
        raise RuntimeError(
            "torch-cudagraph-debug must be built against a CUDA-enabled PyTorch installation."
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
