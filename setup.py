import os
import shutil
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info

TENSOR_DEBUG_MODE_ENV = "TCGD_TENSOR_DEBUG_MODE"
TENSOR_DEBUG_MODE_FILE = "tcgd_tensor_debug_mode"
TENSOR_DEBUG_MODES = frozenset({"full", "offline"})


def configured_tensor_debug_mode() -> str:
    """Return the validated tensor-debug mode requested for this build."""

    mode = os.environ.get(TENSOR_DEBUG_MODE_ENV, "full")
    if mode not in TENSOR_DEBUG_MODES:
        choices = ", ".join(sorted(TENSOR_DEBUG_MODES))
        raise RuntimeError(
            f"{TENSOR_DEBUG_MODE_ENV} must be one of: {choices}; got {mode!r}"
        )
    return mode


TENSOR_DEBUG_MODE = configured_tensor_debug_mode()


class TCGDEggInfo(egg_info):
    """Persist the selected tensor-debug mode in distribution metadata."""

    def run(self) -> None:
        super().run()
        Path(self.egg_info, TENSOR_DEBUG_MODE_FILE).write_text(
            f"{TENSOR_DEBUG_MODE}\n",
            encoding="utf-8",
        )


class OfflineBuildPy(build_py):
    """Build Python-only artifacts without reusing stale native outputs."""

    def run(self) -> None:
        package_dir = Path(self.build_lib, "torch_cudagraph_debug")
        shutil.rmtree(package_dir, ignore_errors=True)
        super().run()
        stale_extensions = sorted(package_dir.glob("_C*.so"))
        stale_extensions.extend(sorted(package_dir.glob("_C*.pyd")))
        if stale_extensions:
            paths = ", ".join(str(path) for path in stale_extensions)
            raise RuntimeError(
                "offline tensor-debug builds must not contain native extensions; "
                f"found: {paths}"
            )


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
    commands = {"egg_info": TCGDEggInfo}
    if is_metadata_command():
        return [], commands

    if TENSOR_DEBUG_MODE == "offline":
        commands["build_py"] = OfflineBuildPy
        return [], commands

    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception as exc:  # pragma: no cover - exercised during package build only.
        raise RuntimeError(
            "torch-cudagraph-debug requires PyTorch at build time. Install a CUDA-enabled "
            "PyTorch first, then install this package with build isolation disabled "
            "(`pip install --no-cache-dir --no-build-isolation "
            "torch-cudagraph-debug`). To install with offline tensor analysis instead, "
            "set TCGD_TENSOR_DEBUG_MODE=offline during installation and pass "
            "`--no-cache-dir`: memory collection and analysis stay fully available, "
            "and tensor bundles remain loadable and comparable offline."
        ) from exc

    if not torch.cuda._is_compiled():
        raise RuntimeError(
            "torch-cudagraph-debug must be built against a CUDA-enabled PyTorch "
            "installation for live tensor debugging. Set "
            "TCGD_TENSOR_DEBUG_MODE=offline during installation and pass "
            "`--no-cache-dir` to skip the compiled extension: memory collection and "
            "analysis stay fully available, and tensor bundles remain loadable and "
            "comparable offline."
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

    class FullModeBuildExt(BuildExtension):
        """Chain an install-mode hint onto native toolchain build failures."""

        def run(self) -> None:
            try:
                super().run()
            except Exception as exc:
                raise RuntimeError(
                    "building the torch-cudagraph-debug native extension failed; "
                    "the chained toolchain error is the root cause. To keep full "
                    "Tensor Debug mode, fix the CUDA toolchain and reinstall. If "
                    "you only need Memory Debug and offline tensor analysis, "
                    "reinstall with TCGD_TENSOR_DEBUG_MODE=offline and "
                    "`--no-cache-dir` instead."
                ) from exc

    commands["build_ext"] = FullModeBuildExt
    return [extension], commands


ext_modules, cmdclass = get_extensions()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
