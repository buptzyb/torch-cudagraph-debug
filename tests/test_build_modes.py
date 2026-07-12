"""Build behavior of the TCGD_TENSOR_DEBUG_MODE install setting."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODE_ENV = "TCGD_TENSOR_DEBUG_MODE"
OLD_FLAG = "TCGD_NO_TENSOR_COLLECTION"
MODE_FILE = "tcgd_tensor_debug_mode"
SOURCE_MODE_MARKER = ROOT / "src" / "torch_cudagraph_debug.egg-info" / MODE_FILE


@pytest.fixture(autouse=True)
def restore_source_mode_marker() -> Iterator[None]:
    """Prevent setup.py subprocesses from changing later test processes."""

    original = SOURCE_MODE_MARKER.read_bytes() if SOURCE_MODE_MARKER.exists() else None
    yield
    if original is None:
        SOURCE_MODE_MARKER.unlink(missing_ok=True)
    else:
        SOURCE_MODE_MARKER.parent.mkdir(parents=True, exist_ok=True)
        SOURCE_MODE_MARKER.write_bytes(original)


@pytest.fixture()
def poison_torch(tmp_path: Path) -> Path:
    """A torch package that fails hard if setup.py ever imports it."""

    package = tmp_path / "poison" / "torch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'raise ImportError("torch must not be imported for this build mode")\n',
        encoding="utf-8",
    )
    return package.parent


@pytest.fixture()
def cpu_only_torch(tmp_path: Path) -> Path:
    """A torch stand-in that reports a CPU-only (non-CUDA) build."""

    package = tmp_path / "cpu_only" / "torch"
    (package / "utils").mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from . import cuda, utils\n", encoding="utf-8"
    )
    (package / "cuda.py").write_text(
        textwrap.dedent(
            """
            def _is_compiled() -> bool:
                return False
            """
        ),
        encoding="utf-8",
    )
    (package / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (package / "utils" / "cpp_extension.py").write_text(
        textwrap.dedent(
            """
            class BuildExtension:
                pass


            class CUDAExtension:
                pass
            """
        ),
        encoding="utf-8",
    )
    return package.parent


def _run_setup(
    *argv: str,
    pythonpath: Path,
    mode: str | None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop(MODE_ENV, None)
    env.pop(OLD_FLAG, None)
    if mode is not None:
        env[MODE_ENV] = mode
    env["PYTHONPATH"] = str(pythonpath)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(ROOT / "setup.py"), *argv],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_offline_skips_extension_without_importing_torch(
    poison_torch: Path, tmp_path: Path
) -> None:
    build_dir = tmp_path / "build"
    result = _run_setup(
        "build_ext",
        f"--build-lib={build_dir / 'lib'}",
        f"--build-temp={build_dir / 'temp'}",
        pythonpath=poison_torch,
        mode="offline",
    )
    assert result.returncode == 0, result.stderr
    assert "torch must not be imported" not in result.stderr


@pytest.mark.parametrize("mode", [None, "full"])
def test_full_mode_cpu_only_torch_error_names_offline_mode(
    cpu_only_torch: Path, tmp_path: Path, mode: str | None
) -> None:
    build_dir = tmp_path / "build"
    result = _run_setup(
        "build_ext",
        f"--build-lib={build_dir / 'lib'}",
        f"--build-temp={build_dir / 'temp'}",
        pythonpath=cpu_only_torch,
        mode=mode,
    )
    assert result.returncode != 0
    assert "CUDA-enabled PyTorch" in result.stderr
    assert f"{MODE_ENV}=offline" in result.stderr
    assert "--no-cache-dir" in result.stderr


@pytest.mark.parametrize("mode", [None, "full", "offline"])
def test_metadata_commands_need_no_torch(poison_torch: Path, mode: str | None) -> None:
    result = _run_setup(
        "--version",
        pythonpath=poison_torch,
        mode=mode,
    )
    assert result.returncode == 0, result.stderr


def test_invalid_mode_fails_before_build(poison_torch: Path) -> None:
    result = _run_setup(
        "--version",
        pythonpath=poison_torch,
        mode="offlien",
    )
    assert result.returncode != 0
    assert MODE_ENV in result.stderr
    assert "full, offline" in result.stderr
    assert "'offlien'" in result.stderr


@pytest.mark.parametrize("mode", ["full", "offline"])
def test_egg_info_persists_mode(poison_torch: Path, tmp_path: Path, mode: str) -> None:
    egg_base = tmp_path / mode
    egg_base.mkdir()
    result = _run_setup(
        "egg_info",
        f"--egg-base={egg_base}",
        pythonpath=poison_torch,
        mode=mode,
    )
    assert result.returncode == 0, result.stderr
    egg_info = next(egg_base.glob("*.egg-info"))
    assert (egg_info / MODE_FILE).read_text(encoding="utf-8") == f"{mode}\n"


def test_offline_wheel_removes_stale_native_extension(
    poison_torch: Path, tmp_path: Path
) -> None:
    build_base = tmp_path / "build"
    stale = build_base / "lib" / "torch_cudagraph_debug" / "_C.stale.so"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale native extension")
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    egg_base = tmp_path / "egg"
    egg_base.mkdir()

    result = _run_setup(
        "egg_info",
        f"--egg-base={egg_base}",
        "build",
        f"--build-base={build_base}",
        "bdist_wheel",
        f"--dist-dir={wheel_dir}",
        pythonpath=poison_torch,
        mode="offline",
    )
    assert result.returncode == 0, result.stderr

    wheel = next(wheel_dir.glob("*.whl"))
    assert wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any(Path(name).name.startswith("_C.") for name in names)
        marker = next(
            name for name in names if name.endswith(f".dist-info/{MODE_FILE}")
        )
        assert archive.read(marker) == b"offline\n"


def test_mode_is_documented_everywhere() -> None:
    documented = (
        "README.md",
        "CHANGELOG.md",
        "docs/tensor_debug.md",
        "docs/memory_debug.md",
        "docs/api.md",
        "docs/architecture.md",
        "docs/release_checklist.md",
        ".github/workflows/ci.yml",
    )
    for relative in documented:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert MODE_ENV in text, relative
    assert OLD_FLAG not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "setup.py",
            ROOT / "README.md",
            ROOT / "CHANGELOG.md",
            *(ROOT / "docs").glob("*.md"),
            ROOT / ".github/workflows/ci.yml",
        )
    )


def test_build_and_runtime_errors_name_mode_and_cache_requirement() -> None:
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    native_source = (ROOT / "src" / "torch_cudagraph_debug" / "_native.py").read_text(
        encoding="utf-8"
    )
    availability_source = (
        ROOT / "src" / "torch_cudagraph_debug" / "tensor_debug" / "_availability.py"
    ).read_text(encoding="utf-8")
    for source in (setup_source, native_source, availability_source):
        assert MODE_ENV in source
        assert "--no-cache-dir" in source


@pytest.fixture()
def cuda_torch_broken_toolchain(tmp_path: Path) -> Path:
    """A CUDA torch stand-in whose native build fails like a broken toolchain."""

    package = tmp_path / "cuda_broken" / "torch"
    (package / "utils").mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from . import cuda, utils\n", encoding="utf-8"
    )
    (package / "cuda.py").write_text(
        textwrap.dedent(
            """
            def _is_compiled() -> bool:
                return True
            """
        ),
        encoding="utf-8",
    )
    (package / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (package / "utils" / "cpp_extension.py").write_text(
        textwrap.dedent(
            """
            from setuptools import Extension
            from setuptools.command.build_ext import build_ext


            class BuildExtension(build_ext):
                def run(self):
                    raise RuntimeError("fake toolchain failure: nvcc not found")


            def CUDAExtension(name, sources, include_dirs=None, extra_compile_args=None):
                return Extension(name, sources=[])
            """
        ),
        encoding="utf-8",
    )
    return package.parent


def test_full_mode_toolchain_failure_chains_offline_hint(
    cuda_torch_broken_toolchain: Path, tmp_path: Path
) -> None:
    build_dir = tmp_path / "build"
    result = _run_setup(
        "build_ext",
        f"--build-lib={build_dir / 'lib'}",
        f"--build-temp={build_dir / 'temp'}",
        pythonpath=cuda_torch_broken_toolchain,
        mode=None,
    )
    assert result.returncode != 0
    assert "fake toolchain failure: nvcc not found" in result.stderr
    assert "direct cause" in result.stderr
    assert f"{MODE_ENV}=offline" in result.stderr
    assert "--no-cache-dir" in result.stderr
