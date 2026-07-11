"""Build-mode behavior of the TCGD_NO_TENSOR_COLLECTION install switch."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLAG = "TCGD_NO_TENSOR_COLLECTION"


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
    flag_value: str | None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop(FLAG, None)
    if flag_value is not None:
        env[FLAG] = flag_value
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


def test_flag_skips_extension_without_importing_torch(
    poison_torch: Path, tmp_path: Path
) -> None:
    build_dir = tmp_path / "build"
    result = _run_setup(
        "build_ext",
        f"--build-lib={build_dir / 'lib'}",
        f"--build-temp={build_dir / 'temp'}",
        pythonpath=poison_torch,
        flag_value="1",
    )
    assert result.returncode == 0, result.stderr
    assert "torch must not be imported" not in result.stderr


def test_cpu_only_torch_error_names_the_flag(
    cpu_only_torch: Path, tmp_path: Path
) -> None:
    build_dir = tmp_path / "build"
    result = _run_setup(
        "build_ext",
        f"--build-lib={build_dir / 'lib'}",
        f"--build-temp={build_dir / 'temp'}",
        pythonpath=cpu_only_torch,
        flag_value=None,
    )
    assert result.returncode != 0
    assert "CUDA-enabled PyTorch" in result.stderr
    assert FLAG in result.stderr


def test_metadata_commands_need_neither_torch_nor_the_flag(poison_torch: Path) -> None:
    result = _run_setup(
        "--version",
        pythonpath=poison_torch,
        flag_value=None,
    )
    assert result.returncode == 0, result.stderr


def test_flag_is_documented_everywhere() -> None:
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
        assert FLAG in text, relative


def test_build_and_runtime_errors_name_the_flag() -> None:
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert setup_source.count(FLAG) >= 3
    native_source = (ROOT / "src" / "torch_cudagraph_debug" / "_native.py").read_text(
        encoding="utf-8"
    )
    assert FLAG in native_source
