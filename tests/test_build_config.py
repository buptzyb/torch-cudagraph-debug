"""Installed Tensor Debug mode metadata."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from torch_cudagraph_debug import _build_config


class _Distribution:
    def __init__(self, value: str | None) -> None:
        self.value = value

    def read_text(self, filename: str) -> str | None:
        assert filename == "tcgd_tensor_debug_mode"
        return self.value


def test_missing_legacy_marker_defaults_to_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_build_config, "distribution", lambda _: _Distribution(None))
    assert _build_config._load_tensor_debug_mode() == "full"


def test_empty_installed_marker_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_build_config, "distribution", lambda _: _Distribution(""))
    with pytest.raises(RuntimeError, match="invalid tensor-debug mode"):
        _build_config._load_tensor_debug_mode()


def test_missing_distribution_defaults_to_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> _Distribution:
        raise PackageNotFoundError

    monkeypatch.setattr(_build_config, "distribution", missing)
    assert _build_config._load_tensor_debug_mode() == "full"


@pytest.mark.parametrize("mode", ["full", "offline"])
def test_valid_installed_mode_is_loaded(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(_build_config, "distribution", lambda _: _Distribution(mode))
    assert _build_config._load_tensor_debug_mode() == mode


def test_invalid_installed_mode_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _build_config,
        "distribution",
        lambda _: _Distribution("unexpected"),
    )
    with pytest.raises(RuntimeError, match="invalid tensor-debug mode"):
        _build_config._load_tensor_debug_mode()


def test_offline_mode_notice_is_emitted_once_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(_build_config, "_TENSOR_DEBUG_MODE", "offline")
    monkeypatch.setattr(_build_config, "_OFFLINE_MODE_NOTICE_EMITTED", False)

    _build_config._emit_offline_mode_notice()
    _build_config._emit_offline_mode_notice()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "torch-cudagraph-debug: offline mode (live TensorProbe/TensorRecorder disabled). "
        "To enable full mode, reinstall with TCGD_TENSOR_DEBUG_MODE=full; "
        "see the installation guide.\n"
    )


def test_full_mode_notice_is_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(_build_config, "_TENSOR_DEBUG_MODE", "full")
    monkeypatch.setattr(_build_config, "_OFFLINE_MODE_NOTICE_EMITTED", False)

    _build_config._emit_offline_mode_notice()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_offline_package_import_emits_notice(tmp_path: Path) -> None:
    dist_info = tmp_path / "torch_cudagraph_debug-0.2.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torch-cudagraph-debug\nVersion: 0.2.0\n"
    )
    (dist_info / "tcgd_tensor_debug_mode").write_text("offline\n")
    source_root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(source_root)))

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch_cudagraph_debug; import torch_cudagraph_debug",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""
    assert result.stderr == (
        "torch-cudagraph-debug: offline mode (live TensorProbe/TensorRecorder disabled). "
        "To enable full mode, reinstall with TCGD_TENSOR_DEBUG_MODE=full; "
        "see the installation guide.\n"
    )
