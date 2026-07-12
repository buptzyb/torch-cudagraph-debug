"""Live Tensor Debug availability in offline installs."""

from __future__ import annotations

import pytest

from torch_cudagraph_debug.tensor_debug import (
    LiveTensorDebugUnavailableError,
    RecordAction,
    TensorProbe,
    TensorRecorder,
    _availability,
)


@pytest.fixture()
def offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_availability, "tensor_debug_mode", lambda: "offline")


def test_offline_rejects_enabled_probe(offline_mode: None) -> None:
    with pytest.raises(
        LiveTensorDebugUnavailableError,
        match="TCGD_TENSOR_DEBUG_MODE=offline",
    ) as exc_info:
        TensorProbe("hidden", [RecordAction()])

    assert "TCGD_TENSOR_DEBUG_MODE=full" in str(exc_info.value)


def test_offline_allows_disabled_noop_probe(offline_mode: None) -> None:
    probe = TensorProbe("hidden", [RecordAction(enabled=False)])
    assert probe.replay_index is None
    probe.close()


@pytest.mark.parametrize("execution", ["eager", "cuda_graph"])
def test_offline_rejects_both_recorder_modes(
    offline_mode: None, execution: str
) -> None:
    with pytest.raises(
        LiveTensorDebugUnavailableError,
        match="Offline tensor bundle analysis and Memory Debug remain available",
    ):
        TensorRecorder(execution=execution)  # type: ignore[arg-type]
