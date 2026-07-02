from __future__ import annotations

import pytest
import torch

from torch_cudagraph_debug.memory_debug import MemoryRecorder
from torch_cudagraph_debug.memory_debug import recording

from ._helpers import segment, snapshot


def test_mark_inside_capture_skips_synchronize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_sync() -> None:
        raise AssertionError("mark() must not synchronize during capture")

    def fail_history(*args: object, **kwargs: object) -> None:
        raise AssertionError("recorder must not control memory history")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_sync)
    monkeypatch.setattr(
        torch.cuda.memory,
        "_record_memory_history",
        fail_history,
    )
    monkeypatch.setattr(
        torch.cuda.memory,
        "_snapshot",
        lambda: snapshot(segment(active=10)),
    )
    monkeypatch.setattr(torch.cuda.memory, "_get_memory_metadata", lambda: "")
    monkeypatch.setattr(torch.cuda.memory, "_set_memory_metadata", lambda value: None)

    point = MemoryRecorder().record_point("inside_capture")

    assert point.label == "inside_capture"
    assert point.pool_stats[(0, 0)].active_bytes == 10


def test_mark_outside_capture_synchronizes_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(
        torch.cuda.memory,
        "_snapshot",
        lambda: snapshot(segment(active=10)),
    )
    monkeypatch.setattr(torch.cuda.memory, "_get_memory_metadata", lambda: "")
    monkeypatch.setattr(torch.cuda.memory, "_set_memory_metadata", lambda value: None)

    MemoryRecorder().record_point("outside_capture")

    assert calls == ["sync"]


def test_device_provenance_is_deferred_until_after_real_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda.memory,
        "_snapshot",
        lambda: snapshot(segment(active=10)),
    )
    monkeypatch.setattr(torch.cuda.memory, "_get_memory_metadata", lambda: "")
    monkeypatch.setattr(torch.cuda.memory, "_set_memory_metadata", lambda value: None)
    monkeypatch.setattr(
        recording,
        "initialized_device_provenance",
        lambda: (
            calls.append("device")
            or {
                "index": 0,
                "name": "Test GPU",
                "capability": [10, 0],
                "total_memory_bytes": 1024,
            }
        ),
    )

    recorder = MemoryRecorder()
    assert calls == []

    recorder.record_point("point")
    run = recorder.finish()

    assert calls == ["device"]
    assert run.provenance["device"]["name"] == "Test GPU"
