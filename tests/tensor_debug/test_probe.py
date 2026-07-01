from __future__ import annotations

import pytest
import torch

from torch_cudagraph_debug import _native
from torch_cudagraph_debug._errors import NativeExtensionUnavailableError
from torch_cudagraph_debug.tensor_debug import (
    TensorProbe,
    CompareTensor,
    TensorMismatchError,
    TensorProbeStatus,
    PrintTensor,
)
from torch_cudagraph_debug.tensor_debug import probe as probe_module


def test_require_native_reports_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_native, "_EXTENSION", None)
    monkeypatch.setattr(
        _native, "_EXTENSION_ERROR", ImportError("missing test extension")
    )

    with pytest.raises(
        NativeExtensionUnavailableError, match="native extension is unavailable"
    ):
        _native.require_native()


def test_probe_uses_opaque_native_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHandle:
        def __init__(self) -> None:
            self.closed = False
            self.input = None

        def enqueue(self, tensor: torch.Tensor) -> torch.Tensor:
            self.input = tensor
            return tensor

        def records(self) -> list[dict[str, object]]:
            return [
                {
                    "probe_name": "mid",
                    "replay_index": 7,
                    "invocation_index": 2,
                    "shape": (2,),
                    "device": "cuda:0",
                    "tensor": torch.tensor([1.0, 2.0]),
                }
            ]

        def clear_records(self) -> None:
            pass

        def status(self) -> dict[str, object]:
            return {
                "ok": True,
                "message": "",
                "replay_index": 0,
                "invocation_index": -1,
            }

        def close(self) -> None:
            self.closed = True

    class FakeNative:
        @staticmethod
        def create_tensor_debug_probe(
            name: str,
            actions: list[dict[str, object]],
            non_contiguous: str,
            mode: str,
        ) -> FakeHandle:
            assert name == "mid"
            assert len(actions) == 1
            assert actions[0]["kind"] == "print"
            assert non_contiguous == "copy"
            assert mode == "always"
            return FakeHandle()

    monkeypatch.setattr(probe_module._native, "require_native", lambda: FakeNative)

    probe = TensorProbe(
        "mid",
        [PrintTensor(max_items=1)],
        non_contiguous="copy",
        when="always",
    )
    tensor = torch.tensor([3.0])

    assert probe(tensor) is tensor
    snapshots = probe.snapshots()
    assert len(snapshots) == 1
    assert snapshots[0].probe_name == "mid"
    assert snapshots[0].replay_index == 7
    assert snapshots[0].invocation_index == 2
    assert snapshots[0].shape == (2,)
    assert snapshots[0].dtype == torch.float32
    assert snapshots[0].device == "cuda:0"
    assert torch.equal(snapshots[0].tensor, torch.tensor([1.0, 2.0]))
    probe.assert_ok()
    probe.close()

    with pytest.raises(RuntimeError, match="closed"):
        probe.snapshots()


def test_assert_ok_raises_compare_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHandle:
        def status(self) -> dict[str, object]:
            return {
                "ok": False,
                "message": "probe mid mismatch",
                "replay_index": 3,
                "invocation_index": 1,
            }

        def close(self) -> None:
            pass

    class FakeNative:
        @staticmethod
        def create_tensor_debug_probe(
            name: str,
            actions: list[dict[str, object]],
            non_contiguous: str,
            mode: str,
        ) -> FakeHandle:
            assert mode == "capture"
            return FakeHandle()

    monkeypatch.setattr(probe_module._native, "require_native", lambda: FakeNative)

    probe = TensorProbe("mid", [PrintTensor()])
    with pytest.raises(TensorMismatchError, match="mismatch"):
        probe.assert_ok()


def test_probe_validates_non_contiguous_policy() -> None:
    with pytest.raises(ValueError, match="non_contiguous"):
        TensorProbe(  # type: ignore[arg-type]
            "mid",
            [PrintTensor()],
            non_contiguous="bad",
        )


def test_probe_validates_when() -> None:
    with pytest.raises(ValueError, match="when"):
        TensorProbe(  # type: ignore[arg-type]
            "mid",
            [PrintTensor()],
            when="sometimes",
        )


def test_probe_filters_disabled_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHandle:
        def close(self) -> None:
            pass

    class FakeNative:
        @staticmethod
        def create_tensor_debug_probe(
            name: str,
            actions: list[dict[str, object]],
            non_contiguous: str,
            mode: str,
        ) -> FakeHandle:
            assert name == "mid"
            assert [action["max_items"] for action in actions] == [3]
            assert mode == "capture"
            return FakeHandle()

    monkeypatch.setattr(probe_module._native, "require_native", lambda: FakeNative)

    probe = TensorProbe(
        "mid",
        [
            PrintTensor(max_items=1, enabled=False),
            PrintTensor(max_items=3, enabled=True),
        ],
    )
    probe.close()


def test_all_disabled_probe_is_noop_without_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_require_native() -> object:
        raise AssertionError("native should not be loaded for an all-disabled probe")

    monkeypatch.setattr(probe_module._native, "require_native", fail_require_native)

    probe = TensorProbe(
        "disabled",
        [
            PrintTensor(enabled=False),
            CompareTensor(object(), enabled=False),
        ],
    )
    tensor = torch.tensor([1.0])

    assert probe(tensor) is tensor
    assert probe.snapshots() == []
    assert probe.status() == TensorProbeStatus(
        ok=True, message="", replay_index=0, invocation_index=-1
    )
    probe.assert_ok()
    probe.clear_snapshots()
    probe.close()

    with pytest.raises(RuntimeError, match="closed"):
        probe(tensor)


def test_watch_grad_noops_for_tensor_without_grad() -> None:
    probe = TensorProbe("disabled", [PrintTensor(enabled=False)])
    tensor = torch.tensor([1.0])

    assert probe.watch_grad(tensor) is None

    with pytest.raises(RuntimeError, match="does not require grad"):
        probe.watch_grad(tensor, strict=True)


def test_watch_grad_probes_but_returns_original_grad() -> None:
    class ReturningWrongProbe(TensorProbe):
        def __init__(self) -> None:
            self.name = "grad"
            self._closed = False
            self.calls: list[torch.Tensor] = []

        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            self.calls.append(tensor.detach().clone())
            return torch.zeros_like(tensor)

    probe = ReturningWrongProbe()
    x = torch.tensor([2.0, -3.0], requires_grad=True)

    handle = probe.watch_grad(x)
    assert handle is not None
    y = (x * torch.tensor([4.0, 5.0])).sum()
    y.backward()

    assert len(probe.calls) == 1
    assert torch.equal(probe.calls[0], torch.tensor([4.0, 5.0]))
    assert torch.equal(x.grad, torch.tensor([4.0, 5.0]))


def test_watch_grad_returned_handle_can_remove_hook() -> None:
    class CountingProbe(TensorProbe):
        def __init__(self) -> None:
            self.name = "grad"
            self._closed = False
            self.calls = 0

        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return tensor

    probe = CountingProbe()
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    handle = probe.watch_grad(x)

    assert handle is not None
    (x * 2).sum().backward()
    assert probe.calls == 1

    x.grad = None
    handle.remove()
    (x * 3).sum().backward()
    assert probe.calls == 1
    assert torch.equal(x.grad, torch.tensor([3.0, 3.0]))
