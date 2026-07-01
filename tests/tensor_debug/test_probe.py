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
    RecordTensor,
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

        def records(self, replay_index: int | None) -> list[dict[str, object]]:
            assert replay_index == 0
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
            replay_index: torch.Tensor,
            non_contiguous: str,
            mode: str,
        ) -> FakeHandle:
            assert name == "mid"
            assert replay_index.dtype == torch.int64
            assert replay_index.shape == ()
            assert len(actions) == 1
            assert actions[0]["kind"] == "record"
            assert non_contiguous == "copy"
            assert mode == "always"
            return FakeHandle()

    monkeypatch.setattr(probe_module._native, "require_native", lambda: FakeNative)
    monkeypatch.setattr(
        probe_module,
        "_create_replay_index",
        lambda device: torch.zeros((), dtype=torch.int64),
    )

    probe = TensorProbe(
        "mid",
        [RecordTensor()],
        non_contiguous="copy",
        when="always",
    )
    tensor = torch.tensor([3.0])

    assert probe(tensor) is tensor
    snapshots = probe.snapshots(synchronize=False)
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
            replay_index: torch.Tensor,
            non_contiguous: str,
            mode: str,
        ) -> FakeHandle:
            assert mode == "capture"
            return FakeHandle()

    monkeypatch.setattr(probe_module._native, "require_native", lambda: FakeNative)
    monkeypatch.setattr(
        probe_module,
        "_create_replay_index",
        lambda device: torch.zeros((), dtype=torch.int64),
    )

    probe = TensorProbe("mid", [PrintTensor()])
    with pytest.raises(TensorMismatchError, match="mismatch"):
        probe.assert_ok(synchronize=False)


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
            replay_index: torch.Tensor,
            non_contiguous: str,
            mode: str,
        ) -> FakeHandle:
            assert name == "mid"
            assert [action["max_items"] for action in actions] == [3]
            assert mode == "capture"
            return FakeHandle()

    monkeypatch.setattr(probe_module._native, "require_native", lambda: FakeNative)
    monkeypatch.setattr(
        probe_module,
        "_create_replay_index",
        lambda device: torch.zeros((), dtype=torch.int64),
    )

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

    def fail_create_replay_index(device: object) -> torch.Tensor:
        raise AssertionError("CUDA should not be initialized for an all-disabled probe")

    monkeypatch.setattr(probe_module._native, "require_native", fail_require_native)
    monkeypatch.setattr(
        probe_module, "_create_replay_index", fail_create_replay_index
    )

    probe = TensorProbe(
        "disabled",
        [
            PrintTensor(enabled=False),
            CompareTensor(object(), enabled=False),
        ],
    )
    tensor = torch.tensor([1.0])

    assert probe(tensor) is tensor
    assert probe.replay_index is None
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


def test_replay_index_property_returns_an_independent_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHandle:
        def close(self) -> None:
            pass

    class FakeNative:
        @staticmethod
        def create_tensor_debug_probe(
            name: str,
            actions: list[dict[str, object]],
            replay_index: torch.Tensor,
            non_contiguous: str,
            mode: str,
        ) -> FakeHandle:
            return FakeHandle()

    counter = torch.zeros((), dtype=torch.int64)
    monkeypatch.setattr(probe_module._native, "require_native", lambda: FakeNative)
    monkeypatch.setattr(
        probe_module, "_create_replay_index", lambda device: counter
    )

    probe = TensorProbe("counter", [PrintTensor()])
    exposed = probe.replay_index

    assert exposed is not None
    assert exposed.data_ptr() != counter.data_ptr()
    exposed.add_(7)
    assert counter.item() == 0
    current = probe.replay_index
    assert current is not None
    assert current.item() == 0

    probe.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = probe.replay_index


def test_replay_index_rejects_cpu_device() -> None:
    with pytest.raises(ValueError, match="CUDA device"):
        probe_module._create_replay_index("cpu")


def test_snapshots_reuses_callback_counter_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHandle:
        def __init__(self) -> None:
            self.records_arguments: list[int | None] = []
            self.clear_count = 0

        def records(self, replay_index: int | None) -> list[dict[str, object]]:
            self.records_arguments.append(replay_index)
            return []

        def clear_records(self) -> None:
            self.clear_count += 1

        def status(self) -> dict[str, object]:
            return {
                "ok": True,
                "message": "",
                "replay_index": 0,
                "invocation_index": -1,
            }

        def close(self) -> None:
            pass

    handle = FakeHandle()

    class FakeNative:
        @staticmethod
        def create_tensor_debug_probe(
            name: str,
            actions: list[dict[str, object]],
            replay_index: torch.Tensor,
            non_contiguous: str,
            mode: str,
        ) -> FakeHandle:
            return handle

    monkeypatch.setattr(probe_module._native, "require_native", lambda: FakeNative)
    monkeypatch.setattr(
        probe_module,
        "_create_replay_index",
        lambda device: torch.zeros((), dtype=torch.int64),
    )
    synchronize_calls: list[tuple[torch.device, object]] = []
    monkeypatch.setattr(
        probe_module,
        "_synchronize_probe_results",
        lambda device, target: synchronize_calls.append((device, target)),
    )

    probe = TensorProbe("combined", [RecordTensor(), PrintTensor()])
    probe._device = torch.device("cuda:0")

    assert probe.snapshots() == []
    assert handle.records_arguments == [None]
    assert synchronize_calls == [(torch.device("cuda:0"), True)]

    synchronize_calls.clear()
    probe.clear_snapshots()
    assert handle.clear_count == 1
    assert synchronize_calls == [(torch.device("cuda:0"), True)]

    synchronize_calls.clear()
    assert probe.status().ok is True
    assert synchronize_calls == [(torch.device("cuda:0"), True)]

    synchronize_calls.clear()
    probe.assert_ok()
    assert synchronize_calls == [(torch.device("cuda:0"), True)]
    probe.close()


def test_synchronization_target_accepts_only_bool_stream_or_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self, device: str) -> None:
            self.device = torch.device(device)

        def synchronize(self) -> None:
            pass

    monkeypatch.setattr(probe_module.torch.cuda, "Stream", FakeStream)

    probe_module._validate_synchronize_target(True)
    probe_module._validate_synchronize_target(False)
    probe_module._validate_synchronize_target(FakeStream("cuda:0"))
    probe_module._validate_synchronize_target(torch.device("cuda:0"))

    for invalid in ("cuda:0", 0, None, object()):
        with pytest.raises(TypeError, match="bool, torch.cuda.Stream, or torch.device"):
            probe_module._validate_synchronize_target(invalid)  # type: ignore[arg-type]


def test_synchronize_probe_results_supports_device_and_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_device = torch.device("cuda:0")
    synchronized: list[torch.device] = []
    monkeypatch.setattr(
        probe_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(
        probe_module.torch.cuda,
        "synchronize",
        lambda device: synchronized.append(device),
    )

    probe_module._synchronize_probe_results(probe_device, True)
    probe_module._synchronize_probe_results(
        probe_device, torch.device("cuda:0")
    )
    probe_module._synchronize_probe_results(probe_device, False)
    assert synchronized == [probe_device, probe_device]

    with pytest.raises(ValueError, match="does not match probe device"):
        probe_module._synchronize_probe_results(
            probe_device, torch.device("cuda:1")
        )
    with pytest.raises(ValueError, match="must identify a CUDA device"):
        probe_module._synchronize_probe_results(
            probe_device, torch.device("cpu")
        )


def test_synchronize_probe_results_supports_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self, device: str) -> None:
            self.device = torch.device(device)
            self.synchronize_count = 0

        def synchronize(self) -> None:
            self.synchronize_count += 1

    monkeypatch.setattr(probe_module.torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(
        probe_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(
        probe_module.torch.cuda,
        "synchronize",
        lambda device: pytest.fail("stream path must not synchronize the device"),
    )

    stream = FakeStream("cuda:0")
    probe_module._synchronize_probe_results(torch.device("cuda:0"), stream)
    assert stream.synchronize_count == 1

    with pytest.raises(ValueError, match="does not match probe device"):
        probe_module._synchronize_probe_results(
            torch.device("cuda:0"), FakeStream("cuda:1")
        )


def test_synchronize_probe_results_rejects_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(
        probe_module.torch.cuda,
        "synchronize",
        lambda device: pytest.fail("capture must be rejected before synchronization"),
    )

    with pytest.raises(RuntimeError, match="during CUDA graph capture"):
        probe_module._synchronize_probe_results(torch.device("cuda:0"), True)

    probe_module._synchronize_probe_results(torch.device("cuda:0"), False)
