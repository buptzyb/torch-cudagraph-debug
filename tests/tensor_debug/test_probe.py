from __future__ import annotations

import pytest
import torch

from torch_cudagraph_debug import _native
from torch_cudagraph_debug._errors import NativeExtensionUnavailableError
from torch_cudagraph_debug.tensor_debug import (
    CheckAction,
    PrintAction,
    RecordAction,
    TensorCheckError,
    TensorCheckStatus,
    TensorDebugError,
    TensorObservationKey,
    TensorProbe,
    compare_snapshots,
)
from torch_cudagraph_debug.tensor_debug import _collector as collector_module


@pytest.fixture(autouse=True)
def _default_to_not_capturing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        collector_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )


def _install_fake_native(monkeypatch: pytest.MonkeyPatch, handle: object) -> None:
    class FakeNative:
        @staticmethod
        def create_tensor_debug_probe(
            name: str,
            actions: list[dict[str, object]],
            replay_index: torch.Tensor,
            non_contiguous: str,
            mode: str,
        ) -> object:
            return handle

    monkeypatch.setattr(collector_module._native, "require_native", lambda: FakeNative)
    monkeypatch.setattr(
        collector_module,
        "_create_replay_index",
        lambda device: torch.zeros((), dtype=torch.int64),
    )


def test_failed_enqueue_does_not_consume_invocation_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingHandle:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []
            self.fail_next = True

        def enqueue(
            self, tensor: torch.Tensor, name: str, invocation_index: int
        ) -> torch.Tensor:
            self.calls.append((name, invocation_index))
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("tensor for observation x must be contiguous")
            return tensor

        def _reclaim_retired_staging(self) -> None:
            pass

        def close(self) -> None:
            pass

    handle = FailingHandle()
    _install_fake_native(monkeypatch, handle)
    probe = TensorProbe("x", [RecordAction()])
    monkeypatch.setattr(
        probe,
        "_classify_invocation",
        lambda tensor, name: (probe._capture_invocation_counts.get(name, 0), True),
    )
    tensor = torch.tensor([1.0])

    with pytest.raises(RuntimeError, match="contiguous"):
        probe(tensor, name="x")
    assert probe(tensor, name="x") is tensor
    # The rejected call must not burn invocation index 0; the retry reuses it.
    assert handle.calls == [("x", 0), ("x", 0)]
    assert probe._capture_invocation_counts == {"x": 1}


def test_record_only_collect_is_rejected_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordHandle:
        def __init__(self) -> None:
            self.reclaims = 0

        def enqueue(
            self, tensor: torch.Tensor, name: str, invocation_index: int
        ) -> torch.Tensor:
            return tensor

        def observations(self, replay_index: int | None) -> list[dict[str, object]]:
            return []

        def _reclaim_retired_staging(self) -> None:
            self.reclaims += 1

        def close(self) -> None:
            pass

    handle = RecordHandle()
    _install_fake_native(monkeypatch, handle)
    probe = TensorProbe("x", [RecordAction()])
    monkeypatch.setattr(
        collector_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )

    with pytest.raises(RuntimeError, match="during CUDA\\s+graph capture"):
        probe.snapshot(synchronize=False)
    # Freeing retired staging is also a CUDA call; it must be deferred too.
    assert handle.reclaims == 0


class _ObservationsHandle:
    def __init__(self, observations: list[dict[str, object]]) -> None:
        self._observations = observations

    def enqueue(
        self, tensor: torch.Tensor, name: str, invocation_index: int
    ) -> torch.Tensor:
        return tensor

    def observations(self, replay_index: int | None) -> list[dict[str, object]]:
        return self._observations

    def _reclaim_retired_staging(self) -> None:
        pass

    def close(self) -> None:
        pass


def _observation_row(
    name: str, order: int, invocation_index: int, **extra: object
) -> dict[str, object]:
    return {
        "name": name,
        "order": order,
        "replay_index": 0,
        "invocation_index": invocation_index,
        "shape": (2,),
        "device": "cuda:0",
        "tensor": torch.tensor([1.0, 2.0]),
        **extra,
    }


def test_snapshot_exposes_eager_overwrite_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _ObservationsHandle([_observation_row("x", 0, 0, eager_overwrites=2)])
    _install_fake_native(monkeypatch, handle)
    probe = TensorProbe("x", [RecordAction()], when="always")
    probe(torch.tensor([1.0]), name="x")

    snapshot = probe.snapshot(synchronize=False)
    assert snapshot.eager_overwrites == (("x", 2),)
    assert dict(snapshot.eager_overwrite_counts) == {"x": 2}

    # Fakes without the field default to zero overwrites.
    plain = _ObservationsHandle([_observation_row("x", 0, 0)])
    _install_fake_native(monkeypatch, plain)
    other = TensorProbe("x", [RecordAction()], when="always")
    other(torch.tensor([1.0]), name="x")
    assert other.snapshot(synchronize=False).eager_overwrites == ()


def test_compare_warns_when_eager_sample_faces_multiple_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled_handle = _ObservationsHandle(
        [_observation_row("x", 0, 0, eager_overwrites=1)]
    )
    _install_fake_native(monkeypatch, sampled_handle)
    sampled_probe = TensorProbe("x", [RecordAction()], when="always")
    sampled_probe(torch.tensor([1.0]), name="x")
    sampled = sampled_probe.snapshot(synchronize=False)

    multi_handle = _ObservationsHandle(
        [
            _observation_row("x", 0, 0),
            _observation_row("x", 1, 1),
        ]
    )
    _install_fake_native(monkeypatch, multi_handle)
    multi_probe = TensorProbe("x", [RecordAction()], when="always")
    monkeypatch.setattr(
        multi_probe,
        "_classify_invocation",
        lambda tensor, name: (
            multi_probe._capture_invocation_counts.get(name, 0),
            True,
        ),
    )
    multi_probe(torch.tensor([1.0]), name="x")
    multi_probe(torch.tensor([1.0]), name="x")
    multi = multi_probe.snapshot(synchronize=False)

    reference_comparison = compare_snapshots(sampled, multi)
    assert reference_comparison.warnings == (
        "reference observation 'x' is an eager sample overwritten 1 time(s) "
        "while the candidate recorded multiple invocations; the sample keeps "
        "only the latest occurrence; collect the reference side with "
        "TensorRecorder when complete invocation alignment matters",
    )

    candidate_comparison = compare_snapshots(multi, sampled)
    assert candidate_comparison.warnings == (
        "candidate observation 'x' is an eager sample overwritten 1 time(s) "
        "while the reference recorded multiple invocations; the sample keeps "
        "only the latest occurrence; collect the candidate side with "
        "TensorRecorder when complete invocation alignment matters",
    )
    # No warning when the multi-invocation side is absent.
    quiet = compare_snapshots(sampled, sampled_probe.snapshot(synchronize=False))
    assert quiet.warnings == ()


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

        def enqueue(
            self,
            tensor: torch.Tensor,
            name: str,
            invocation_index: int,
        ) -> torch.Tensor:
            self.input = (tensor, name, invocation_index)
            return tensor

        def observations(self, replay_index: int | None) -> list[dict[str, object]]:
            assert replay_index == 0
            return [
                {
                    "name": self.input[1],
                    "order": 0,
                    "replay_index": 7,
                    "invocation_index": 0,
                    "shape": (2,),
                    "device": "cuda:0",
                    "tensor": torch.tensor([1.0, 2.0]),
                }
            ]

        def check_status(self) -> dict[str, object]:
            return {
                "ok": True,
                "message": "",
                "replay_index": 0,
                "order": -1,
                "name": None,
                "invocation_index": -1,
            }

        def _reclaim_retired_staging(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

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
            assert name == "mid"
            assert replay_index.dtype == torch.int64
            assert replay_index.shape == ()
            assert len(actions) == 1
            assert actions[0]["kind"] == "record"
            assert non_contiguous == "copy"
            assert mode == "always"
            return handle

    monkeypatch.setattr(collector_module._native, "require_native", lambda: FakeNative)
    monkeypatch.setattr(
        collector_module,
        "_create_replay_index",
        lambda device: torch.zeros((), dtype=torch.int64),
    )

    probe = TensorProbe(
        "mid",
        [RecordAction()],
        non_contiguous="copy",
        when="always",
    )
    tensor = torch.tensor([3.0])

    assert probe(tensor) is tensor
    assert handle.input is not None
    recorded_tensor, recorded_name, recorded_invocation = handle.input
    assert recorded_tensor is tensor
    assert (recorded_name, recorded_invocation) == ("mid", 0)

    assert probe(tensor, name="activation") is tensor
    assert handle.input[1:] == ("activation", 0)
    snapshot = probe.snapshot(synchronize=False)
    assert snapshot.probe_name == "mid"
    assert snapshot.replay_index == 7
    assert len(snapshot.observations) == 1
    observation = snapshot.observation("activation")
    assert observation.invocation_index == 0
    assert observation.shape == (2,)
    assert observation.dtype == torch.float32
    assert observation.source_device == "cuda:0"
    assert torch.equal(observation.tensor(), torch.tensor([1.0, 2.0]))
    probe.assert_check_ok()
    probe.close(synchronize=False)

    with pytest.raises(RuntimeError, match="closed"):
        probe.snapshot()


def test_rejected_close_keeps_watch_grad_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHandle:
        def __init__(self) -> None:
            self.calls = 0

        def enqueue(
            self,
            tensor: torch.Tensor,
            name: str,
            invocation_index: int,
        ) -> torch.Tensor:
            self.calls += 1
            return tensor

        def _reclaim_retired_staging(self) -> None:
            pass

        def close(self) -> None:
            pass

    handle = FakeHandle()
    _install_fake_native(monkeypatch, handle)

    probe = TensorProbe("grads", [RecordAction()], when="always")
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    assert probe.watch_grad(source) is not None

    with pytest.raises(TypeError):
        probe.close(synchronize=123)

    # A rejected close must leave the probe fully open: the gradient hooks
    # must keep firing, not silently vanish.
    assert probe._closed is False
    source.sum().backward()
    assert handle.calls == 1


def test_close_removes_watch_grad_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHandle:
        def _reclaim_retired_staging(self) -> None:
            pass

        def close(self) -> None:
            pass

    _install_fake_native(monkeypatch, FakeHandle())

    probe = TensorProbe("grads", [RecordAction()], when="always")
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    assert probe.watch_grad(source) is not None
    probe.close(synchronize=False)

    # A hook surviving close() would fire against the closed collector and
    # blow up inside autograd; close() must remove every registered hook.
    source.sum().backward()
    assert source.grad is not None
    assert torch.equal(source.grad, torch.ones(2))


def test_first_capture_replaces_eager_stride_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHandle:
        def enqueue(
            self,
            tensor: torch.Tensor,
            name: str,
            invocation_index: int,
        ) -> torch.Tensor:
            return tensor

        def close(self) -> None:
            pass

    handle = FakeHandle()
    _install_fake_native(monkeypatch, handle)
    capturing = [False]
    monkeypatch.setattr(
        collector_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: capturing[0],
    )
    probe = TensorProbe("metadata", [RecordAction()], when="always")
    tensor = torch.arange(4)

    probe(tensor, name="x")
    probe(tensor, name="y")
    assert set(probe._collector._source_strides) == {("x", 0), ("y", 0)}

    capturing[0] = True
    probe(tensor, name="captured")
    assert set(probe._collector._source_strides) == {("captured", 0)}

    capturing[0] = False
    probe.close(synchronize=False)


def test_assert_check_ok_raises_check_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHandle:
        def check_status(self) -> dict[str, object]:
            return {
                "ok": False,
                "message": "probe mid mismatch",
                "replay_index": 3,
                "order": 1,
                "name": "mid",
                "invocation_index": 1,
            }

        def _reclaim_retired_staging(self) -> None:
            pass

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

    monkeypatch.setattr(collector_module._native, "require_native", lambda: FakeNative)
    monkeypatch.setattr(
        collector_module,
        "_create_replay_index",
        lambda device: torch.zeros((), dtype=torch.int64),
    )

    probe = TensorProbe("mid", [PrintAction()])
    status = probe.check_status(synchronize=False)
    assert status.order == 1
    assert status.name == "mid"
    assert status.invocation_index == 1
    assert status.key == TensorObservationKey("mid", 1)
    with pytest.raises(TensorCheckError, match="mismatch"):
        probe.assert_check_ok(synchronize=False)


def test_probe_validates_non_contiguous_policy() -> None:
    with pytest.raises(ValueError, match="non_contiguous"):
        TensorProbe(  # type: ignore[arg-type]
            "mid",
            [PrintAction()],
            non_contiguous="bad",
        )


def test_probe_validates_when() -> None:
    with pytest.raises(ValueError, match="when"):
        TensorProbe(  # type: ignore[arg-type]
            "mid",
            [PrintAction()],
            when="sometimes",
        )


def test_probe_close_validates_synchronization_target() -> None:
    probe = TensorProbe("disabled", [PrintAction(enabled=False)])

    with pytest.raises(TypeError, match="bool, torch.cuda.Stream, or torch.device"):
        probe.close(synchronize="cuda:0")  # type: ignore[arg-type]

    probe.close(synchronize=False)


def test_probe_filters_disabled_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHandle:
        def _reclaim_retired_staging(self) -> None:
            pass

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

    monkeypatch.setattr(collector_module._native, "require_native", lambda: FakeNative)
    monkeypatch.setattr(
        collector_module,
        "_create_replay_index",
        lambda device: torch.zeros((), dtype=torch.int64),
    )

    probe = TensorProbe(
        "mid",
        [
            PrintAction(max_items=1, enabled=False),
            PrintAction(max_items=3, enabled=True),
        ],
    )
    probe.close(synchronize=False)


def test_all_disabled_probe_is_noop_without_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_require_native() -> object:
        raise AssertionError("native should not be loaded for an all-disabled probe")

    def fail_create_replay_index(device: object) -> torch.Tensor:
        raise AssertionError("CUDA should not be initialized for an all-disabled probe")

    monkeypatch.setattr(collector_module._native, "require_native", fail_require_native)
    monkeypatch.setattr(
        collector_module, "_create_replay_index", fail_create_replay_index
    )

    probe = TensorProbe(
        "disabled",
        [
            PrintAction(enabled=False),
            CheckAction(object(), enabled=False),
        ],
    )
    tensor = torch.tensor([1.0])

    assert probe(tensor) is tensor
    assert probe.replay_index is None
    with pytest.raises(TensorDebugError, match="requires an enabled RecordAction"):
        probe.snapshot()
    assert probe.check_status() == TensorCheckStatus(
        ok=True,
        message="",
        replay_index=0,
        order=-1,
        name=None,
        invocation_index=-1,
    )
    probe.assert_check_ok()
    probe.close(synchronize=False)

    with pytest.raises(RuntimeError, match="closed"):
        probe(tensor)


def test_watch_grad_noops_for_tensor_without_grad() -> None:
    probe = TensorProbe("disabled", [PrintAction(enabled=False)])
    tensor = torch.tensor([1.0])

    with pytest.raises(ValueError, match="non-empty"):
        probe.watch_grad(tensor, name="")
    assert probe.watch_grad(tensor) is None

    with pytest.raises(RuntimeError, match="does not require grad"):
        probe.watch_grad(tensor, strict=True)


def test_watch_grad_probes_but_returns_original_grad() -> None:
    class ReturningWrongProbe(TensorProbe):
        def __init__(self) -> None:
            self.name = "grad"
            self._closed = False
            self._grad_handles = []
            self.calls: list[tuple[str | None, torch.Tensor]] = []

        def __call__(
            self,
            tensor: torch.Tensor,
            *,
            name: str | None = None,
        ) -> torch.Tensor:
            self.calls.append((name, tensor.detach().clone()))
            return torch.zeros_like(tensor)

    probe = ReturningWrongProbe()
    x = torch.tensor([2.0, -3.0], requires_grad=True)

    handle = probe.watch_grad(x, name="activation.grad")
    assert handle is not None
    y = (x * torch.tensor([4.0, 5.0])).sum()
    y.backward()

    assert len(probe.calls) == 1
    assert probe.calls[0][0] == "activation.grad"
    assert torch.equal(probe.calls[0][1], torch.tensor([4.0, 5.0]))
    assert torch.equal(x.grad, torch.tensor([4.0, 5.0]))


def test_watch_grad_returned_handle_can_remove_hook() -> None:
    class CountingProbe(TensorProbe):
        def __init__(self) -> None:
            self.name = "grad"
            self._closed = False
            self._grad_handles = []
            self.calls = 0

        def __call__(
            self,
            tensor: torch.Tensor,
            *,
            name: str | None = None,
        ) -> torch.Tensor:
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
        def _reclaim_retired_staging(self) -> None:
            pass

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
    monkeypatch.setattr(collector_module._native, "require_native", lambda: FakeNative)
    monkeypatch.setattr(
        collector_module, "_create_replay_index", lambda device: counter
    )

    probe = TensorProbe("counter", [PrintAction()])
    exposed = probe.replay_index

    assert exposed is not None
    assert exposed.data_ptr() != counter.data_ptr()
    exposed.add_(7)
    assert counter.item() == 0
    current = probe.replay_index
    assert current is not None
    assert current.item() == 0

    probe.close(synchronize=False)
    with pytest.raises(RuntimeError, match="closed"):
        _ = probe.replay_index


def test_replay_index_rejects_cpu_device() -> None:
    with pytest.raises(ValueError, match="CUDA device"):
        collector_module._create_replay_index("cpu")


def test_snapshot_reuses_callback_counter_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHandle:
        def __init__(self) -> None:
            self.observation_arguments: list[int | None] = []

        def observations(self, replay_index: int | None) -> list[dict[str, object]]:
            self.observation_arguments.append(replay_index)
            return []

        def check_status(self) -> dict[str, object]:
            return {
                "ok": True,
                "message": "",
                "replay_index": 0,
                "order": -1,
                "name": None,
                "invocation_index": -1,
            }

        def _reclaim_retired_staging(self) -> None:
            pass

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

    monkeypatch.setattr(collector_module._native, "require_native", lambda: FakeNative)
    monkeypatch.setattr(
        collector_module,
        "_create_replay_index",
        lambda device: torch.zeros((), dtype=torch.int64),
    )
    synchronize_calls: list[tuple[torch.device, object]] = []
    monkeypatch.setattr(
        collector_module,
        "synchronize_tensor_results",
        lambda device, target: synchronize_calls.append((device, target)),
    )

    probe = TensorProbe("combined", [RecordAction(), PrintAction()])
    probe._collector._device = torch.device("cuda:0")

    with pytest.raises(TensorDebugError, match="no recorded invocation"):
        probe.snapshot()
    assert handle.observation_arguments == [None]
    assert synchronize_calls == [(torch.device("cuda:0"), True)]

    synchronize_calls.clear()
    assert probe.check_status().ok is True
    assert synchronize_calls == [(torch.device("cuda:0"), True)]

    synchronize_calls.clear()
    probe.assert_check_ok()
    assert synchronize_calls == [(torch.device("cuda:0"), True)]
    probe.close(synchronize=False)


def test_synchronization_target_accepts_only_bool_stream_or_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self, device: str) -> None:
            self.device = torch.device(device)

        def synchronize(self) -> None:
            pass

    monkeypatch.setattr(collector_module.torch.cuda, "Stream", FakeStream)

    collector_module.validate_synchronize_target(True)
    collector_module.validate_synchronize_target(False)
    collector_module.validate_synchronize_target(FakeStream("cuda:0"))
    collector_module.validate_synchronize_target(torch.device("cuda:0"))

    for invalid in ("cuda:0", 0, None, object()):
        with pytest.raises(TypeError, match="bool, torch.cuda.Stream, or torch.device"):
            collector_module.validate_synchronize_target(invalid)  # type: ignore[arg-type]


def test_synchronize_tensor_results_supports_device_and_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_device = torch.device("cuda:0")
    synchronized: list[torch.device] = []
    monkeypatch.setattr(
        collector_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(
        collector_module.torch.cuda,
        "synchronize",
        lambda device: synchronized.append(device),
    )

    collector_module.synchronize_tensor_results(probe_device, True)
    collector_module.synchronize_tensor_results(probe_device, torch.device("cuda:0"))
    collector_module.synchronize_tensor_results(probe_device, False)
    assert synchronized == [probe_device, probe_device]

    with pytest.raises(ValueError, match="does not match collector device"):
        collector_module.synchronize_tensor_results(
            probe_device, torch.device("cuda:1")
        )
    with pytest.raises(ValueError, match="must identify a CUDA device"):
        collector_module.synchronize_tensor_results(probe_device, torch.device("cpu"))


def test_synchronize_tensor_results_supports_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self, device: str) -> None:
            self.device = torch.device(device)
            self.synchronize_count = 0

        def synchronize(self) -> None:
            self.synchronize_count += 1

    monkeypatch.setattr(collector_module.torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(
        collector_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(
        collector_module.torch.cuda,
        "synchronize",
        lambda device: pytest.fail("stream path must not synchronize the device"),
    )

    stream = FakeStream("cuda:0")
    collector_module.synchronize_tensor_results(torch.device("cuda:0"), stream)
    assert stream.synchronize_count == 1

    with pytest.raises(ValueError, match="does not match collector device"):
        collector_module.synchronize_tensor_results(
            torch.device("cuda:0"), FakeStream("cuda:1")
        )


def test_synchronize_tensor_results_rejects_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collector_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(
        collector_module.torch.cuda,
        "synchronize",
        lambda device: pytest.fail("capture must be rejected before synchronization"),
    )

    with pytest.raises(RuntimeError, match="during CUDA Graph capture"):
        collector_module.synchronize_tensor_results(torch.device("cuda:0"), True)

    collector_module.synchronize_tensor_results(torch.device("cuda:0"), False)


def test_tensor_probe_name_must_be_a_non_empty_string() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        TensorProbe(1, [RecordAction()])  # type: ignore[arg-type]
