from __future__ import annotations

import os

import pytest
import torch

from torch_cudagraph_debug import _native
from torch_cudagraph_debug.tensor_debug import (
    CheckAction,
    PrintAction,
    RecordAction,
    TensorCheckError,
    TensorDebugError,
    TensorObservationKey,
    TensorProbe,
)

pytestmark = pytest.mark.gpu


def replay_index_value(probe: TensorProbe) -> int:
    replay_index = probe.replay_index
    assert replay_index is not None
    assert replay_index.dtype == torch.int64
    assert replay_index.shape == ()
    assert replay_index.is_cuda
    return int(replay_index.item())


def debug_resource_counts(probe: TensorProbe) -> dict[str, int]:
    handle = probe._collector._handle
    assert handle is not None
    return {
        str(key): int(value) for key, value in handle._debug_resource_counts().items()
    }


def test_check_inside_cuda_graph() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.ones(4, device="cuda")
    expected = torch.full((4,), 3.0, device="cpu")
    probe = TensorProbe(
        "mid",
        actions=[CheckAction([expected], rtol=1e-5, atol=1e-8)],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = probe(x + 2)

    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    probe.assert_check_ok()
    with pytest.raises(TensorDebugError, match="requires an enabled RecordAction"):
        probe.snapshot()
    assert replay_index_value(probe) == 3
    assert y is not None
    probe.close()


def test_eager_probe_is_noop_by_default() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    view = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4).t()
    wrong_expected = torch.zeros(tuple(view.shape), device="cpu")
    probe = TensorProbe(
        "eager-noop",
        actions=[
            RecordAction(),
            CheckAction([wrong_expected], rtol=0.0, atol=0.0),
        ],
    )

    assert probe(view) is view
    torch.cuda.synchronize()
    with pytest.raises(TensorDebugError, match="no recorded invocation"):
        probe.snapshot()
    probe.assert_check_ok()

    probe.close()


def test_always_mode_runs_in_eager_path() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = x.detach().cpu()
    probe = TensorProbe(
        "eager-always",
        actions=[
            RecordAction(),
            CheckAction([expected], rtol=0.0, atol=0.0),
        ],
        when="always",
    )

    assert probe(x) is x
    torch.cuda.synchronize()
    probe.assert_check_ok()
    snapshots = probe.snapshot()
    assert len(snapshots.observations) == 1
    assert snapshots.replay_index == 0
    assert snapshots.observations[0].invocation_index == 0
    assert torch.equal(snapshots.observations[0].tensor(), expected)

    probe.close()


def test_non_contiguous_copy_inside_cuda_graph() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    base = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    view = base.t()
    expected = view.detach().cpu().contiguous()
    probe = TensorProbe(
        "view",
        actions=[
            RecordAction(),
            CheckAction([expected], rtol=0.0, atol=0.0),
        ],
        non_contiguous="copy",
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = probe(view)

    assert y is view
    g.replay()
    torch.cuda.synchronize()

    probe.assert_check_ok()
    snapshots = probe.snapshot()
    assert len(snapshots.observations) == 1
    assert snapshots.replay_index == 1
    assert snapshots.observations[0].invocation_index == 0
    assert snapshots.observations[0].shape == tuple(view.shape)
    assert snapshots.observations[0].source_device.startswith("cuda")
    assert torch.equal(snapshots.observations[0].tensor(), expected)
    probe.close()


def test_record_slot_is_zero_before_first_replay() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32) + 10
    expected = x.detach().cpu()
    probe = TensorProbe("record-before-replay", actions=[RecordAction()])

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x)

    torch.cuda.synchronize()
    snapshots = probe.snapshot()
    assert len(snapshots.observations) == 1
    assert snapshots.replay_index == 0
    assert snapshots.observations[0].shape == tuple(x.shape)
    assert torch.equal(snapshots.observations[0].tensor(), torch.zeros_like(expected))
    assert replay_index_value(probe) == 0

    g.replay()
    torch.cuda.synchronize()
    replayed = probe.snapshot()
    assert replayed.replay_index == 1
    assert torch.equal(replayed.tensor(), expected)
    assert replay_index_value(probe) == 1
    probe.close()


def test_check_mismatch_is_sticky() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.ones(4, device="cuda")
    expected = torch.zeros(4, device="cpu")
    probe = TensorProbe(
        "bad",
        actions=[CheckAction([expected], rtol=0.0, atol=0.0)],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x)

    g.replay()
    torch.cuda.synchronize()

    with pytest.raises(TensorCheckError, match="mismatch"):
        probe.assert_check_ok()
    status = probe.check_status()
    assert status.ok is False
    assert status.replay_index == 1
    assert status.invocation_index == 0
    assert "mismatch" in status.message
    probe.close()


def test_zero_element_tensor_record() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.empty((0,), device="cuda")
    expected = torch.empty((0,), device="cpu")
    probe = TensorProbe(
        "empty",
        actions=[RecordAction(), CheckAction([expected])],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x)

    g.replay()
    torch.cuda.synchronize()

    probe.assert_check_ok()
    snapshots = probe.snapshot()
    assert len(snapshots.observations) == 1
    assert snapshots.observations[0].invocation_index == 0
    assert snapshots.observations[0].shape == (0,)
    assert snapshots.observations[0].tensor().numel() == 0
    probe.close()


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (torch.float32, [1.0, 2.0, 3.0]),
        (torch.int32, [1, 2, 3]),
        (torch.bool, [True, False, True]),
    ],
)
def test_supported_dtype_record_and_check(
    dtype: torch.dtype, values: list[object]
) -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    expected = torch.tensor(values, dtype=dtype, device="cpu")
    x = expected.to("cuda")
    probe = TensorProbe(
        f"dtype-{dtype}",
        actions=[RecordAction(), CheckAction([expected], rtol=0.0, atol=0.0)],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x)

    g.replay()
    torch.cuda.synchronize()

    probe.assert_check_ok()
    assert torch.equal(probe.snapshot().tensor(), expected)
    probe.close()


def test_unsupported_dtype_errors() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.ones(2, device="cuda", dtype=torch.complex64)
    probe = TensorProbe(
        "complex",
        actions=[RecordAction()],
        when="always",
    )

    with pytest.raises(RuntimeError, match="unsupported dtype"):
        probe(x)

    probe.close()


def test_non_contiguous_default_errors_inside_cuda_graph() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    view = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4).t()
    anchor = torch.empty_like(view.contiguous())
    probe = TensorProbe("view-error", actions=[RecordAction()])

    g = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="non_contiguous|contiguous"):
        with torch.cuda.graph(g):
            anchor.copy_(view)
            probe(view)
    probe.close()


def test_multiple_probes_in_one_cuda_graph() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.ones(4, device="cuda")
    expected_a = torch.full((4,), 2.0, device="cpu")
    expected_b = torch.full((4,), 6.0, device="cpu")
    probe_a = TensorProbe(
        "a",
        actions=[RecordAction(), CheckAction([expected_a], rtol=0.0, atol=0.0)],
    )
    probe_b = TensorProbe(
        "b",
        actions=[RecordAction(), CheckAction([expected_b], rtol=0.0, atol=0.0)],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = probe_a(x + 1)
        probe_b(y * 3)

    g.replay()
    torch.cuda.synchronize()

    probe_a.assert_check_ok()
    probe_b.assert_check_ok()
    assert torch.equal(probe_a.snapshot().tensor(), expected_a)
    assert torch.equal(probe_b.snapshot().tensor(), expected_b)
    assert replay_index_value(probe_a) == 1
    assert replay_index_value(probe_b) == 1
    probe_a.close()
    probe_b.close()


def test_one_probe_multiple_invocations_records_and_offline_compares() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = [torch.arange(4, dtype=torch.float32) + offset for offset in (1, 2, 3)]
    probe = TensorProbe(
        "loop",
        actions=[RecordAction()],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y0 = probe(x + 1)
        y1 = probe(x + 2)
        y2 = probe(x + 3)

    g.replay()
    torch.cuda.synchronize()

    snapshots = probe.snapshot()
    assert len(snapshots.observations) == 3
    assert snapshots.replay_index == 1
    assert [item.invocation_index for item in snapshots.observations] == [0, 1, 2]
    assert all(
        torch.equal(item.tensor(), expected[item.invocation_index])
        for item in snapshots.observations
    )
    assert y0 is not None and y1 is not None and y2 is not None
    probe.close()


def test_named_observations_use_per_name_invocations_and_global_order() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    probe = TensorProbe("named", actions=[RecordAction()])

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        probe(x + 1, name="hidden")
        probe(x + 2, name="logits")
        probe(x + 3, name="hidden")

    graph.replay()
    snapshot = probe.snapshot()

    assert [
        (item.name, item.invocation_index, item.order) for item in snapshot.observations
    ] == [
        ("hidden", 0, 0),
        ("logits", 0, 1),
        ("hidden", 1, 2),
    ]
    assert list(snapshot.by_key) == [
        TensorObservationKey("hidden", 0),
        TensorObservationKey("logits", 0),
        TensorObservationKey("hidden", 1),
    ]
    torch.testing.assert_close(
        snapshot.tensor("hidden", invocation_index=1),
        torch.arange(4, dtype=torch.float32) + 3,
    )
    probe.close()


def test_record_allows_different_shapes_in_one_cuda_graph() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(6, device="cuda", dtype=torch.float32)
    probe = TensorProbe(
        "mixed-record-shapes",
        actions=[RecordAction()],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x[:4] + 1)
        probe((x + 2).reshape(2, 3))

    g.replay()
    torch.cuda.synchronize()

    snapshots = probe.snapshot()
    assert [item.invocation_index for item in snapshots.observations] == [0, 1]
    assert snapshots.observations[0].shape == (4,)
    assert snapshots.observations[1].shape == (2, 3)
    torch.testing.assert_close(
        snapshots.observations[0].tensor(), torch.arange(4, dtype=torch.float32) + 1
    )
    torch.testing.assert_close(
        snapshots.observations[1].tensor(),
        (torch.arange(6, dtype=torch.float32) + 2).reshape(2, 3),
    )

    probe.close()


def test_check_allows_different_shape_and_dtype_in_one_cuda_graph() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x_float = torch.arange(4, device="cuda", dtype=torch.float32)
    x_int = torch.arange(6, device="cuda", dtype=torch.int32).reshape(2, 3)
    probe = TensorProbe(
        "mixed-compare-metadata",
        actions=[
            CheckAction(
                [
                    torch.arange(4, dtype=torch.float32) + 1,
                    torch.arange(6, dtype=torch.int32).reshape(2, 3) + 2,
                ],
                rtol=0.0,
                atol=0.0,
            )
        ],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x_float + 1)
        probe(x_int + 2)

    g.replay()
    torch.cuda.synchronize()

    probe.assert_check_ok()
    with pytest.raises(TensorDebugError, match="requires an enabled RecordAction"):
        probe.snapshot()
    probe.close()


def test_callback_only_probe_keeps_snapshots_empty_for_multiple_invocations() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = [torch.arange(4, dtype=torch.float32) + offset for offset in (1, 2)]
    probe = TensorProbe(
        "compare-only-slots",
        actions=[CheckAction(expected, rtol=0.0, atol=0.0)],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x + 1)
        probe(x + 2)

    g.replay()
    torch.cuda.synchronize()

    probe.assert_check_ok()
    with pytest.raises(TensorDebugError, match="requires an enabled RecordAction"):
        probe.snapshot()
    probe.close()


def test_callback_only_side_stream_invocations_use_private_staging() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = [
        torch.arange(4, dtype=torch.float32) + 20,
        torch.arange(4, dtype=torch.float32) + 10,
    ]
    probe = TensorProbe(
        "side-stream-compare",
        actions=[
            RecordAction(),
            CheckAction(expected, rtol=0.0, atol=0.0),
        ],
    )
    side_stream = torch.cuda.Stream()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        capture_stream = torch.cuda.current_stream()
        side_stream.wait_stream(capture_stream)
        with torch.cuda.stream(side_stream):
            side_value = probe(x + 20)
        main_value = probe(x + 10)
        capture_stream.wait_stream(side_stream)

    g.replay()
    torch.cuda.synchronize()

    probe.assert_check_ok()
    snapshots = probe.snapshot()
    assert snapshots.replay_index == 1
    assert [item.invocation_index for item in snapshots.observations] == [0, 1]
    assert side_value is not None and main_value is not None
    probe.close()


def test_reusing_one_probe_in_second_cuda_graph_capture_errors() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    probe = TensorProbe(
        "single-capture",
        actions=[RecordAction()],
    )

    g0 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g0):
        probe(x + 1)
        probe(x + 2)

    g1 = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="already been captured by a CUDA graph"):
        with torch.cuda.graph(g1):
            probe(x + 1)
    probe.close()


def test_always_mode_eager_calls_do_not_claim_capture_ownership() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = [torch.arange(4, dtype=torch.float32) + offset for offset in (1, 2)]
    probe = TensorProbe(
        "always-before-capture",
        actions=[RecordAction()],
        when="always",
    )

    assert probe(x) is x
    torch.cuda.synchronize()
    eager_snapshot = probe.snapshot()
    assert eager_snapshot.replay_index == 0
    assert replay_index_value(probe) == 0

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x + 1)
        probe(x + 2)

    g.replay()
    torch.cuda.synchronize()

    snapshots = probe.snapshot()
    assert len(snapshots.observations) == 2
    assert snapshots.replay_index == 1
    assert [item.invocation_index for item in snapshots.observations] == [0, 1]
    assert replay_index_value(probe) == 1
    assert torch.equal(snapshots.observations[0].tensor(), expected[0])
    assert torch.equal(snapshots.observations[1].tensor(), expected[1])
    probe.close()


def test_tensor_check_uses_expected_list_by_invocation() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(6, device="cuda", dtype=torch.float32)
    expected = [
        torch.arange(4, dtype=torch.float32) + 1,
        (torch.arange(6, dtype=torch.float32) + 2).reshape(2, 3),
    ]
    probe = TensorProbe(
        "per-slot-compare",
        actions=[
            RecordAction(),
            CheckAction(expected, rtol=0.0, atol=0.0),
        ],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x[:4] + 1)
        probe((x + 2).reshape(2, 3))

    g.replay()
    torch.cuda.synchronize()

    probe.assert_check_ok()
    snapshots = probe.snapshot()
    assert [item.invocation_index for item in snapshots.observations] == [0, 1]
    assert torch.equal(snapshots.observations[0].tensor(), expected[0])
    assert torch.equal(snapshots.observations[1].tensor(), expected[1])
    probe.close()


def test_tensor_check_reports_missing_expected_for_invocation() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    probe = TensorProbe("short-expected", actions=[CheckAction([x.detach().cpu()])])

    g = torch.cuda.CUDAGraph()
    with pytest.raises(
        RuntimeError, match=r"no tensor for observation short-expected\[1\] at order 1"
    ):
        with torch.cuda.graph(g):
            probe(x)
            probe(x)

    torch.cuda.synchronize()
    probe.close()


def test_always_mode_allows_shape_and_dtype_changes_for_slot_zero() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    probe = TensorProbe(
        "shape-contract",
        actions=[RecordAction()],
        when="always",
    )
    first = torch.ones(4, device="cuda", dtype=torch.float32)
    second = torch.arange(5, device="cuda", dtype=torch.float64)

    probe(first)
    probe(second)
    torch.cuda.synchronize()

    snapshots = probe.snapshot()
    assert len(snapshots.observations) == 1
    assert snapshots.observations[0].shape == (5,)
    assert snapshots.observations[0].dtype == torch.float64
    assert torch.equal(
        snapshots.observations[0].tensor(), torch.arange(5, dtype=torch.float64)
    )

    probe.close()


def test_grad_probes_inside_cuda_graph_backward() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    class GradDebugModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 3, bias=False)
            self.fc2 = torch.nn.Linear(3, 2, bias=False)
            self.activation_grad_probe = TensorProbe(
                "hidden.grad",
                actions=[RecordAction()],
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = self.fc1(x)
            self.activation_grad_probe.watch_grad(hidden)
            return self.fc2(hidden).sum()

    torch.manual_seed(1234)
    model = GradDebugModule().cuda()
    static_x = torch.randn(2, 4, device="cuda")
    weight_grad_probe = TensorProbe(
        "fc1.weight.grad.hook",
        actions=[RecordAction()],
    )
    final_grad_probe = TensorProbe(
        "fc1.weight.grad.final",
        actions=[RecordAction()],
    )
    weight_grad_probe.watch_grad(model.fc1.weight)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            loss = model(static_x)
            loss.backward()
    del loss
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    for probe in (model.activation_grad_probe, weight_grad_probe, final_grad_probe):
        with pytest.raises(TensorDebugError, match="no recorded invocation"):
            probe.snapshot()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        model.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            loss = model(static_x)
            loss.backward()
            if model.fc1.weight.grad is None:
                raise AssertionError("fc1.weight.grad should exist after backward")
            final_grad_probe(model.fc1.weight.grad)
    torch.cuda.current_stream().wait_stream(capture_stream)

    g.replay()
    torch.cuda.synchronize()

    activation_snapshots = model.activation_grad_probe.snapshot()
    weight_snapshots = weight_grad_probe.snapshot()
    final_snapshots = final_grad_probe.snapshot()
    assert len(activation_snapshots.observations) == 1
    assert activation_snapshots.observations[0].shape == (2, 3)
    assert activation_snapshots.observations[0].dtype == torch.float32
    assert len(weight_snapshots.observations) == 1
    assert weight_snapshots.observations[0].shape == tuple(model.fc1.weight.shape)
    assert weight_snapshots.observations[0].dtype == torch.float32
    assert len(final_snapshots.observations) == 1
    assert final_snapshots.observations[0].shape == tuple(model.fc1.weight.shape)
    assert final_snapshots.observations[0].dtype == torch.float32

    model.activation_grad_probe.close()
    weight_grad_probe.close()
    final_grad_probe.close()


def test_record_replay_index_advances_and_old_snapshots_stay_stable() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.zeros(4, device="cuda", dtype=torch.float32)
    probe = TensorProbe("record-replay-index", actions=[RecordAction()])

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x)

    x.fill_(1)
    g.replay()
    first = probe.snapshot()

    x.fill_(2)
    g.replay()
    second = probe.snapshot(
        synchronize=torch.device("cuda", torch.cuda.current_device())
    )

    assert first.replay_index == 1
    assert second.replay_index == 2
    assert torch.equal(first.tensor(), torch.ones(4))
    assert torch.equal(second.tensor(), torch.full((4,), 2.0))
    assert replay_index_value(probe) == 2

    exposed = probe.replay_index
    assert exposed is not None
    exposed.add_(100)
    assert replay_index_value(probe) == 2
    probe.close()


def test_snapshots_can_synchronize_non_default_replay_stream() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.zeros(4, device="cuda", dtype=torch.float32)
    probe = TensorProbe("stream-synchronize", actions=[RecordAction()])

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x)

    replay_stream = torch.cuda.Stream()
    with torch.cuda.stream(replay_stream):
        x.fill_(3)
        g.replay()

    snapshot = probe.snapshot(synchronize=replay_stream)
    assert snapshot.replay_index == 1
    assert torch.equal(snapshot.tensor(), torch.full((4,), 3.0))
    probe.close()


def test_queued_replays_report_latest_replay_index() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    probe = TensorProbe("queued-replays", actions=[RecordAction()])

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x)

    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    snapshot = probe.snapshot()
    assert snapshot.replay_index == 3
    assert replay_index_value(probe) == 3
    probe.close()


def test_record_and_check_share_first_mismatch_replay_index() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.ones(4, device="cuda")
    probe = TensorProbe(
        "record-check-index",
        actions=[
            RecordAction(),
            CheckAction(torch.ones(4), rtol=0.0, atol=0.0),
        ],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x, name="activation")

    replay_stream = torch.cuda.current_stream()
    g.replay()
    probe.assert_check_ok(synchronize=replay_stream)
    assert probe.snapshot(synchronize=False).replay_index == 1

    x.fill_(2)
    g.replay()

    status = probe.check_status(synchronize=replay_stream)
    assert status.ok is False
    assert status.replay_index == 2
    assert status.order == 0
    assert status.name == "activation"
    assert status.invocation_index == 0
    assert status.key == TensorObservationKey("activation", 0)
    assert probe.snapshot(synchronize=False).replay_index == 2
    assert replay_index_value(probe) == 2
    probe.close()


def test_print_every_uses_graph_replay_index() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    probe = TensorProbe(
        "print-every",
        actions=[PrintAction(max_items=1, every=2, summary=False)],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x, name="printed")

    read_fd, write_fd = os.pipe()
    original_stderr = os.dup(2)
    try:
        os.dup2(write_fd, 2)
        os.close(write_fd)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
    finally:
        os.dup2(original_stderr, 2)
        os.close(original_stderr)

    with os.fdopen(read_fd, encoding="utf-8") as captured:
        stderr = captured.read()

    lines = [
        line
        for line in stderr.splitlines()
        if "torch-cudagraph-debug:print-every" in line
    ]
    assert len(lines) == 1
    assert "replay=2 observation=printed[0] order=0" in lines[0]
    assert replay_index_value(probe) == 3
    probe.close()


def test_probe_device_must_match_active_tensor_device() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")

    probe = TensorProbe(
        "device-mismatch",
        actions=[RecordAction()],
        when="always",
        device="cuda:0",
    )
    x = torch.ones(1, device="cuda:1")

    with pytest.raises(RuntimeError, match="created on cuda:0.*cuda:1"):
        probe(x)
    probe.close()


def test_eager_callback_payloads_do_not_accumulate() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    value = torch.arange(16, device="cuda", dtype=torch.float32)
    probe = TensorProbe(
        "bounded-eager-callbacks",
        actions=[CheckAction(value.detach().cpu(), rtol=0.0, atol=0.0)],
        when="always",
    )

    for _ in range(256):
        probe(value)

    stream = torch.cuda.current_stream()
    probe.assert_check_ok(synchronize=stream)
    counts = debug_resource_counts(probe)
    assert counts["captured_payloads"] == 0
    assert counts["eager_callbacks_in_flight"] == 0
    assert counts["source_owners"] == 0
    probe.close(synchronize=False)


def test_eager_non_contiguous_temporaries_and_retired_staging_are_reclaimed() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    base = torch.arange(64 * 64, device="cuda", dtype=torch.float32).reshape(64, 64)
    probe = TensorProbe(
        "bounded-eager-copy",
        actions=[RecordAction()],
        non_contiguous="copy",
        when="always",
    )

    latest = None
    for size in (8, 16, 32, 48, 64):
        latest = base[:size, :size].t()
        assert not latest.is_contiguous()
        probe(latest)

    assert latest is not None
    stream = torch.cuda.current_stream()
    snapshot = probe.snapshot(synchronize=stream)
    torch.testing.assert_close(snapshot.tensor(), latest.detach().cpu().contiguous())
    counts = debug_resource_counts(probe)
    assert counts["retired_staging"] == 0
    assert counts["source_owners"] == 0
    assert counts["eager_callbacks_in_flight"] == 0
    probe.close(synchronize=False)


def test_always_mode_rejects_a_second_eager_stream() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    value = torch.arange(4, device="cuda", dtype=torch.float32)
    probe = TensorProbe(
        "single-eager-stream",
        actions=[RecordAction()],
        when="always",
    )
    first_stream = torch.cuda.current_stream()
    probe(value)

    second_stream = torch.cuda.Stream()
    with torch.cuda.stream(second_stream):
        with pytest.raises(RuntimeError, match="different eager CUDA stream"):
            probe(value)

    first_stream.synchronize()
    probe.close(synchronize=False)


def test_always_mode_rejects_eager_calls_after_capture() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    value = torch.arange(4, device="cuda", dtype=torch.float32)
    probe = TensorProbe(
        "eager-then-capture",
        actions=[RecordAction()],
        when="always",
    )
    probe(value)
    torch.cuda.current_stream().synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        probe(value + 1)
    graph.replay()
    torch.cuda.current_stream().synchronize()

    with pytest.raises(RuntimeError, match="eager calls after capture"):
        probe(value)
    probe.close(synchronize=False)


def test_capture_callback_payload_count_is_fixed_per_invocation() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    value = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = [value.detach().cpu(), (value + 1).detach().cpu()]
    probe = TensorProbe(
        "fixed-capture-payloads",
        actions=[CheckAction(expected, rtol=0.0, atol=0.0)],
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        probe(value)
        probe(value + 1)

    assert debug_resource_counts(probe)["captured_payloads"] == 2
    for _ in range(3):
        graph.replay()
    stream = torch.cuda.current_stream()
    probe.assert_check_ok(synchronize=stream)
    counts = debug_resource_counts(probe)
    assert counts["captured_payloads"] == 2
    assert counts["eager_callbacks_in_flight"] == 0
    probe.close(synchronize=False)


def test_close_without_synchronization_rejects_pending_eager_callback() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    first = torch.arange(4, device="cuda", dtype=torch.float32)
    second = torch.arange(8, device="cuda", dtype=torch.float32)
    probe = TensorProbe(
        "pending-eager-callback",
        actions=[PrintAction(every=1_000_000, summary=False)],
        when="always",
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
        probe(first)
        probe(second)

    before = debug_resource_counts(probe)
    assert before["eager_callbacks_in_flight"] == 2
    assert before["retired_staging"] == 1
    with pytest.raises(RuntimeError, match="eager callback.*pending"):
        probe.close(synchronize=False)
    after = debug_resource_counts(probe)
    assert after["eager_callbacks_in_flight"] == 2
    assert after["retired_staging"] == 1
    probe.close(synchronize=stream)


def test_close_is_rejected_during_capture() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    value = torch.arange(4, device="cuda", dtype=torch.float32)
    probe = TensorProbe("close-during-capture", actions=[RecordAction()])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        probe(value)
        with pytest.raises(RuntimeError, match="during CUDA graph capture"):
            probe.close(synchronize=False)

    torch.cuda.synchronize()
    probe.close(synchronize=False)
