from __future__ import annotations

import pytest
import torch

from torch_cudagraph_debug import _native
from torch_cudagraph_debug.tensor_debug import (
    TensorProbe,
    CheckAction,
    TensorCheckError,
    TensorDebugError,
    PrintAction,
    RecordAction,
)

pytestmark = pytest.mark.gpu


def replay_index_value(probe: TensorProbe) -> int:
    replay_index = probe.replay_index
    assert replay_index is not None
    assert replay_index.dtype == torch.int64
    assert replay_index.shape == ()
    assert replay_index.is_cuda
    return int(replay_index.item())


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


def test_clear_snapshot_zeroes_latest_buffers_without_dropping_slots() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = x.detach().cpu() + 1
    probe = TensorProbe("clear-snapshot", actions=[RecordAction()])

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x + 1)

    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(probe.snapshot().tensor(), expected)

    probe.clear_snapshot()
    assert torch.equal(probe.snapshot().tensor(), torch.zeros_like(expected))

    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(probe.snapshot().tensor(), expected)
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
    with pytest.raises(RuntimeError, match="no tensor for invocation 1"):
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
        probe(x)

    replay_stream = torch.cuda.current_stream()
    g.replay()
    probe.assert_check_ok(synchronize=replay_stream)
    assert probe.snapshot(synchronize=False).replay_index == 1

    x.fill_(2)
    g.replay()

    status = probe.check_status(synchronize=replay_stream)
    assert status.ok is False
    assert status.replay_index == 2
    assert status.invocation_index == 0
    assert probe.snapshot(synchronize=False).replay_index == 2
    assert replay_index_value(probe) == 2
    probe.close()


def test_print_every_uses_graph_replay_index(capfd: pytest.CaptureFixture[str]) -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    probe = TensorProbe(
        "print-every",
        actions=[PrintAction(max_items=1, every=2, summary=False)],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x)

    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    lines = [
        line
        for line in capfd.readouterr().err.splitlines()
        if "torch-cudagraph-debug:print-every" in line
    ]
    assert len(lines) == 1
    assert "replay=2 invocation=0" in lines[0]
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
