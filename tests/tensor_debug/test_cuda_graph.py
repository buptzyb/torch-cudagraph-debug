from __future__ import annotations

import pytest
import torch

from torch_cudagraph_debug import _native
from torch_cudagraph_debug.tensor_debug import (
    CudaGraphTensorProbe,
    TensorCompare,
    TensorCompareMismatchError,
    TensorRecord,
)


pytestmark = pytest.mark.gpu


def test_compare_inside_cuda_graph() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.ones(4, device="cuda")
    expected = torch.full((4,), 3.0, device="cpu")
    probe = CudaGraphTensorProbe(
        "mid",
        actions=[TensorCompare([expected], rtol=1e-5, atol=1e-8)],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = probe(x + 2)

    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    probe.assert_ok()
    assert probe.records() == []
    assert y is not None
    probe.close()


def test_eager_probe_is_noop_by_default() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    view = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4).t()
    wrong_expected = torch.zeros(tuple(view.shape), device="cpu")
    probe = CudaGraphTensorProbe(
        "eager-noop",
        actions=[
            TensorRecord(),
            TensorCompare([wrong_expected], rtol=0.0, atol=0.0),
        ],
    )

    assert probe(view) is view
    torch.cuda.synchronize()
    assert probe.records() == []
    probe.assert_ok()

    probe.close()


def test_always_mode_runs_in_eager_path() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = x.detach().cpu()
    probe = CudaGraphTensorProbe(
        "eager-always",
        actions=[
            TensorRecord(),
            TensorCompare([expected], rtol=0.0, atol=0.0),
        ],
        mode="always",
    )

    assert probe(x) is x
    torch.cuda.synchronize()
    probe.assert_ok()
    records = probe.records()
    assert len(records) == 1
    assert records[0].replay_index == 0
    assert records[0].invocation_index == 0
    assert torch.equal(records[0].tensor, expected)

    probe.close()


def test_non_contiguous_copy_inside_cuda_graph() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    base = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    view = base.t()
    expected = view.detach().cpu().contiguous()
    probe = CudaGraphTensorProbe(
        "view",
        actions=[
            TensorRecord(),
            TensorCompare([expected], rtol=0.0, atol=0.0),
        ],
        non_contiguous="copy",
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = probe(view)

    assert y is view
    g.replay()
    torch.cuda.synchronize()

    probe.assert_ok()
    records = probe.records()
    assert len(records) == 1
    assert records[0].replay_index == 0
    assert records[0].invocation_index == 0
    assert records[0].shape == tuple(view.shape)
    assert records[0].device.startswith("cuda")
    assert torch.equal(records[0].tensor, expected)
    probe.close()


def test_record_slot_is_zero_before_first_replay() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32) + 10
    expected = x.detach().cpu()
    probe = CudaGraphTensorProbe("record-before-replay", actions=[TensorRecord()])

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x)

    torch.cuda.synchronize()
    records = probe.records()
    assert len(records) == 1
    assert records[0].shape == tuple(x.shape)
    assert torch.equal(records[0].tensor, torch.zeros_like(expected))

    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(probe.records()[0].tensor, expected)
    probe.close()


def test_compare_mismatch_is_sticky() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.ones(4, device="cuda")
    expected = torch.zeros(4, device="cpu")
    probe = CudaGraphTensorProbe(
        "bad",
        actions=[TensorCompare([expected], rtol=0.0, atol=0.0)],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x)

    g.replay()
    torch.cuda.synchronize()

    with pytest.raises(TensorCompareMismatchError, match="mismatch"):
        probe.assert_ok()
    status = probe.status()
    assert status["ok"] is False
    assert status["replay_index"] == 1
    assert status["invocation_index"] == 0
    assert "mismatch" in status["message"]
    probe.close()


def test_zero_element_tensor_record() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.empty((0,), device="cuda")
    expected = torch.empty((0,), device="cpu")
    probe = CudaGraphTensorProbe(
        "empty",
        actions=[TensorRecord(), TensorCompare([expected])],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x)

    g.replay()
    torch.cuda.synchronize()

    probe.assert_ok()
    records = probe.records()
    assert len(records) == 1
    assert records[0].invocation_index == 0
    assert records[0].shape == (0,)
    assert records[0].tensor.numel() == 0
    probe.close()


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (torch.float32, [1.0, 2.0, 3.0]),
        (torch.int32, [1, 2, 3]),
        (torch.bool, [True, False, True]),
    ],
)
def test_supported_dtype_record_and_compare(dtype: torch.dtype, values: list[object]) -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    expected = torch.tensor(values, dtype=dtype, device="cpu")
    x = expected.to("cuda")
    probe = CudaGraphTensorProbe(
        f"dtype-{dtype}",
        actions=[TensorRecord(), TensorCompare([expected], rtol=0.0, atol=0.0)],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x)

    g.replay()
    torch.cuda.synchronize()

    probe.assert_ok()
    assert torch.equal(probe.records()[0].tensor, expected)
    probe.close()


def test_unsupported_dtype_errors() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.ones(2, device="cuda", dtype=torch.complex64)
    probe = CudaGraphTensorProbe(
        "complex",
        actions=[TensorRecord()],
        mode="always",
    )

    with pytest.raises(RuntimeError, match="unsupported dtype"):
        probe(x)

    probe.close()


def test_non_contiguous_default_errors_inside_cuda_graph() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    view = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4).t()
    anchor = torch.empty_like(view.contiguous())
    probe = CudaGraphTensorProbe("view-error", actions=[TensorRecord()])

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
    probe_a = CudaGraphTensorProbe(
        "a",
        actions=[TensorRecord(), TensorCompare([expected_a], rtol=0.0, atol=0.0)],
    )
    probe_b = CudaGraphTensorProbe(
        "b",
        actions=[TensorRecord(), TensorCompare([expected_b], rtol=0.0, atol=0.0)],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = probe_a(x + 1)
        probe_b(y * 3)

    g.replay()
    torch.cuda.synchronize()

    probe_a.assert_ok()
    probe_b.assert_ok()
    assert torch.equal(probe_a.records()[0].tensor, expected_a)
    assert torch.equal(probe_b.records()[0].tensor, expected_b)
    probe_a.close()
    probe_b.close()


def test_one_probe_multiple_invocations_records_and_offline_compares() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = [torch.arange(4, dtype=torch.float32) + offset for offset in (1, 2, 3)]
    probe = CudaGraphTensorProbe(
        "loop",
        actions=[TensorRecord()],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y0 = probe(x + 1)
        y1 = probe(x + 2)
        y2 = probe(x + 3)

    g.replay()
    torch.cuda.synchronize()

    records = probe.records()
    assert len(records) == 3
    assert [record.replay_index for record in records] == [0, 0, 0]
    assert [record.invocation_index for record in records] == [0, 1, 2]
    assert all(
        torch.equal(record.tensor, expected[record.invocation_index]) for record in records
    )
    assert y0 is not None and y1 is not None and y2 is not None
    probe.close()


def test_record_allows_different_shapes_in_one_cuda_graph() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(6, device="cuda", dtype=torch.float32)
    probe = CudaGraphTensorProbe(
        "mixed-record-shapes",
        actions=[TensorRecord()],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x[:4] + 1)
        probe((x + 2).reshape(2, 3))

    g.replay()
    torch.cuda.synchronize()

    records = probe.records()
    assert [record.invocation_index for record in records] == [0, 1]
    assert records[0].shape == (4,)
    assert records[1].shape == (2, 3)
    torch.testing.assert_close(records[0].tensor, torch.arange(4, dtype=torch.float32) + 1)
    torch.testing.assert_close(
        records[1].tensor,
        (torch.arange(6, dtype=torch.float32) + 2).reshape(2, 3),
    )

    probe.close()


def test_compare_allows_different_shape_and_dtype_in_one_cuda_graph() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x_float = torch.arange(4, device="cuda", dtype=torch.float32)
    x_int = torch.arange(6, device="cuda", dtype=torch.int32).reshape(2, 3)
    probe = CudaGraphTensorProbe(
        "mixed-compare-metadata",
        actions=[
            TensorCompare(
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

    probe.assert_ok()
    assert probe.records() == []
    probe.close()


def test_callback_only_probe_keeps_records_empty_for_multiple_invocations() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = [torch.arange(4, dtype=torch.float32) + offset for offset in (1, 2)]
    probe = CudaGraphTensorProbe(
        "compare-only-slots",
        actions=[TensorCompare(expected, rtol=0.0, atol=0.0)],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x + 1)
        probe(x + 2)

    g.replay()
    torch.cuda.synchronize()

    probe.assert_ok()
    assert probe.records() == []
    probe.close()


def test_callback_only_side_stream_invocations_use_private_staging() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = [
        torch.arange(4, dtype=torch.float32) + 20,
        torch.arange(4, dtype=torch.float32) + 10,
    ]
    probe = CudaGraphTensorProbe(
        "side-stream-compare",
        actions=[TensorCompare(expected, rtol=0.0, atol=0.0)],
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

    probe.assert_ok()
    assert probe.records() == []
    assert side_value is not None and main_value is not None
    probe.close()


def test_reusing_one_probe_in_second_cuda_graph_capture_errors() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    probe = CudaGraphTensorProbe(
        "single-capture",
        actions=[TensorRecord()],
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
    probe = CudaGraphTensorProbe(
        "always-before-capture",
        actions=[TensorRecord()],
        mode="always",
    )

    assert probe(x) is x
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x + 1)
        probe(x + 2)

    g.replay()
    torch.cuda.synchronize()

    records = probe.records()
    assert len(records) == 2
    assert [record.invocation_index for record in records] == [0, 1]
    assert torch.equal(records[0].tensor, expected[0])
    assert torch.equal(records[1].tensor, expected[1])
    probe.close()


def test_record_clear_zeroes_latest_buffers_without_dropping_slots() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    expected = x.detach().cpu() + 1
    probe = CudaGraphTensorProbe("clear-record", actions=[TensorRecord()])

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x + 1)

    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(probe.records()[0].tensor, expected)

    probe.clear_records()
    assert torch.equal(probe.records()[0].tensor, torch.zeros_like(expected))

    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(probe.records()[0].tensor, expected)
    probe.close()


def test_tensor_compare_uses_expected_list_by_invocation() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(6, device="cuda", dtype=torch.float32)
    expected = [
        torch.arange(4, dtype=torch.float32) + 1,
        (torch.arange(6, dtype=torch.float32) + 2).reshape(2, 3),
    ]
    probe = CudaGraphTensorProbe(
        "per-slot-compare",
        actions=[
            TensorRecord(),
            TensorCompare(expected, rtol=0.0, atol=0.0),
        ],
    )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        probe(x[:4] + 1)
        probe((x + 2).reshape(2, 3))

    g.replay()
    torch.cuda.synchronize()

    probe.assert_ok()
    records = probe.records()
    assert [record.invocation_index for record in records] == [0, 1]
    assert torch.equal(records[0].tensor, expected[0])
    assert torch.equal(records[1].tensor, expected[1])
    probe.close()


def test_tensor_compare_reports_missing_expected_for_invocation() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    probe = CudaGraphTensorProbe("short-expected", actions=[TensorCompare([x.detach().cpu()])])

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

    probe = CudaGraphTensorProbe(
        "shape-contract",
        actions=[TensorRecord()],
        mode="always",
    )
    first = torch.ones(4, device="cuda", dtype=torch.float32)
    second = torch.arange(5, device="cuda", dtype=torch.float64)

    probe(first)
    probe(second)
    torch.cuda.synchronize()

    records = probe.records()
    assert len(records) == 1
    assert records[0].shape == (5,)
    assert records[0].dtype == torch.float64
    assert torch.equal(records[0].tensor, torch.arange(5, dtype=torch.float64))

    probe.close()


def test_grad_probes_inside_cuda_graph_backward() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")

    class GradDebugModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 3, bias=False)
            self.fc2 = torch.nn.Linear(3, 2, bias=False)
            self.activation_grad_probe = CudaGraphTensorProbe(
                "hidden.grad",
                actions=[TensorRecord()],
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = self.fc1(x)
            hidden = self.activation_grad_probe.attach_grad(hidden)
            return self.fc2(hidden).sum()

    torch.manual_seed(1234)
    model = GradDebugModule().cuda()
    static_x = torch.randn(2, 4, device="cuda")
    weight_grad_probe = CudaGraphTensorProbe(
        "fc1.weight.grad.hook",
        actions=[TensorRecord()],
    )
    final_grad_probe = CudaGraphTensorProbe(
        "fc1.weight.grad.final",
        actions=[TensorRecord()],
    )
    weight_grad_probe.attach_grad(model.fc1.weight)

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

    assert model.activation_grad_probe.records() == []
    assert weight_grad_probe.records() == []
    assert final_grad_probe.records() == []

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

    activation_records = model.activation_grad_probe.records()
    weight_records = weight_grad_probe.records()
    final_records = final_grad_probe.records()
    assert len(activation_records) == 1
    assert activation_records[0].shape == (2, 3)
    assert activation_records[0].dtype == torch.float32
    assert len(weight_records) == 1
    assert weight_records[0].shape == tuple(model.fc1.weight.shape)
    assert weight_records[0].dtype == torch.float32
    assert len(final_records) == 1
    assert final_records[0].shape == tuple(model.fc1.weight.shape)
    assert final_records[0].dtype == torch.float32

    model.activation_grad_probe.close()
    weight_grad_probe.close()
    final_grad_probe.close()
