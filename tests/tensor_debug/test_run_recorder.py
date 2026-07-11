from __future__ import annotations

import pytest
import torch

from torch_cudagraph_debug import _native
from torch_cudagraph_debug.tensor_debug import (
    TensorDebugError,
    TensorRecorder,
    compare_points,
)

pytestmark = pytest.mark.gpu


def _require_gpu_extension() -> None:
    if not torch.cuda.is_available() or not _native.extension_available():
        pytest.skip("requires CUDA and built torch-cudagraph-debug native extension")


def test_eager_recorder_preserves_repeated_named_invocations() -> None:
    _require_gpu_extension()
    stream = torch.cuda.current_stream()
    recorder = TensorRecorder(execution="eager", name="eager")
    x = torch.arange(4, dtype=torch.float32, device="cuda")

    with recorder.record_point("forward", synchronize=stream):
        assert recorder.observe(x + 1, name="hidden") is not None
        recorder.observe(x + 2, name="hidden", payload="summary")

    run = recorder.finish()
    first = run["forward"].observation("hidden", 0)
    second = run["forward"].observation("hidden", 1)
    assert run["forward"].replay_index is None
    assert second.payload == "summary"
    assert torch.equal(first.tensor(), torch.arange(4) + 1)
    assert second.summary.mean == pytest.approx(3.5)
    recorder.close()


def test_eager_and_cuda_graph_runs_compare_and_replays_form_a_series() -> None:
    _require_gpu_extension()
    stream = torch.cuda.current_stream()
    static_x = torch.arange(4, dtype=torch.float32, device="cuda")

    eager_recorder = TensorRecorder(execution="eager", name="eager")
    with eager_recorder.record_point("forward", synchronize=stream):
        eager_first = eager_recorder.observe(static_x + 1, name="hidden")
        eager_recorder.observe(eager_first * 2, name="hidden")
    eager = eager_recorder.finish()
    eager_recorder.close()

    graph_recorder = TensorRecorder(execution="cuda_graph", name="cuda-graph")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_first = graph_recorder.observe(static_x + 1, name="hidden")
        graph_recorder.observe(graph_first * 2, name="hidden")

    with graph_recorder.record_point("replay-1", synchronize=stream):
        graph.replay()
    with pytest.raises(TensorDebugError, match="new CUDA Graph replay"):
        with graph_recorder.record_point("stale", synchronize=stream):
            pass

    with graph_recorder.record_point("replay-2", synchronize=stream):
        graph.replay()

    candidate = graph_recorder.finish()
    assert [item.invocation_index for item in candidate["replay-1"].observations] == [
        0,
        1,
    ]
    assert candidate["replay-1"].replay_index == 1
    assert candidate["replay-2"].replay_index == 2
    assert compare_points(eager["forward"], candidate["replay-1"]).ok
    assert compare_points(eager["forward"], candidate["replay-2"]).ok
    graph_recorder.close()


def test_cuda_graph_recorder_captures_activation_and_gradient() -> None:
    _require_gpu_extension()
    torch.manual_seed(7)
    recorder = TensorRecorder(
        execution="cuda_graph",
        name="backward",
        non_contiguous="copy",
    )
    model = torch.nn.Linear(4, 3, bias=False).cuda()
    static_x = torch.randn(2, 4, device="cuda")
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(capture_stream):
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            warmup_hidden = model(static_x)
            warmup_hidden.sum().backward()
    del warmup_hidden
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.current_stream().synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        model.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph):
            hidden = recorder.observe(model(static_x), name="activation")
            recorder.watch_grad(hidden, name="activation.grad", strict=True)
            hidden.sum().backward()
    replay_stream = torch.cuda.current_stream()
    replay_stream.wait_stream(capture_stream)

    with recorder.record_point("forward-backward", synchronize=replay_stream):
        graph.replay()

    run = recorder.finish()
    point = run["forward-backward"]
    assert point.observation("activation").shape == (2, 3)
    gradient = point.observation("activation.grad").tensor()
    assert torch.equal(gradient, torch.ones_like(gradient))
    recorder.close()


def test_cuda_graph_recorder_out_of_scope_cpu_tensor_is_a_noop() -> None:
    _require_gpu_extension()
    recorder = TensorRecorder(execution="cuda_graph", name="noop")
    cpu_tensor = torch.ones(2)

    assert recorder.observe(cpu_tensor, name="ignored") is cpu_tensor
    recorder.close()

    strict = TensorRecorder(execution="cuda_graph", name="strict", strict_scope=True)
    with pytest.raises(TensorDebugError, match="active CUDA Graph capture"):
        strict.observe(cpu_tensor, name="ignored")
    strict.close()


def test_cuda_graph_recorder_failed_enqueue_does_not_consume_invocation() -> None:
    _require_gpu_extension()
    recorder = TensorRecorder(execution="cuda_graph", name="rollback")
    assert recorder._collector is not None
    handle = recorder._collector._handle
    assert handle is not None
    handle._debug_fail_next_enqueue("host_preparation")
    value = torch.arange(4, dtype=torch.float32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(
            RuntimeError, match="injected tensor debug host preparation"
        ):
            recorder.observe(value, name="hidden")
        recorder.observe(value + 1, name="hidden")

    stream = torch.cuda.current_stream()
    with recorder.record_point("replay", synchronize=stream):
        graph.replay()

    run = recorder.finish()
    observation = run["replay"].observation("hidden")
    assert observation.invocation_index == 0
    torch.testing.assert_close(observation.tensor(), (value + 1).cpu())

    del graph
    recorder.close(synchronize=False)
