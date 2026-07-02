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
        assert recorder.observe("hidden", x + 1) is not None
        recorder.observe("hidden", x + 2, payload="summary")

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
        eager_first = eager_recorder.observe("hidden", static_x + 1)
        eager_recorder.observe("hidden", eager_first * 2)
    eager = eager_recorder.finish()
    eager_recorder.close()

    graph_recorder = TensorRecorder(execution="cuda_graph", name="cuda-graph")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_first = graph_recorder.observe("hidden", static_x + 1)
        graph_recorder.observe("hidden", graph_first * 2)

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
            hidden = recorder.observe("activation", model(static_x))
            recorder.watch_grad("activation.grad", hidden, strict=True)
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
