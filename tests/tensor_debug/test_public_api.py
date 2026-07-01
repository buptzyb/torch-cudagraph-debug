from __future__ import annotations

import torch_cudagraph_debug.tensor_debug as tensor_debug
from torch_cudagraph_debug.tensor_debug import postprocess


def test_tensor_stable_facade() -> None:
    assert tensor_debug.__all__ == [
        "TensorProbe",
        "PrintTensor",
        "RecordTensor",
        "CompareTensor",
        "TensorSnapshot",
        "TensorProbeStatus",
        "TensorDebugError",
        "TensorMismatchError",
    ]


def test_tensor_postprocess_facade() -> None:
    assert postprocess.__all__ == ["export_snapshots_to_tensorboard"]
