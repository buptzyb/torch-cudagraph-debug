from __future__ import annotations

import torch_cudagraph_debug.tensor_debug as tensor_debug
from torch_cudagraph_debug.tensor_debug import postprocess


def test_tensor_exact_public_facade() -> None:
    assert tensor_debug.__all__ == [
        "TensorProbe",
        "TensorRecorder",
        "PrintAction",
        "RecordAction",
        "CheckAction",
        "TensorProbeSnapshot",
        "TensorCheckStatus",
        "TensorRun",
        "TensorPoint",
        "TensorObservation",
        "TensorObservationKey",
        "TensorValueSummary",
        "TensorComparisonOptions",
        "TensorObservationComparison",
        "TensorSnapshotComparison",
        "TensorPointComparison",
        "TensorRunComparison",
        "TensorPointSeriesComparison",
        "compare_snapshots",
        "compare_points",
        "compare_runs",
        "compare_point_series",
        "TensorDebugError",
        "TensorCheckError",
        "TensorComparisonError",
        "TensorBundleError",
        "TensorOwnershipError",
        "TensorPayloadUnavailableError",
    ]


def test_tensor_postprocess_facade() -> None:
    assert postprocess.__all__ == ["export_snapshots_to_tensorboard"]
