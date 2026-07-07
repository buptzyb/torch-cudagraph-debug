from __future__ import annotations

import torch_cudagraph_debug.tensor_debug as tensor_debug
from torch_cudagraph_debug.tensor_debug import postprocess


def test_tensor_exact_public_facade() -> None:
    assert set(tensor_debug.__all__) == {
        "CheckAction",
        "ComparisonMode",
        "ComparisonStatus",
        "DTypePolicy",
        "LayoutPolicy",
        "ObservationComparisonKind",
        "PrintAction",
        "RecordAction",
        "TensorBundleError",
        "TensorCheckError",
        "TensorCheckStatus",
        "TensorComparisonError",
        "TensorComparisonOptions",
        "TensorDebugError",
        "TensorExpected",
        "TensorExpectedValue",
        "TensorObservation",
        "TensorObservationComparison",
        "TensorObservationKey",
        "TensorOwnershipError",
        "TensorPayloadUnavailableError",
        "TensorPoint",
        "TensorPointComparison",
        "TensorPointSeriesComparison",
        "TensorProbe",
        "TensorProbeSnapshot",
        "TensorRankPointSummary",
        "TensorRankRunComparison",
        "TensorRecorder",
        "TensorRun",
        "TensorRunComparison",
        "TensorRunGroup",
        "TensorRunGroupComparison",
        "TensorRunGroupSummary",
        "TensorSnapshotComparison",
        "TensorValueSummary",
        "compare_point_series",
        "compare_points",
        "compare_run_groups",
        "compare_runs",
        "compare_snapshots",
    }
    assert len(tensor_debug.__all__) == len(set(tensor_debug.__all__))
    assert all(hasattr(tensor_debug, name) for name in tensor_debug.__all__)


def test_tensor_postprocess_facade() -> None:
    assert postprocess.__all__ == ["export_snapshots_to_tensorboard"]
