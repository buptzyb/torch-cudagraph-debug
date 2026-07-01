"""Tensor value debugging helpers for eager and CUDA Graph execution."""

from .actions import CompareTensor, PrintTensor, RecordTensor
from .comparison import (
    TensorCompareOptions,
    TensorComparison,
    TensorDifference,
    TensorRunComparison,
    TensorSeriesComparison,
    compare_points,
    compare_runs,
    compare_series,
)
from .errors import (
    TensorBundleError,
    TensorDebugError,
    TensorMismatchError,
    TensorOwnershipError,
    TensorPayloadUnavailableError,
)
from .probe import TensorProbe
from .records import TensorProbeStatus, TensorSnapshot
from .runs import (
    TensorObservation,
    TensorPoint,
    TensorRecorder,
    TensorRun,
    TensorValueSummary,
)

__all__ = [
    "TensorProbe",
    "TensorRecorder",
    "PrintTensor",
    "RecordTensor",
    "CompareTensor",
    "TensorSnapshot",
    "TensorProbeStatus",
    "TensorRun",
    "TensorPoint",
    "TensorObservation",
    "TensorValueSummary",
    "TensorCompareOptions",
    "TensorDifference",
    "TensorComparison",
    "TensorRunComparison",
    "TensorSeriesComparison",
    "compare_points",
    "compare_runs",
    "compare_series",
    "TensorDebugError",
    "TensorMismatchError",
    "TensorBundleError",
    "TensorOwnershipError",
    "TensorPayloadUnavailableError",
]
