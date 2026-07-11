"""Action configuration objects for eager and CUDA Graph tensor probes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch

from ._identity import TensorObservationKey

NonContiguousPolicy = Literal["error", "copy"]
TensorExpectedValue: TypeAlias = torch.Tensor | np.ndarray
TensorExpected: TypeAlias = (
    TensorExpectedValue
    | Sequence[TensorExpectedValue]
    | Mapping[TensorObservationKey, TensorExpectedValue]
)


def validate_non_contiguous_policy(policy: str) -> NonContiguousPolicy:
    """Validate the non-contiguous tensor handling policy."""

    if not isinstance(policy, str):
        raise TypeError("non_contiguous must be a string")
    if policy not in {"error", "copy"}:
        raise ValueError('non_contiguous must be either "error" or "copy"')
    return policy  # type: ignore[return-value]


@dataclass(frozen=True)
class PrintAction:
    """Print a compact tensor summary through a CUDA host callback.

    Each line goes to stderr and always carries the replay index, the
    observation name and invocation, the enqueue order, the dtype, and the
    shape. ``every`` throttles printing: a captured observation prints when
    the graph replay index is divisible by it, while an eager observation
    prints on every ``every``-th eager callback of the probe, counted from
    one across all observations. ``max_items`` caps how many leading
    elements appear in the printed value listing; longer tensors are
    truncated with an ellipsis. ``summary=True`` adds numel, the min, max,
    and mean over finite values, and the NaN and inf counts;
    ``summary=False`` prints only the header, shape, and leading values.
    """

    max_items: int = 16
    every: int = 1
    summary: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        if type(self.max_items) is not int:
            raise TypeError("max_items must be an integer")
        if type(self.every) is not int:
            raise TypeError("every must be an integer")
        if type(self.summary) is not bool:
            raise TypeError("summary must be a boolean")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if self.max_items < 0:
            raise ValueError("max_items must be non-negative")
        if self.every <= 0:
            raise ValueError("every must be positive")

    def _to_native(self) -> dict[str, object]:
        return {
            "kind": "print",
            "max_items": int(self.max_items),
            "every": int(self.every),
            "summary": bool(self.summary),
            "enabled": bool(self.enabled),
        }


@dataclass(frozen=True)
class RecordAction:
    """Stage the latest eligible tensor values without a CUDA host callback."""

    enabled: bool = True

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")

    def _to_native(self) -> dict[str, object]:
        return {
            "kind": "record",
            "enabled": bool(self.enabled),
        }


@dataclass(frozen=True)
class CheckAction:
    """Check eligible eager or replay values against CPU or NumPy ground truth.

    Positional entries bind eager slots by first-use name order and captured
    slots by global capture order; a single value is never broadcast. A
    ``TensorObservationKey`` mapping binds by ``(name, invocation_index)``.
    """

    expected: TensorExpected
    rtol: float = 1e-5
    atol: float = 1e-8
    equal_nan: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.rtol, bool) or not isinstance(self.rtol, (int, float)):
            raise TypeError("rtol must be a finite number")
        if isinstance(self.atol, bool) or not isinstance(self.atol, (int, float)):
            raise TypeError("atol must be a finite number")
        if not math.isfinite(float(self.rtol)):
            raise ValueError("rtol must be finite")
        if not math.isfinite(float(self.atol)):
            raise ValueError("atol must be finite")
        if type(self.equal_nan) is not bool:
            raise TypeError("equal_nan must be a boolean")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if self.rtol < 0:
            raise ValueError("rtol must be non-negative")
        if self.atol < 0:
            raise ValueError("atol must be non-negative")

    def _to_native(self) -> dict[str, object]:

        def normalize(expected: object, label: str) -> torch.Tensor:
            if isinstance(expected, np.ndarray):
                expected = torch.from_numpy(expected)
            if not isinstance(expected, torch.Tensor):
                raise TypeError(
                    f"CheckAction {label} must be a CPU torch.Tensor or NumPy array"
                )
            if expected.device.type != "cpu":
                raise ValueError(f"CheckAction {label} must be on CPU")
            return expected.detach().contiguous()

        native_expected: list[object]
        if isinstance(self.expected, Mapping):
            if not self.expected:
                raise ValueError("CheckAction expected mapping must be non-empty")
            native_expected = []
            for key, expected in self.expected.items():
                if not isinstance(key, TensorObservationKey):
                    raise TypeError(
                        "CheckAction expected mapping keys must be TensorObservationKey"
                    )
                native_expected.append(
                    {
                        "name": key.name,
                        "invocation_index": key.invocation_index,
                        "tensor": normalize(expected, key.label),
                    }
                )
        else:
            expected_items = self.expected
            if isinstance(expected_items, (torch.Tensor, np.ndarray)):
                expected_items = [expected_items]
            elif isinstance(expected_items, (str, bytes)) or not isinstance(
                expected_items, Sequence
            ):
                raise TypeError(
                    "CheckAction expected must be a CPU tensor, NumPy array, "
                    "a sequence, or a TensorObservationKey mapping"
                )
            if len(expected_items) == 0:
                raise ValueError("CheckAction expected sequence must be non-empty")
            native_expected = [
                normalize(expected, f"expected[{index}]")
                for index, expected in enumerate(expected_items)
            ]

        return {
            "kind": "check",
            "expected": native_expected,
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "equal_nan": bool(self.equal_nan),
            "enabled": bool(self.enabled),
        }
