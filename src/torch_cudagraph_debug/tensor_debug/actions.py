"""Action configuration objects for CUDA Graph tensor probes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

NonContiguousPolicy = Literal["error", "copy"]


def validate_non_contiguous_policy(policy: str) -> NonContiguousPolicy:
    """Validate the non-contiguous tensor handling policy."""

    if policy not in {"error", "copy"}:
        raise ValueError('non_contiguous must be either "error" or "copy"')
    return policy  # type: ignore[return-value]


@dataclass(frozen=True)
class PrintTensor:
    """Print a compact tensor summary from a CUDA Graph host callback."""

    max_items: int = 16
    every: int = 1
    summary: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_items < 0:
            raise ValueError("max_items must be non-negative")
        if self.every <= 0:
            raise ValueError("every must be positive")

    def _to_native(self) -> dict[str, Any]:
        return {
            "kind": "print",
            "max_items": int(self.max_items),
            "every": int(self.every),
            "summary": bool(self.summary),
            "enabled": bool(self.enabled),
        }


@dataclass(frozen=True)
class RecordTensor:
    """Record the latest replay snapshot without a CUDA host callback."""

    enabled: bool = True

    def _to_native(self) -> dict[str, Any]:
        return {
            "kind": "record",
            "enabled": bool(self.enabled),
        }


@dataclass(frozen=True)
class CompareTensor:
    """Compare replay snapshots with per-invocation CPU or NumPy ground truth."""

    expected: Any
    rtol: float = 1e-5
    atol: float = 1e-8
    equal_nan: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.rtol < 0:
            raise ValueError("rtol must be non-negative")
        if self.atol < 0:
            raise ValueError("atol must be non-negative")

    def _to_native(self) -> dict[str, Any]:
        import numpy as np
        import torch

        expected_items = self.expected
        if isinstance(expected_items, (torch.Tensor, np.ndarray)):
            expected_items = [expected_items]
        elif isinstance(expected_items, (str, bytes)) or not isinstance(
            expected_items, Sequence
        ):
            raise TypeError(
                "CompareTensor expected must be a CPU tensor, NumPy array, or a "
                "sequence of them"
            )
        if len(expected_items) == 0:
            raise ValueError("CompareTensor expected sequence must be non-empty")

        expected_tensors: list[torch.Tensor] = []
        for index, expected in enumerate(expected_items):
            if isinstance(expected, np.ndarray):
                expected = torch.from_numpy(expected)
            if not isinstance(expected, torch.Tensor):
                raise TypeError(
                    f"CompareTensor expected[{index}] must be a CPU torch.Tensor "
                    "or NumPy array"
                )
            if expected.device.type != "cpu":
                raise ValueError(f"CompareTensor expected[{index}] must be on CPU")
            expected_tensors.append(expected.detach().contiguous())

        return {
            "kind": "compare",
            "expected": expected_tensors,
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "equal_nan": bool(self.equal_nan),
            "enabled": bool(self.enabled),
        }
