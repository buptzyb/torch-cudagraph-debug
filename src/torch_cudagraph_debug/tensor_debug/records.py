"""Record types returned by tensor debug probes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class TensorSnapshot:
    """A CPU tensor snapshot recorded from CUDA Graph replay."""

    probe_name: str
    replay_index: int
    tensor: torch.Tensor
    shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    device: str = ""
    invocation_index: int = 0

    def __post_init__(self) -> None:
        if self.shape is None:
            object.__setattr__(self, "shape", tuple(self.tensor.shape))
        if self.dtype is None:
            object.__setattr__(self, "dtype", self.tensor.dtype)
