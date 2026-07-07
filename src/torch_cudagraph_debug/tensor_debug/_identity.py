"""Stable tensor observation identity."""

from __future__ import annotations

from dataclasses import dataclass


def validate_observation_name(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("tensor observation name must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class TensorObservationKey:
    """Semantic identity for one named invocation within a point or snapshot."""

    name: str
    invocation_index: int

    def __post_init__(self) -> None:
        validate_observation_name(self.name)
        if type(self.invocation_index) is not int or self.invocation_index < 0:
            raise ValueError("invocation_index must be a non-negative integer")

    @property
    def label(self) -> str:
        return f"{self.name}[{self.invocation_index}]"
