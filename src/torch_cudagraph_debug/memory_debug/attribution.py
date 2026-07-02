"""Allocator-history attribution policy shared by memory analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MissingPolicy = Literal["warn", "error"]


@dataclass(frozen=True)
class MemoryAttributionOptions:
    """Optional allocator-history attribution for memory analysis."""

    stacks: bool = False
    events: bool = False
    lifetimes: bool = False
    on_missing: MissingPolicy = "warn"
    stack_depth: int = 2
    limit: int = 20

    def __post_init__(self) -> None:
        if self.on_missing not in {"warn", "error"}:
            raise ValueError("on_missing must be 'warn' or 'error'")
        if self.stack_depth < 1:
            raise ValueError("stack_depth must be >= 1")
        if self.limit < 1:
            raise ValueError("limit must be >= 1")
