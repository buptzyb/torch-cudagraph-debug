"""Allocator-history attribution policy shared by memory analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MissingPolicy = Literal["warn", "error"]


@dataclass(frozen=True)
class MemoryAttributionOptions:
    """Optional allocator-history attribution and report presentation.

    `stack_depth` and `limit` affect text and HTML presentation only.
    Stack identity and structured in-memory, JSON, and CSV results always use
    complete stacks and retain every row.
    """

    stacks: bool = False
    events: bool = False
    lifetimes: bool = False
    on_missing: MissingPolicy = "warn"
    stack_depth: int = 2
    limit: int = 20

    def __post_init__(self) -> None:
        for name in ("stacks", "events", "lifetimes"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if not isinstance(self.on_missing, str) or self.on_missing not in {
            "warn",
            "error",
        }:
            raise ValueError("on_missing must be 'warn' or 'error'")
        if type(self.stack_depth) is not int:
            raise TypeError("stack_depth must be an integer")
        if self.stack_depth < 1:
            raise ValueError("stack_depth must be >= 1")
        if type(self.limit) is not int:
            raise TypeError("limit must be an integer")
        if self.limit < 1:
            raise ValueError("limit must be >= 1")

    def lifetime_options(self) -> "MemoryLifetimeOptions":
        """Return the lifetime policy embedded in this attribution request."""

        return MemoryLifetimeOptions(
            events=self.events,
            on_missing=self.on_missing,
            stack_depth=self.stack_depth,
            limit=self.limit,
        )


@dataclass(frozen=True)
class MemoryLifetimeOptions:
    """Allocator-history and display policy for allocation lifetime analysis.

    ``stack_depth`` and ``limit`` affect text and HTML presentation only. Cohort
    identity and structured JSON/CSV data always use the complete analysis.
    """

    events: bool = True
    on_missing: MissingPolicy = "warn"
    stack_depth: int = 4
    limit: int = 20

    def __post_init__(self) -> None:
        if type(self.events) is not bool:
            raise TypeError("events must be a boolean")
        if not isinstance(self.on_missing, str) or self.on_missing not in {
            "warn",
            "error",
        }:
            raise ValueError("on_missing must be 'warn' or 'error'")
        if type(self.stack_depth) is not int:
            raise TypeError("stack_depth must be an integer")
        if self.stack_depth < 1:
            raise ValueError("stack_depth must be >= 1")
        if type(self.limit) is not int:
            raise TypeError("limit must be an integer")
        if self.limit < 1:
            raise ValueError("limit must be >= 1")
