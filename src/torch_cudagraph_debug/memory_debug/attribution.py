"""Allocator-history analysis and presentation policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MissingPolicy = Literal["warn", "error"]


@dataclass(frozen=True)
class MemoryEvidenceStatus:
    """Request and availability state for one optional history analysis."""

    requested: bool
    available: bool
    complete: bool

    def __post_init__(self) -> None:
        for name in ("requested", "available", "complete"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if not self.requested and (self.available or self.complete):
            raise ValueError("unrequested evidence cannot be available or complete")
        if self.complete and not self.available:
            raise ValueError("complete evidence must be available")

    @classmethod
    def not_requested(cls) -> "MemoryEvidenceStatus":
        return cls(requested=False, available=False, complete=False)

    def to_dict(self) -> dict[str, bool]:
        return {
            "requested": self.requested,
            "available": self.available,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class MemoryAttributionStatus:
    """Evidence status for stack, event, and lifetime attribution."""

    stacks: MemoryEvidenceStatus = field(
        default_factory=MemoryEvidenceStatus.not_requested
    )
    events: MemoryEvidenceStatus = field(
        default_factory=MemoryEvidenceStatus.not_requested
    )
    lifetimes: MemoryEvidenceStatus = field(
        default_factory=MemoryEvidenceStatus.not_requested
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "stacks": self.stacks.to_dict(),
            "events": self.events.to_dict(),
            "lifetimes": self.lifetimes.to_dict(),
        }


@dataclass(frozen=True)
class MemoryDisplayOptions:
    """Presentation-only limits for text and HTML reports."""

    stack_depth: int = 2
    limit: int = 20

    def __post_init__(self) -> None:
        if type(self.stack_depth) is not int:
            raise TypeError("stack_depth must be an integer")
        if self.stack_depth < 1:
            raise ValueError("stack_depth must be >= 1")
        if type(self.limit) is not int:
            raise TypeError("limit must be an integer")
        if self.limit < 1:
            raise ValueError("limit must be >= 1")


@dataclass(frozen=True)
class MemoryAttributionOptions:
    """Optional allocator-history analyses for comparisons and timelines."""

    stacks: bool = False
    events: bool = False
    lifetimes: bool = False
    on_missing: MissingPolicy = "warn"
    display: MemoryDisplayOptions = field(default_factory=MemoryDisplayOptions)

    def __post_init__(self) -> None:
        for name in ("stacks", "events", "lifetimes"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        _validate_missing_policy(self.on_missing)
        if not isinstance(self.display, MemoryDisplayOptions):
            raise TypeError("display must be MemoryDisplayOptions")

    def lifetime_options(self) -> "MemoryLifetimeOptions":
        """Return the lifetime policy embedded in this attribution request."""

        return MemoryLifetimeOptions(
            events=self.events,
            on_missing=self.on_missing,
            display=self.display,
        )


@dataclass(frozen=True)
class MemoryLifetimeOptions:
    """Allocator-history policy for allocation lifetime analysis."""

    events: bool = True
    on_missing: MissingPolicy = "warn"
    display: MemoryDisplayOptions = field(
        default_factory=lambda: MemoryDisplayOptions(stack_depth=4)
    )

    def __post_init__(self) -> None:
        if type(self.events) is not bool:
            raise TypeError("events must be a boolean")
        _validate_missing_policy(self.on_missing)
        if not isinstance(self.display, MemoryDisplayOptions):
            raise TypeError("display must be MemoryDisplayOptions")


def _validate_missing_policy(value: MissingPolicy) -> None:
    if not isinstance(value, str) or value not in {"warn", "error"}:
        raise ValueError("on_missing must be 'warn' or 'error'")
