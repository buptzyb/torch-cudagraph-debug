"""Shared normalized allocator stack identity and rendering helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

_STACK_FIELDS = (
    "filename",
    "line",
    "name",
    "fx_node_op",
    "fx_node_name",
    "fx_original_trace",
)


def normalize_stack_frames(
    frames: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return immutable-by-convention frame mappings with stable field types."""

    normalized = []
    for frame in frames:
        item: dict[str, Any] = {}
        for key in _STACK_FIELDS:
            value = frame.get(key)
            if value is None:
                continue
            if key == "line":
                if type(value) is not int:
                    raise TypeError("frame.line must be an integer")
                if value < 0:
                    raise ValueError("frame.line must be non-negative")
            elif not isinstance(value, str):
                raise TypeError(f"frame.{key} must be a string")
            item[key] = value
        normalized.append(item)
    return tuple(normalized)


def stack_key(
    frames: Sequence[Mapping[str, Any]],
    *,
    depth: int | None = None,
    fallback: str = "<unattributed>",
) -> str:
    """Format a normalized stack, optionally limiting rendered frames."""

    if depth is not None:
        validate_stack_depth(depth)
    if not frames:
        return fallback
    selected = frames if depth is None else frames[:depth]
    return " <- ".join(
        f"{frame.get('filename', '<unknown>')}:{frame.get('line', 0)}:"
        f"{frame.get('name', '<unknown>')}"
        for frame in selected
    )


def stack_fingerprint(frames: Sequence[Mapping[str, Any]]) -> str:
    """Return a stable fingerprint over every normalized frame field."""

    if not frames:
        return "unattributed"
    encoded = json.dumps(
        list(frames), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_stack_depth(value: int) -> None:
    if type(value) is not int:
        raise TypeError("stack_depth must be an integer")
    if value < 1:
        raise ValueError("stack_depth must be >= 1")
