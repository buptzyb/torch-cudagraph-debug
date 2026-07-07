"""Shared public types for JSON-compatible persisted metadata."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeAlias

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
FrozenJSONValue: TypeAlias = (
    JSONScalar | tuple["FrozenJSONValue", ...] | Mapping[str, "FrozenJSONValue"]
)


def _freeze_json(value: JSONValue | FrozenJSONValue) -> FrozenJSONValue:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: FrozenJSONValue) -> JSONValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = ["FrozenJSONValue", "JSONScalar", "JSONValue"]
