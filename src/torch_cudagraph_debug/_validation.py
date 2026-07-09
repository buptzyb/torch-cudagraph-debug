"""Strict validation helpers shared by persisted debug domains."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, cast

from .types import JSONValue

ErrorT = TypeVar("ErrorT", bound=Exception)


def strict_json_loads(text: str, *, error_type: type[ErrorT], context: str) -> Any:
    """Decode strict JSON, rejecting non-standard numeric constants and
    duplicate object keys."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise error_type(f"could not decode {context}: {exc}") from exc


def require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    context: str,
    *,
    error_type: type[ErrorT],
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if not missing and not unexpected:
        return
    details = []
    if missing:
        details.append(f"missing {missing!r}")
    if unexpected:
        details.append(f"unexpected {unexpected!r}")
    raise error_type(f"{context} has invalid fields: {', '.join(details)}")


def require_bool(value: Any, context: str, *, error_type: type[ErrorT]) -> bool:
    if type(value) is not bool:
        raise error_type(f"{context} must be a boolean")
    return value


def require_int(value: Any, context: str, *, error_type: type[ErrorT]) -> int:
    if type(value) is not int:
        raise error_type(f"{context} must be an integer")
    return value


def require_nonnegative_int(
    value: Any, context: str, *, error_type: type[ErrorT]
) -> int:
    result = require_int(value, context, error_type=error_type)
    if result < 0:
        raise error_type(f"{context} must be non-negative")
    return result


def require_finite_number(
    value: Any, context: str, *, error_type: type[ErrorT]
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{context} must be a finite JSON number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise error_type(f"{context} must be finite") from exc
    if not math.isfinite(result):
        raise error_type(f"{context} must be finite")
    return result


def require_nonempty_string(
    value: Any, context: str, *, error_type: type[ErrorT]
) -> str:
    if not isinstance(value, str) or not value:
        raise error_type(f"{context} must be a non-empty string")
    return value


def require_optional_int(
    value: Any, context: str, *, error_type: type[ErrorT]
) -> int | None:
    if value is None:
        return None
    return require_int(value, context, error_type=error_type)


def require_optional_nonempty_string(
    value: Any, context: str, *, error_type: type[ErrorT]
) -> str | None:
    if value is None:
        return None
    return require_nonempty_string(value, context, error_type=error_type)


def json_value(value: Any, path: str, *, error_type: type[ErrorT]) -> JSONValue:
    """Return a JSON-compatible copy while preserving exact scalar types."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error_type(f"{path} contains a non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [
            json_value(item, f"{path}[{index}]", error_type=error_type)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise error_type(f"{path} contains non-string mapping key {key!r}")
            output[key] = json_value(item, f"{path}.{key}", error_type=error_type)
        return output
    raise error_type(f"{path} contains unsupported value {type(value).__name__}")


def require_json_mapping(
    value: Any, context: str, *, error_type: type[ErrorT]
) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise error_type(f"{context} must be a JSON object")
    result = json_value(dict(value), f"$.{context}", error_type=error_type)
    assert isinstance(result, dict)
    return cast(dict[str, JSONValue], result)


def resolve_distributed_identity(
    rank: int | None,
    world_size: int | None,
) -> tuple[int | None, int | None]:
    """Resolve explicit, environment, or initialized torch.distributed identity."""

    resolved_rank = _resolve_distributed_value(rank, "RANK", "get_rank")
    resolved_world_size = _resolve_distributed_value(
        world_size, "WORLD_SIZE", "get_world_size"
    )
    validate_group_identity(resolved_rank, None, resolved_world_size)
    return resolved_rank, resolved_world_size


def validate_group_identity(
    rank: int | None,
    group_id: str | None,
    world_size: int | None,
) -> None:
    if rank is not None and type(rank) is not int:
        raise TypeError("rank must be an integer or None")
    if world_size is not None and type(world_size) is not int:
        raise TypeError("world_size must be an integer or None")
    if group_id is not None and (not isinstance(group_id, str) or not group_id):
        raise ValueError("group_id must be a non-empty string when provided")
    if rank is not None and rank < 0:
        raise ValueError("rank must be non-negative")
    if world_size is not None and world_size < 1:
        raise ValueError("world_size must be >= 1")
    if rank is not None and world_size is not None and rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")


def comparable_provenance(value: Mapping[str, Any]) -> dict[str, object]:
    """Return rank-comparable runtime fields, excluding host and UUID identity."""

    result: dict[str, object] = {}
    producer = value.get("producer")
    if isinstance(producer, Mapping):
        result["producer"] = dict(producer)
    runtime = value.get("runtime")
    if isinstance(runtime, Mapping):
        result["runtime"] = {
            key: runtime.get(key)
            for key in ("python", "platform", "torch", "cuda")
            if key in runtime
        }
    device = value.get("device")
    if isinstance(device, Mapping):
        result["device"] = _comparable_device(device)
    devices = value.get("devices")
    if isinstance(devices, Sequence) and not isinstance(
        devices, (str, bytes, bytearray)
    ):
        result["devices"] = [
            _comparable_device(item) for item in devices if isinstance(item, Mapping)
        ]
    return result


def json_signature(value: object) -> str:
    """Return a canonical JSON signature for recursively frozen metadata."""

    return json.dumps(_plain_json(value), sort_keys=True, separators=(",", ":"))


def _comparable_device(value: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: value.get(key)
        for key in ("name", "capability", "total_memory_bytes")
        if key in value
    }


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _resolve_distributed_value(
    explicit: int | None,
    environment_name: str,
    distributed_getter: str,
) -> int | None:
    if explicit is not None:
        if type(explicit) is not int:
            raise TypeError(f"{environment_name.lower()} must be an integer or None")
        return explicit

    environment_value = os.environ.get(environment_name)
    if environment_value is not None:
        try:
            return int(environment_value)
        except ValueError as exc:
            raise ValueError(
                f"environment variable {environment_name} must be an integer; "
                f"got {environment_value!r}"
            ) from exc

    try:
        import torch.distributed as distributed
    except ImportError:
        return None
    if not distributed.is_available() or not distributed.is_initialized():
        return None
    return int(getattr(distributed, distributed_getter)())
