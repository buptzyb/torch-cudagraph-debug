"""High-level tensor run collection and persistence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.utils.hooks import RemovableHandle

from torch_cudagraph_debug._provenance import (
    initialized_device_provenance,
    runtime_provenance,
)
from torch_cudagraph_debug._validation import (
    json_value,
    require_bool,
    require_exact_fields,
    require_finite_number,
    require_json_mapping,
    require_nonempty_string,
    require_nonnegative_int,
    require_optional_int,
    require_optional_nonempty_string,
    resolve_distributed_identity,
    strict_json_loads,
    validate_group_identity,
)
from torch_cudagraph_debug.types import (
    FrozenJSONValue,
    _freeze_json,
    _thaw_json,
)

from ._collector import (
    _EagerTensorCollector,
    _TensorCollector,
    synchronize_tensor_results,
    validate_synchronize_target,
)
from ._identity import TensorObservationKey, validate_observation_name
from .actions import NonContiguousPolicy, RecordAction, validate_non_contiguous_policy
from .errors import (
    TensorBundleError,
    TensorDebugError,
    TensorOwnershipError,
    TensorPayloadUnavailableError,
)

if TYPE_CHECKING:
    from .comparison import TensorComparisonOptions, TensorPointComparison

BUNDLE_SCHEMA = "torch-cudagraph-debug/tensor-run"
BUNDLE_FORMAT_VERSION = 1
ExecutionMode = Literal["eager", "cuda_graph"]
PayloadKind = Literal["full", "summary"]
SynchronizeTarget = bool | torch.cuda.Stream | torch.device
DeviceLike = torch.device | str | int

_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}
_NAME_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "run_id",
        "name",
        "execution",
        "rank",
        "group_id",
        "world_size",
        "created_at",
        "finished_at",
        "complete",
        "default_payload",
        "provenance",
        "run_metadata",
        "points",
    }
)
_POINT_FIELDS = frozenset(
    {"index", "label", "timestamp", "metadata", "replay_index", "observations"}
)
_OBSERVATION_FIELDS = frozenset(
    {
        "order",
        "name",
        "invocation_index",
        "shape",
        "stride",
        "dtype",
        "source_device",
        "nbytes",
        "payload",
        "sha256",
        "summary",
        "blob_file",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "numel",
        "finite_count",
        "nan_count",
        "pos_inf_count",
        "neg_inf_count",
        "zero_count",
        "minimum",
        "maximum",
        "mean",
        "std",
        "l2_norm",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUMMARY_CHUNK_ELEMENTS = 1_000_000


def validate_execution_mode(value: str) -> ExecutionMode:
    if not isinstance(value, str):
        raise TypeError("execution must be a string")
    if value not in {"eager", "cuda_graph"}:
        raise ValueError('execution must be either "eager" or "cuda_graph"')
    return value  # type: ignore[return-value]


def validate_payload_kind(value: str) -> PayloadKind:
    if not isinstance(value, str):
        raise TypeError("payload must be a string")
    if value not in {"full", "summary"}:
        raise ValueError('payload must be either "full" or "summary"')
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class TensorValueSummary:
    """Compact numerical summary stored for every observation."""

    numel: int
    finite_count: int
    nan_count: int
    pos_inf_count: int
    neg_inf_count: int
    zero_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    l2_norm: float | None

    def __post_init__(self) -> None:
        counts = (
            self.numel,
            self.finite_count,
            self.nan_count,
            self.pos_inf_count,
            self.neg_inf_count,
            self.zero_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("tensor summary counts must be non-negative integers")
        if sum(counts[1:5]) != self.numel:
            raise ValueError("tensor summary counts do not add up to numel")
        for name in ("minimum", "maximum", "mean", "std", "l2_norm"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"tensor summary {name} must be finite or None")

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "numel": self.numel,
            "finite_count": self.finite_count,
            "nan_count": self.nan_count,
            "pos_inf_count": self.pos_inf_count,
            "neg_inf_count": self.neg_inf_count,
            "zero_count": self.zero_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "l2_norm": self.l2_norm,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorValueSummary":
        require_exact_fields(
            value,
            _SUMMARY_FIELDS,
            "tensor summary",
            error_type=TensorBundleError,
        )
        integer_fields = (
            "numel",
            "finite_count",
            "nan_count",
            "pos_inf_count",
            "neg_inf_count",
            "zero_count",
        )
        integers = {
            name: require_nonnegative_int(
                value[name], f"tensor summary {name}", error_type=TensorBundleError
            )
            for name in integer_fields
        }

        optional: dict[str, float | None] = {}
        for name in ("minimum", "maximum", "mean", "std", "l2_norm"):
            raw = value[name]
            if raw is None:
                optional[name] = None
                continue
            optional[name] = require_finite_number(
                raw, f"tensor summary {name}", error_type=TensorBundleError
            )
        try:
            return cls(**integers, **optional)
        except ValueError as exc:
            raise TensorBundleError(str(exc)) from exc


@dataclass(frozen=True, eq=False)
class TensorObservation:
    """One named tensor value captured by a snapshot or point."""

    order: int
    name: str
    invocation_index: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    source_device: str
    nbytes: int
    payload: PayloadKind
    sha256: str
    summary: TensorValueSummary
    _blob_path: Path | None = field(default=None, repr=False, compare=False)
    _tensor_cache: dict[str, torch.Tensor] = field(
        default_factory=dict, repr=False, compare=False
    )
    _cache_tensors: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "stride", tuple(self.stride))
        if not isinstance(self.summary, TensorValueSummary):
            raise TypeError("tensor observation summary must be TensorValueSummary")
        if type(self.order) is not int or self.order < 0:
            raise ValueError("observation order must be a non-negative integer")
        validate_observation_name(self.name)
        if type(self.invocation_index) is not int or self.invocation_index < 0:
            raise ValueError("invocation_index must be a non-negative integer")
        if any(type(value) is not int or value < 0 for value in self.shape):
            raise ValueError("shape values must be non-negative integers")
        if any(type(value) is not int or value < 0 for value in self.stride):
            raise ValueError("stride values must be non-negative integers")
        if len(self.shape) != len(self.stride):
            raise ValueError("tensor stride rank must equal shape rank")
        if self.dtype not in _DTYPE_NAMES:
            raise ValueError(f"unsupported tensor dtype {self.dtype}")
        if not self.source_device:
            raise ValueError("source_device must be non-empty")
        if type(self.nbytes) is not int or self.nbytes < 0:
            raise ValueError("nbytes must be a non-negative integer")
        expected_nbytes = (
            math.prod(self.shape) * torch.empty((), dtype=self.dtype).element_size()
        )
        if self.nbytes != expected_nbytes:
            raise ValueError(f"nbytes={self.nbytes}; expected {expected_nbytes}")
        validate_payload_kind(self.payload)
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("sha256 must contain 64 lowercase hexadecimal digits")
        if self.summary.numel != math.prod(self.shape):
            raise ValueError("tensor summary numel does not match shape")

    @property
    def key(self) -> TensorObservationKey:
        return TensorObservationKey(self.name, self.invocation_index)

    @property
    def has_payload(self) -> bool:
        return self.payload == "full"

    def descriptor(self) -> dict[str, object]:
        return {
            "order": self.order,
            "name": self.name,
            "invocation_index": self.invocation_index,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": _dtype_name(self.dtype),
            "source_device": self.source_device,
            "nbytes": self.nbytes,
            "payload": self.payload,
            "sha256": self.sha256,
            "summary": self.summary.to_dict(),
        }

    def tensor(self) -> torch.Tensor:
        """Return an independent CPU tensor with validated payload content."""

        return self._materialize_tensor().clone()

    def _materialize_tensor(self) -> torch.Tensor:
        """Return the canonical private payload used by internal analysis."""

        if self.payload != "full":
            raise TensorPayloadUnavailableError(
                f"observation {self.name!r}[{self.invocation_index}] "
                "contains summary data only"
            )
        cached = self._tensor_cache.get("tensor")
        if cached is not None:
            return cached
        if self._blob_path is None:
            raise TensorBundleError(
                f"observation {self.name!r}[{self.invocation_index}] "
                "has no tensor payload"
            )
        try:
            raw = self._blob_path.read_bytes()
        except OSError as exc:
            raise TensorBundleError(
                f"could not read tensor blob {self._blob_path}: {exc}"
            ) from exc
        if len(raw) != self.nbytes:
            raise TensorBundleError(
                f"tensor blob {self._blob_path} has {len(raw)} bytes; "
                f"expected {self.nbytes}"
            )
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise TensorBundleError(
                f"tensor blob {self._blob_path} does not match its SHA-256"
            )
        tensor = _tensor_from_bytes(raw, self.dtype, self.shape)
        if self._cache_tensors:
            self._tensor_cache["tensor"] = tensor
        return tensor

    def _verify_payload(self) -> None:
        if self.payload == "full" and "tensor" not in self._tensor_cache:
            self._materialize_tensor()


def _validate_observation_sequence(
    observations: Sequence[TensorObservation],
    *,
    owner: str,
) -> None:
    expected_orders = list(range(len(observations)))
    if [item.order for item in observations] != expected_orders:
        raise ValueError(f"{owner} observations must be contiguous and ordered")

    invocation_counts: dict[str, int] = {}
    for item in observations:
        expected_invocation = invocation_counts.get(item.name, 0)
        if item.invocation_index != expected_invocation:
            raise ValueError(
                f"{owner} invocations must be contiguous independently per name"
            )
        invocation_counts[item.name] = expected_invocation + 1


@dataclass(frozen=True)
class TensorPoint:
    """One ordered set of named tensor observations."""

    run_id: str
    index: int
    label: str
    timestamp: float
    metadata: Mapping[str, FrozenJSONValue]
    replay_index: int | None
    observations: tuple[TensorObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_json(dict(self.metadata)))
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("point index must be a non-negative integer")
        if not self.label:
            raise ValueError("point label must be non-empty")
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not math.isfinite(float(self.timestamp))
        ):
            raise ValueError("point timestamp must be finite")
        if self.replay_index is not None and (
            type(self.replay_index) is not int or self.replay_index < 1
        ):
            raise ValueError("point replay_index must be a positive integer or None")
        _validate_observation_sequence(self.observations, owner="tensor point")

    @cached_property
    def by_key(self) -> Mapping[TensorObservationKey, TensorObservation]:
        return MappingProxyType({item.key: item for item in self.observations})

    def observation(
        self,
        name: str,
        invocation_index: int = 0,
    ) -> TensorObservation:
        try:
            return self.by_key[TensorObservationKey(name, invocation_index)]
        except KeyError as exc:
            raise KeyError(
                f"tensor observation {name!r}[{invocation_index}] "
                f"does not exist at point {self.label!r}"
            ) from exc

    def descriptor(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "index": self.index,
            "label": self.label,
            "timestamp": self.timestamp,
            "metadata": _thaw_json(self.metadata),
            "replay_index": self.replay_index,
            "observation_count": len(self.observations),
        }


@dataclass(frozen=True)
class TensorRun:
    """Immutable sequence of tensor observation points."""

    run_id: str
    name: str
    execution: ExecutionMode
    rank: int | None
    created_at: float
    finished_at: float | None
    complete: bool
    default_payload: PayloadKind
    points: tuple[TensorPoint, ...]
    group_id: str | None = None
    world_size: int | None = None
    provenance: Mapping[str, FrozenJSONValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    run_metadata: Mapping[str, FrozenJSONValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    bundle_dir: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _freeze_json(dict(self.provenance)))
        object.__setattr__(
            self,
            "run_metadata",
            _freeze_json(dict(self.run_metadata)),
        )
        if not self.run_id or not self.name:
            raise ValueError("run_id and name must be non-empty")
        validate_execution_mode(self.execution)
        validate_payload_kind(self.default_payload)
        validate_group_identity(self.rank, self.group_id, self.world_size)
        if type(self.complete) is not bool:
            raise TypeError("complete must be a boolean")
        if (
            isinstance(self.created_at, bool)
            or not isinstance(self.created_at, (int, float))
            or not math.isfinite(float(self.created_at))
        ):
            raise ValueError("created_at must be finite")
        if self.finished_at is not None and (
            isinstance(self.finished_at, bool)
            or not isinstance(self.finished_at, (int, float))
            or not math.isfinite(float(self.finished_at))
        ):
            raise ValueError("finished_at must be finite or None")
        labels: set[str] = set()
        last_replay = 0
        for expected_index, point in enumerate(self.points):
            if point.run_id != self.run_id or point.index != expected_index:
                raise ValueError("run points must be owned, contiguous, and ordered")
            if point.label in labels:
                raise ValueError(f"duplicate point label {point.label!r}")
            labels.add(point.label)
            if self.execution == "eager":
                if point.replay_index is not None:
                    raise ValueError("eager points must not have replay_index")
            else:
                if point.replay_index is None or point.replay_index <= last_replay:
                    raise ValueError(
                        "CUDA Graph replay indices must be positive and increasing"
                    )
                last_replay = point.replay_index

    def descriptor(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "execution": self.execution,
            "rank": self.rank,
            "group_id": self.group_id,
            "world_size": self.world_size,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "complete": self.complete,
            "default_payload": self.default_payload,
            "provenance": _thaw_json(self.provenance),
            "run_metadata": _thaw_json(self.run_metadata),
        }

    def point(self, ref: str | int | TensorPoint) -> TensorPoint:
        if isinstance(ref, TensorPoint):
            if ref.run_id != self.run_id:
                raise TensorOwnershipError(
                    f"point {ref.label!r} belongs to run {ref.run_id}, not {self.run_id}"
                )
            if ref.index < 0 or ref.index >= len(self.points):
                raise TensorOwnershipError(
                    f"point {ref.label!r} is not present in run {self.run_id}"
                )
            owned = self.points[ref.index]
            if owned.label != ref.label:
                raise TensorOwnershipError(
                    f"point {ref.label!r} is not present in run {self.run_id}"
                )
            return owned
        if isinstance(ref, int):
            try:
                return self.points[ref]
            except IndexError as exc:
                raise IndexError(f"tensor point index {ref} does not exist") from exc
        for point in self.points:
            if point.label == ref:
                return point
        raise KeyError(f"tensor point {ref!r} does not exist")

    def __getitem__(self, ref: str | int) -> TensorPoint:
        return self.point(ref)

    def compare(
        self,
        reference: str | int | TensorPoint,
        candidate: str | int | TensorPoint,
        *,
        options: TensorComparisonOptions | None = None,
    ) -> TensorPointComparison:
        from .comparison import compare_points

        return compare_points(
            self.point(reference),
            self.point(candidate),
            options=options,
        )

    @classmethod
    def load(
        cls,
        bundle_dir: str | Path,
        *,
        cache_tensors: bool = False,
    ) -> "TensorRun":
        root = Path(bundle_dir).resolve()
        manifest_path = root / "manifest.json"
        try:
            manifest_text = manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TensorBundleError(
                f"could not read tensor bundle {root}: {exc}"
            ) from exc
        manifest = strict_json_loads(
            manifest_text,
            error_type=TensorBundleError,
            context=f"tensor bundle {root}",
        )
        if not isinstance(manifest, Mapping):
            raise TensorBundleError("tensor bundle manifest must be a JSON object")
        if manifest.get("schema") != BUNDLE_SCHEMA:
            raise TensorBundleError(
                f"unsupported tensor bundle schema {manifest.get('schema')!r}"
            )
        require_exact_fields(
            manifest,
            _MANIFEST_FIELDS,
            "tensor bundle manifest",
            error_type=TensorBundleError,
        )
        if manifest.get("format_version") != BUNDLE_FORMAT_VERSION:
            raise TensorBundleError(
                "unsupported tensor bundle format_version "
                f"{manifest.get('format_version')!r}"
            )

        run_id = require_nonempty_string(
            manifest["run_id"], "run_id", error_type=TensorBundleError
        )
        name = require_nonempty_string(
            manifest["name"], "name", error_type=TensorBundleError
        )
        try:
            execution = validate_execution_mode(manifest["execution"])
            default_payload = validate_payload_kind(manifest["default_payload"])
        except (TypeError, ValueError) as exc:
            raise TensorBundleError(f"invalid tensor bundle mode: {exc}") from exc
        rank = require_optional_int(
            manifest["rank"], "rank", error_type=TensorBundleError
        )
        group_id = require_optional_nonempty_string(
            manifest["group_id"], "group_id", error_type=TensorBundleError
        )
        world_size = require_optional_int(
            manifest["world_size"], "world_size", error_type=TensorBundleError
        )
        try:
            validate_group_identity(rank, group_id, world_size)
        except (TypeError, ValueError) as exc:
            raise TensorBundleError(f"invalid tensor run identity: {exc}") from exc

        raw_points = manifest["points"]
        if not isinstance(raw_points, list):
            raise TensorBundleError("tensor bundle points must be a JSON list")
        points: list[TensorPoint] = []
        labels: set[str] = set()
        for expected_index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, Mapping):
                raise TensorBundleError(
                    f"tensor point {expected_index} must be a JSON object"
                )
            require_exact_fields(
                raw_point,
                _POINT_FIELDS,
                f"tensor point {expected_index}",
                error_type=TensorBundleError,
            )
            index = require_nonnegative_int(
                raw_point["index"],
                "tensor point index",
                error_type=TensorBundleError,
            )
            label = require_nonempty_string(
                raw_point["label"],
                "tensor point label",
                error_type=TensorBundleError,
            )
            if index != expected_index:
                raise TensorBundleError(
                    "tensor bundle point indices must be contiguous and ordered"
                )
            if label in labels:
                raise TensorBundleError(
                    f"tensor bundle has duplicate point label {label!r}"
                )
            labels.add(label)
            metadata = require_json_mapping(
                raw_point["metadata"],
                "point metadata",
                error_type=TensorBundleError,
            )
            replay_index = require_optional_int(
                raw_point["replay_index"],
                "tensor point replay_index",
                error_type=TensorBundleError,
            )
            if replay_index is not None and replay_index < 1:
                raise TensorBundleError("point replay_index must be positive")
            raw_observations = raw_point["observations"]
            if not isinstance(raw_observations, list):
                raise TensorBundleError(
                    f"observations for point {label!r} must be a JSON list"
                )
            observations: list[TensorObservation] = []
            keys: set[TensorObservationKey] = set()
            for expected_order, raw_observation in enumerate(raw_observations):
                observation = _observation_from_manifest(
                    root,
                    expected_order,
                    raw_observation,
                    cache_tensors=cache_tensors,
                )
                if observation.key in keys:
                    raise TensorBundleError(
                        f"point {label!r} has duplicate observation "
                        f"{observation.name!r}[{observation.invocation_index}]"
                    )
                keys.add(observation.key)
                observations.append(observation)
            try:
                point = TensorPoint(
                    run_id=run_id,
                    index=index,
                    label=label,
                    timestamp=require_finite_number(
                        raw_point["timestamp"],
                        "tensor point timestamp",
                        error_type=TensorBundleError,
                    ),
                    metadata=MappingProxyType(metadata),
                    observations=tuple(observations),
                    replay_index=replay_index,
                )
            except ValueError as exc:
                raise TensorBundleError(str(exc)) from exc
            points.append(point)

        provenance = require_json_mapping(
            manifest["provenance"], "provenance", error_type=TensorBundleError
        )
        run_metadata = require_json_mapping(
            manifest["run_metadata"], "run_metadata", error_type=TensorBundleError
        )
        finished_at = manifest["finished_at"]
        try:
            return cls(
                run_id=run_id,
                name=name,
                execution=execution,
                rank=rank,
                created_at=require_finite_number(
                    manifest["created_at"],
                    "created_at",
                    error_type=TensorBundleError,
                ),
                finished_at=(
                    require_finite_number(
                        finished_at,
                        "finished_at",
                        error_type=TensorBundleError,
                    )
                    if finished_at is not None
                    else None
                ),
                complete=require_bool(
                    manifest["complete"], "complete", error_type=TensorBundleError
                ),
                default_payload=default_payload,
                points=tuple(points),
                group_id=group_id,
                world_size=world_size,
                provenance=MappingProxyType(provenance),
                run_metadata=MappingProxyType(run_metadata),
                bundle_dir=root,
            )
        except (TypeError, ValueError) as exc:
            raise TensorBundleError(f"invalid tensor run: {exc}") from exc


@dataclass(frozen=True)
class _PendingObservation:
    order: int
    name: str
    invocation_index: int
    replay_index: int | None
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    source_device: str
    payload: PayloadKind
    tensor: torch.Tensor
    source_owner: torch.Tensor | None = None


@dataclass
class _ActivePoint:
    label: str
    metadata: Mapping[str, Any]
    synchronize: SynchronizeTarget
    pending: list[_PendingObservation] = field(default_factory=list)
    invocation_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _CaptureSlot:
    order: int
    name: str
    invocation_index: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    source_device: str
    payload: PayloadKind


class TensorRecorder:
    """Collect named eager or CUDA Graph tensor observations into a TensorRun."""

    def __init__(
        self,
        *,
        execution: ExecutionMode,
        name: str = "run",
        bundle_dir: str | Path | None = None,
        device: DeviceLike | None = None,
        payload: PayloadKind = "full",
        non_contiguous: NonContiguousPolicy = "error",
        strict_scope: bool = False,
        synchronize: SynchronizeTarget = True,
        rank: int | None = None,
        group_id: str | None = None,
        world_size: int | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        self.execution = validate_execution_mode(execution)
        self.name = name
        self.bundle_dir = Path(bundle_dir).resolve() if bundle_dir is not None else None
        self.default_payload = validate_payload_kind(payload)
        self.non_contiguous = validate_non_contiguous_policy(non_contiguous)
        if type(strict_scope) is not bool:
            raise TypeError("strict_scope must be a boolean")
        self.strict_scope = strict_scope
        validate_synchronize_target(synchronize)
        self.synchronize = synchronize
        self.rank, self.world_size = resolve_distributed_identity(rank, world_size)
        self.group_id = group_id
        validate_group_identity(self.rank, self.group_id, self.world_size)
        metadata = json_value(
            dict(run_metadata or {}),
            "$.run_metadata",
            error_type=TensorBundleError,
        )
        provenance = json_value(
            runtime_provenance(), "$.provenance", error_type=TensorBundleError
        )
        assert isinstance(metadata, dict)
        assert isinstance(provenance, dict)
        self.run_metadata = _freeze_json(metadata)
        self._provenance: dict[str, Any] = provenance
        self._run_id = uuid.uuid4().hex
        self._created_at = time.time()
        self._finished_at: float | None = None
        self._points: list[TensorPoint] = []
        self._result: TensorRun | None = None
        self._closed = False
        self._context_active = False
        self._active_point: _ActivePoint | None = None
        self._capture_slots: list[_CaptureSlot] = []
        self._capture_invocation_counts: dict[str, int] = {}
        self._device: torch.device | None = None
        self._collector: _TensorCollector | None = None
        self._eager_collector: _EagerTensorCollector | None = (
            _EagerTensorCollector() if self.execution == "eager" else None
        )
        self._last_point_replay_index: int | None = None

        if self.execution == "cuda_graph":
            self._device = _resolve_cuda_device(device)
            self._collector = _TensorCollector(
                f"{self.name}.__tensor_run__",
                [RecordAction()],
                non_contiguous=self.non_contiguous,
                when="capture",
                device=self._device,
            )
        elif device is not None:
            self._device = _resolve_cuda_device(device)

        if self.bundle_dir is not None:
            if self.bundle_dir.exists() and any(self.bundle_dir.iterdir()):
                raise FileExistsError(
                    f"tensor bundle directory is not empty: {self.bundle_dir}"
                )
            (self.bundle_dir / "blobs").mkdir(parents=True, exist_ok=True)
            self._write_manifest(complete=False)

    def observe(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        payload: PayloadKind | None = None,
    ) -> torch.Tensor:
        """Return tensor unchanged while recording one named observation."""

        self._ensure_open()
        name = validate_observation_name(name)
        selected_payload = (
            self.default_payload if payload is None else validate_payload_kind(payload)
        )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("observe tensor must be a torch.Tensor")

        if self.execution == "eager":
            if self._active_point is None:
                if self.strict_scope:
                    raise TensorDebugError(
                        "eager observe() requires an active record_point() context"
                    )
                return tensor
            source = self._validate_tensor(tensor)
            invocation_index = self._next_eager_invocation(name)
            assert self._eager_collector is not None
            staging = self._eager_collector.collect(source)
            self._active_point.pending.append(
                _PendingObservation(
                    order=len(self._active_point.pending),
                    name=name,
                    invocation_index=invocation_index,
                    replay_index=None,
                    shape=tuple(tensor.shape),
                    stride=tuple(tensor.stride()),
                    dtype=tensor.dtype,
                    source_device=str(tensor.device),
                    payload=selected_payload,
                    tensor=staging,
                    source_owner=source if source is not tensor else None,
                )
            )
            return tensor

        with torch.cuda.device(tensor.device):
            is_capturing = torch.cuda.is_current_stream_capturing()
        if not is_capturing:
            if self.strict_scope:
                raise TensorDebugError(
                    "cuda_graph observe() requires an active CUDA graph capture"
                )
            return tensor
        self._validate_tensor(tensor)
        assert self._collector is not None
        invocation_index = self._capture_invocation_counts.get(name, 0)
        self._capture_invocation_counts[name] = invocation_index + 1
        result = self._collector.enqueue(
            tensor,
            name=name,
            invocation_index=invocation_index,
        )
        self._capture_slots.append(
            _CaptureSlot(
                order=len(self._capture_slots),
                name=name,
                invocation_index=invocation_index,
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                dtype=tensor.dtype,
                source_device=str(tensor.device),
                payload=selected_payload,
            )
        )
        return result

    def watch_grad(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        payload: PayloadKind | None = None,
        strict: bool = False,
    ) -> RemovableHandle | None:
        """Register an autograd hook that records a named gradient observation."""

        self._ensure_open()
        name = validate_observation_name(name)
        if not tensor.requires_grad:
            if strict:
                raise RuntimeError(
                    "cannot watch gradients for a tensor that does not require grad"
                )
            return None

        def hook(grad: torch.Tensor | None) -> torch.Tensor | None:
            if grad is None:
                return None
            self.observe(grad, name=name, payload=payload)
            return grad

        return tensor.register_hook(hook)

    @contextmanager
    def record_point(
        self,
        label: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        synchronize: SynchronizeTarget | None = None,
    ) -> Iterator[None]:
        """Record one eager execution or the latest values from one graph replay."""

        self._ensure_collecting()
        if not label:
            raise ValueError("tensor point label must be non-empty")
        if any(point.label == label for point in self._points):
            raise ValueError(f"tensor point label {label!r} already exists")
        if self._active_point is not None:
            raise TensorDebugError("tensor point contexts cannot be nested")
        if self._device is not None:
            with torch.cuda.device(self._device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "TensorRecorder.record_point() cannot run during CUDA graph "
                        "capture; capture observe() calls first, then wrap graph.replay()"
                    )
        target = self.synchronize if synchronize is None else synchronize
        validate_synchronize_target(target)
        serializable_metadata = json_value(
            dict(metadata or {}),
            "$.point.metadata",
            error_type=TensorBundleError,
        )
        assert isinstance(serializable_metadata, dict)
        active = _ActivePoint(
            label=label,
            metadata=MappingProxyType(serializable_metadata),
            synchronize=target,
        )
        self._active_point = active
        try:
            yield
        except BaseException:
            self._abort_point(active)
            raise
        else:
            self._finish_point(active)
        finally:
            if self._active_point is active:
                self._active_point = None

    def preview(self) -> TensorRun:
        """Return an immutable nonterminal view of collected points."""

        if self._result is not None:
            return self._result
        return self._build_run(complete=False)

    def finish(self) -> TensorRun:
        if self._context_active:
            raise TensorDebugError(
                "cannot finish a tensor recorder inside its context manager"
            )
        return self._finish()

    def _finish(self) -> TensorRun:
        if self._result is not None:
            return self._result
        if self._active_point is not None:
            raise TensorDebugError("cannot finish while a tensor point is active")
        finished_at = time.time()
        candidate = self._build_run(complete=True, finished_at=finished_at)
        self._write_manifest(complete=True, finished_at=finished_at)
        self._finished_at = finished_at
        self._result = candidate
        return candidate

    def _abort(self) -> TensorRun:
        if self._result is not None:
            return self._result
        if self._active_point is not None:
            self._abort_point(self._active_point)
        finished_at = time.time()
        candidate = self._build_run(complete=False, finished_at=finished_at)
        self._write_manifest(complete=False, finished_at=finished_at)
        self._finished_at = finished_at
        self._result = candidate
        return candidate

    @property
    def result(self) -> TensorRun:
        if self._result is None:
            raise TensorDebugError("tensor recorder has not been finished")
        return self._result

    def close(
        self,
        *,
        synchronize: SynchronizeTarget | None = None,
    ) -> None:
        """Release native resources after the captured graph can no longer replay."""

        if self._closed:
            return
        selected = self.synchronize if synchronize is None else synchronize
        validate_synchronize_target(selected)
        if self._active_point is not None:
            raise TensorDebugError("cannot close while a tensor point is active")
        if self._collector is not None:
            self._collector.close(synchronize=selected)
            self._collector = None
        self._closed = True

    def __enter__(self) -> "TensorRecorder":
        self._ensure_open()
        if self._context_active:
            raise TensorDebugError("tensor recorder context cannot be re-entered")
        self._context_active = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._context_active:
            raise TensorDebugError("tensor recorder context is not active")
        try:
            if exc_type is None:
                self._finish()
            else:
                try:
                    self._abort()
                except Exception as cleanup_error:
                    warnings.warn(
                        "could not persist aborted tensor run: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        finally:
            try:
                self.close(synchronize=self.synchronize)
            except Exception as cleanup_error:
                if exc_type is None:
                    raise
                warnings.warn(
                    "could not close tensor recorder after application error: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            finally:
                self._context_active = False

    def _next_eager_invocation(self, name: str) -> int:
        assert self._active_point is not None
        index = self._active_point.invocation_counts.get(name, 0)
        self._active_point.invocation_counts[name] = index + 1
        return index

    def _validate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cuda":
            raise ValueError("TensorRecorder observations must be CUDA tensors")
        device = torch.device("cuda", tensor.device.index)
        if self._device is None:
            self._device = device
        if device != self._device:
            raise ValueError(
                f"observation device {device} does not match recorder device "
                f"{self._device}"
            )
        if tensor.dtype not in _DTYPE_NAMES:
            raise ValueError(f"unsupported tensor dtype {tensor.dtype}")
        if tensor.is_contiguous():
            return tensor
        if self.non_contiguous == "error":
            raise ValueError(
                "TensorRecorder observations must be contiguous; pass "
                'non_contiguous="copy" to allow a debug-only CUDA copy'
            )
        return tensor.contiguous()

    def _finish_point(self, active: _ActivePoint) -> None:
        if self._active_point is not active:
            raise TensorDebugError("tensor point ownership changed unexpectedly")
        if self.execution == "cuda_graph":
            active.pending.extend(self._collect_cuda_graph_observations(active))
        elif active.pending and self._device is not None:
            assert self._eager_collector is not None
            self._eager_collector.synchronize(self._device, active.synchronize)
        self._commit_point(active.label, active.metadata, active.pending)
        self._active_point = None

    def _abort_point(self, active: _ActivePoint) -> None:
        if self.execution == "eager" and active.pending and self._device is not None:
            try:
                assert self._eager_collector is not None
                self._eager_collector.synchronize(self._device, active.synchronize)
            except Exception:
                pass
        active.pending.clear()
        self._active_point = None

    def _collect_cuda_graph_observations(
        self,
        active: _ActivePoint,
    ) -> list[_PendingObservation]:
        if self._collector is None or self._device is None:
            raise TensorDebugError("CUDA Graph tensor recorder has no native session")
        if not self._capture_slots:
            raise TensorDebugError(
                "CUDA Graph tensor recorder captured no observations"
            )
        synchronize_tensor_results(self._device, active.synchronize)
        snapshots = self._collector.collect(synchronize=False)
        if len(snapshots) != len(self._capture_slots):
            raise TensorDebugError(
                "captured tensor slot count does not match collected snapshots: "
                f"{len(self._capture_slots)} slots, {len(snapshots)} snapshots"
            )
        replay_indices = {snapshot.replay_index for snapshot in snapshots}
        if len(replay_indices) != 1:
            raise TensorDebugError(
                "captured tensor observations reported different replay indices"
            )
        replay_index = next(iter(replay_indices))
        if replay_index < 1:
            raise TensorDebugError(
                "TensorRecorder.record_point() did not observe a CUDA Graph replay"
            )
        if (
            self._last_point_replay_index is not None
            and replay_index <= self._last_point_replay_index
        ):
            raise TensorDebugError(
                "TensorRecorder.record_point() did not observe a new CUDA Graph replay"
            )
        pending: list[_PendingObservation] = []
        for slot, snapshot in zip(self._capture_slots, snapshots):
            if snapshot.order != slot.order:
                raise TensorDebugError(
                    "captured tensor slot ordering changed unexpectedly"
                )
            if (
                snapshot.name != slot.name
                or snapshot.invocation_index != slot.invocation_index
            ):
                raise TensorDebugError(
                    "captured tensor observation identity changed unexpectedly"
                )
            if (
                tuple(snapshot.shape or ()) != slot.shape
                or snapshot.dtype != slot.dtype
            ):
                raise TensorDebugError(
                    f"captured tensor metadata changed for {slot.name!r}"
                )
            pending.append(
                _PendingObservation(
                    order=slot.order,
                    name=slot.name,
                    invocation_index=slot.invocation_index,
                    replay_index=snapshot.replay_index,
                    shape=slot.shape,
                    stride=slot.stride,
                    dtype=slot.dtype,
                    source_device=slot.source_device,
                    payload=slot.payload,
                    tensor=snapshot.tensor,
                )
            )
        self._last_point_replay_index = replay_index
        return pending

    def _commit_point(
        self,
        label: str,
        metadata: Mapping[str, Any],
        pending: Sequence[_PendingObservation],
    ) -> TensorPoint:
        index = len(self._points)
        observations: list[TensorObservation] = []
        seen: set[TensorObservationKey] = set()
        replay_indices = {
            item.replay_index for item in pending if item.replay_index is not None
        }
        if len(replay_indices) > 1:
            raise TensorDebugError("tensor point contains multiple replay indices")
        replay_index = next(iter(replay_indices), None)
        for expected_order, item in enumerate(pending):
            if item.order != expected_order:
                raise TensorDebugError("tensor observation order must be contiguous")
            key = TensorObservationKey(item.name, item.invocation_index)
            if key in seen:
                raise TensorDebugError(
                    f"duplicate tensor observation {item.name!r}"
                    f"[{item.invocation_index}]"
                )
            seen.add(key)
            tensor = item.tensor.detach().contiguous().cpu()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest = hashlib.sha256(raw).hexdigest()
            blob_path: Path | None = None
            cache: dict[str, torch.Tensor] = {}
            if item.payload == "full":
                if self.bundle_dir is None:
                    cache["tensor"] = tensor
                else:
                    blob_path = self._write_blob(digest, raw)
            observations.append(
                TensorObservation(
                    order=expected_order,
                    name=item.name,
                    invocation_index=item.invocation_index,
                    shape=item.shape,
                    stride=item.stride,
                    dtype=item.dtype,
                    source_device=item.source_device,
                    nbytes=len(raw),
                    payload=item.payload,
                    sha256=digest,
                    summary=_summarize_tensor(tensor),
                    _blob_path=blob_path,
                    _tensor_cache=cache,
                    _cache_tensors=self.bundle_dir is None,
                )
            )
        point = TensorPoint(
            run_id=self._run_id,
            index=index,
            label=label,
            timestamp=time.time(),
            metadata=metadata,
            observations=tuple(observations),
            replay_index=replay_index,
        )
        self._record_device_provenance()
        self._write_manifest(complete=False, points=(*self._points, point))
        self._points.append(point)
        return point

    def _write_blob(self, digest: str, raw: bytes) -> Path:
        assert self.bundle_dir is not None
        path = self.bundle_dir / "blobs" / f"{digest}.bin"
        if path.exists():
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise TensorBundleError(
                    f"could not verify existing tensor blob {path}: {exc}"
                ) from exc
            if (
                len(existing) != len(raw)
                or hashlib.sha256(existing).hexdigest() != digest
            ):
                raise TensorBundleError(
                    f"existing tensor blob {path} does not match its content address"
                )
            return path
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(raw)
            temporary.replace(path)
        except OSError as exc:
            raise TensorBundleError(
                f"could not persist tensor blob {path}: {exc}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def _write_manifest(
        self,
        *,
        complete: bool,
        points: Sequence[TensorPoint] | None = None,
        finished_at: float | None = None,
    ) -> None:
        if self.bundle_dir is None:
            return
        payload = {
            "schema": BUNDLE_SCHEMA,
            "format_version": BUNDLE_FORMAT_VERSION,
            "run_id": self._run_id,
            "name": self.name,
            "execution": self.execution,
            "rank": self.rank,
            "group_id": self.group_id,
            "world_size": self.world_size,
            "created_at": self._created_at,
            "finished_at": self._finished_at if finished_at is None else finished_at,
            "complete": complete,
            "default_payload": self.default_payload,
            "provenance": self._provenance,
            "run_metadata": _thaw_json(self.run_metadata),
            "points": [
                _point_manifest(point, self.bundle_dir)
                for point in (self._points if points is None else points)
            ],
        }
        path = self.bundle_dir / "manifest.json"
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            raise TensorBundleError(
                f"could not persist tensor bundle manifest: {exc}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)

    def _record_device_provenance(self) -> None:
        if "device" in self._provenance or self._device is None:
            return
        device = initialized_device_provenance(self._device)
        if device is not None:
            self._provenance["device"] = device

    def _build_run(
        self, *, complete: bool, finished_at: float | None = None
    ) -> TensorRun:
        return TensorRun(
            run_id=self._run_id,
            name=self.name,
            execution=self.execution,
            rank=self.rank,
            created_at=self._created_at,
            finished_at=self._finished_at if finished_at is None else finished_at,
            complete=complete,
            default_payload=self.default_payload,
            points=tuple(self._points),
            group_id=self.group_id,
            world_size=self.world_size,
            provenance=_freeze_json(dict(self._provenance)),
            run_metadata=self.run_metadata,
            bundle_dir=self.bundle_dir,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise TensorDebugError("tensor recorder is closed")

    def _ensure_collecting(self) -> None:
        self._ensure_open()
        if self._result is not None:
            raise TensorDebugError("cannot add points to a finished tensor recorder")


def _point_manifest(point: TensorPoint, root: Path) -> dict[str, object]:
    observations = []
    for item in point.observations:
        blob_file: str | None = None
        if item._blob_path is not None:
            blob_file = item._blob_path.relative_to(root).as_posix()
        observations.append(
            {
                "order": item.order,
                "name": item.name,
                "invocation_index": item.invocation_index,
                "shape": list(item.shape),
                "stride": list(item.stride),
                "dtype": _dtype_name(item.dtype),
                "source_device": item.source_device,
                "nbytes": item.nbytes,
                "payload": item.payload,
                "sha256": item.sha256,
                "summary": item.summary.to_dict(),
                "blob_file": blob_file,
            }
        )
    return {
        "index": point.index,
        "label": point.label,
        "timestamp": point.timestamp,
        "metadata": _thaw_json(point.metadata),
        "replay_index": point.replay_index,
        "observations": observations,
    }


def _observation_from_manifest(
    root: Path,
    expected_order: int,
    raw: Any,
    *,
    cache_tensors: bool,
) -> TensorObservation:
    if not isinstance(raw, Mapping):
        raise TensorBundleError(
            f"tensor observation {expected_order} must be a JSON object"
        )
    require_exact_fields(
        raw,
        _OBSERVATION_FIELDS,
        f"tensor observation {expected_order}",
        error_type=TensorBundleError,
    )
    order = require_nonnegative_int(
        raw["order"], "tensor observation order", error_type=TensorBundleError
    )
    if order != expected_order:
        raise TensorBundleError(
            "tensor observation order must be contiguous and ordered"
        )
    name = require_nonempty_string(raw["name"], "name", error_type=TensorBundleError)
    invocation_index = require_nonnegative_int(
        raw["invocation_index"],
        "invocation_index",
        error_type=TensorBundleError,
    )
    shape = _strict_integer_tuple(raw["shape"], "shape")
    stride = _strict_integer_tuple(raw["stride"], "stride")
    if len(stride) != len(shape):
        raise TensorBundleError("tensor stride rank must equal shape rank")
    dtype_name = require_nonempty_string(
        raw["dtype"], "dtype", error_type=TensorBundleError
    )
    try:
        dtype = _NAME_DTYPES[dtype_name]
    except KeyError as exc:
        raise TensorBundleError(f"unsupported tensor dtype {dtype_name!r}") from exc
    source_device = require_nonempty_string(
        raw["source_device"], "source_device", error_type=TensorBundleError
    )
    nbytes = require_nonnegative_int(
        raw["nbytes"], "nbytes", error_type=TensorBundleError
    )
    expected_nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if nbytes != expected_nbytes:
        raise TensorBundleError(
            f"tensor observation {name!r} has nbytes={nbytes}; "
            f"expected {expected_nbytes}"
        )
    try:
        payload = validate_payload_kind(raw["payload"])
    except (TypeError, ValueError) as exc:
        raise TensorBundleError(f"invalid observation payload: {exc}") from exc
    digest = require_nonempty_string(
        raw["sha256"], "sha256", error_type=TensorBundleError
    )
    if not _SHA256_RE.fullmatch(digest):
        raise TensorBundleError("tensor observation has an invalid SHA-256")
    summary_raw = raw["summary"]
    if not isinstance(summary_raw, Mapping):
        raise TensorBundleError("tensor observation summary must be a JSON object")
    summary = TensorValueSummary.from_dict(summary_raw)
    if summary.numel != math.prod(shape):
        raise TensorBundleError("tensor summary numel does not match shape")

    blob_path: Path | None = None
    if payload == "full":
        blob_path = _safe_blob_path(root, raw["blob_file"], digest)
        if not blob_path.is_file():
            raise TensorBundleError(
                f"tensor blob for {name!r}[{invocation_index}] does not exist"
            )
        if blob_path.stat().st_size != nbytes:
            raise TensorBundleError(
                f"tensor blob for {name!r}[{invocation_index}] has an unexpected size"
            )
    elif raw["blob_file"] is not None:
        raise TensorBundleError("summary-only observations must not reference a blob")

    try:
        return TensorObservation(
            order=order,
            name=name,
            invocation_index=invocation_index,
            shape=shape,
            stride=stride,
            dtype=dtype,
            source_device=source_device,
            nbytes=nbytes,
            payload=payload,
            sha256=digest,
            summary=summary,
            _blob_path=blob_path,
            _cache_tensors=cache_tensors,
        )
    except (TypeError, ValueError) as exc:
        raise TensorBundleError(
            f"invalid tensor observation {expected_order}: {exc}"
        ) from exc


def _summarize_tensor(tensor: torch.Tensor) -> TensorValueSummary:
    flat = tensor.detach().contiguous().view(-1)
    numel = flat.numel()
    finite_count = 0
    nan_count = 0
    pos_inf_count = 0
    neg_inf_count = 0
    zero_count = 0
    minimum: float | None = None
    maximum: float | None = None
    total = 0.0
    total_squares = 0.0

    for start in range(0, numel, _SUMMARY_CHUNK_ELEMENTS):
        chunk = flat[start : start + _SUMMARY_CHUNK_ELEMENTS]
        zero_count += int((chunk == 0).sum().item())
        if chunk.is_floating_point():
            nan_count += int(torch.isnan(chunk).sum().item())
            pos_inf_count += int(torch.isposinf(chunk).sum().item())
            neg_inf_count += int(torch.isneginf(chunk).sum().item())
            finite_mask = torch.isfinite(chunk)
            finite = chunk[finite_mask]
        else:
            finite = chunk
        finite_count += finite.numel()
        if finite.numel() == 0:
            continue
        values = finite.to(torch.float64)
        chunk_min = float(values.min().item())
        chunk_max = float(values.max().item())
        minimum = chunk_min if minimum is None else min(minimum, chunk_min)
        maximum = chunk_max if maximum is None else max(maximum, chunk_max)
        total += float(values.sum().item())
        total_squares += float((values * values).sum().item())

    if finite_count:
        mean = total / finite_count if math.isfinite(total) else None
        if mean is not None and math.isfinite(total_squares):
            variance = max(total_squares / finite_count - mean * mean, 0.0)
            std = math.sqrt(variance) if math.isfinite(variance) else None
        else:
            std = None
        l2_norm = (
            math.sqrt(max(total_squares, 0.0)) if math.isfinite(total_squares) else None
        )
    else:
        mean = None
        std = None
        l2_norm = None
    return TensorValueSummary(
        numel=numel,
        finite_count=finite_count,
        nan_count=nan_count,
        pos_inf_count=pos_inf_count,
        neg_inf_count=neg_inf_count,
        zero_count=zero_count,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        l2_norm=l2_norm,
    )


def _tensor_from_bytes(
    raw: bytes,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not raw:
        return torch.empty(shape, dtype=dtype)
    byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return byte_tensor.view(dtype).reshape(shape).clone()


def _dtype_name(dtype: torch.dtype) -> str:
    try:
        return _DTYPE_NAMES[dtype]
    except KeyError as exc:
        raise TensorBundleError(f"unsupported tensor dtype {dtype}") from exc


def _resolve_cuda_device(device: DeviceLike | None) -> torch.device:
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    if isinstance(device, int) and not isinstance(device, bool):
        return torch.device("cuda", device)
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("device must identify a CUDA device")
    if resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _safe_blob_path(root: Path, raw: Any, digest: str) -> Path:
    expected = f"blobs/{digest}.bin"
    if raw != expected:
        raise TensorBundleError(
            f"tensor blob path must be content-addressed as {expected!r}"
        )
    resolved = (root / expected).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TensorBundleError("tensor blob path escapes the bundle") from exc
    return resolved


def _strict_integer_tuple(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TensorBundleError(f"tensor {context} must be a JSON list")
    return tuple(
        require_nonnegative_int(
            item, f"tensor {context} value", error_type=TensorBundleError
        )
        for item in value
    )
