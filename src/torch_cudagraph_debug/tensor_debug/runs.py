"""High-level tensor run collection and persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch.utils.hooks import RemovableHandle

from torch_cudagraph_debug._provenance import (
    initialized_device_provenance,
    runtime_provenance,
)

from .actions import NonContiguousPolicy, RecordTensor, validate_non_contiguous_policy
from .errors import (
    TensorBundleError,
    TensorDebugError,
    TensorOwnershipError,
    TensorPayloadUnavailableError,
)
from .probe import TensorProbe

BUNDLE_SCHEMA = "torch-cudagraph-debug/tensor-run"
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
_POINT_FIELDS = frozenset({"index", "label", "timestamp", "metadata", "observations"})
_OBSERVATION_FIELDS = frozenset(
    {
        "order",
        "probe_name",
        "invocation_index",
        "replay_index",
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
    if value not in {"eager", "cuda_graph"}:
        raise ValueError('execution must be either "eager" or "cuda_graph"')
    return value  # type: ignore[return-value]


def validate_payload_kind(value: str) -> PayloadKind:
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
        _require_exact_fields(value, _SUMMARY_FIELDS, "tensor summary")
        integer_fields = (
            "numel",
            "finite_count",
            "nan_count",
            "pos_inf_count",
            "neg_inf_count",
            "zero_count",
        )
        integers = {name: int(value[name]) for name in integer_fields}
        if any(item < 0 for item in integers.values()):
            raise TensorBundleError("tensor summary counts must be non-negative")
        if (
            integers["finite_count"]
            + integers["nan_count"]
            + integers["pos_inf_count"]
            + integers["neg_inf_count"]
            != integers["numel"]
        ):
            raise TensorBundleError("tensor summary counts do not add up to numel")

        optional: dict[str, float | None] = {}
        for name in ("minimum", "maximum", "mean", "std", "l2_norm"):
            raw = value[name]
            if raw is None:
                optional[name] = None
                continue
            number = float(raw)
            if not math.isfinite(number):
                raise TensorBundleError(f"tensor summary {name} must be finite")
            optional[name] = number
        return cls(**integers, **optional)


@dataclass(frozen=True, eq=False)
class TensorObservation:
    """One named tensor value owned by a TensorPoint."""

    run_id: str
    point_index: int
    order: int
    probe_name: str
    invocation_index: int
    replay_index: int | None
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

    @property
    def key(self) -> tuple[str, int]:
        return (self.probe_name, self.invocation_index)

    @property
    def has_payload(self) -> bool:
        return self.payload == "full"

    def descriptor(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "point_index": self.point_index,
            "order": self.order,
            "probe_name": self.probe_name,
            "invocation_index": self.invocation_index,
            "replay_index": self.replay_index,
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
        """Materialize the CPU tensor, validating its content digest."""

        if self.payload != "full":
            raise TensorPayloadUnavailableError(
                f"observation {self.probe_name!r}[{self.invocation_index}] "
                "contains summary data only"
            )
        cached = self._tensor_cache.get("tensor")
        if cached is not None:
            return cached
        if self._blob_path is None:
            raise TensorBundleError(
                f"observation {self.probe_name!r}[{self.invocation_index}] "
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


@dataclass(frozen=True)
class TensorPoint:
    """One ordered set of named tensor observations."""

    run_id: str
    index: int
    label: str
    timestamp: float
    metadata: Mapping[str, Any]
    observations: tuple[TensorObservation, ...]

    @cached_property
    def by_key(self) -> Mapping[tuple[str, int], TensorObservation]:
        return MappingProxyType({item.key: item for item in self.observations})

    def observation(
        self,
        probe_name: str,
        invocation_index: int = 0,
    ) -> TensorObservation:
        try:
            return self.by_key[(probe_name, invocation_index)]
        except KeyError as exc:
            raise KeyError(
                f"tensor observation {probe_name!r}[{invocation_index}] "
                f"does not exist at point {self.label!r}"
            ) from exc

    def descriptor(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "index": self.index,
            "label": self.label,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
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
    provenance: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    run_metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    bundle_dir: Path | None = field(default=None, compare=False)

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
            "provenance": dict(self.provenance),
            "run_metadata": dict(self.run_metadata),
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
        options: Any = None,
    ) -> Any:
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
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TensorBundleError(
                f"could not read tensor bundle {root}: {exc}"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise TensorBundleError("tensor bundle manifest must be a JSON object")
        if manifest.get("schema") != BUNDLE_SCHEMA:
            raise TensorBundleError(
                f"unsupported tensor bundle schema {manifest.get('schema')!r}"
            )
        _require_exact_fields(manifest, _MANIFEST_FIELDS, "tensor bundle manifest")

        run_id = _nonempty_string(manifest["run_id"], "run_id")
        name = _nonempty_string(manifest["name"], "name")
        try:
            execution = validate_execution_mode(str(manifest["execution"]))
            default_payload = validate_payload_kind(str(manifest["default_payload"]))
        except ValueError as exc:
            raise TensorBundleError(f"invalid tensor bundle mode: {exc}") from exc
        rank = _optional_int(manifest["rank"], "rank")
        group_id = _optional_nonempty_string(manifest["group_id"], "group_id")
        world_size = _optional_int(manifest["world_size"], "world_size")
        try:
            _validate_group_identity(rank, group_id, world_size)
        except ValueError as exc:
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
            _require_exact_fields(
                raw_point,
                _POINT_FIELDS,
                f"tensor point {expected_index}",
            )
            index = int(raw_point["index"])
            label = _nonempty_string(raw_point["label"], "tensor point label")
            if index != expected_index:
                raise TensorBundleError(
                    "tensor bundle point indices must be contiguous and ordered"
                )
            if label in labels:
                raise TensorBundleError(
                    f"tensor bundle has duplicate point label {label!r}"
                )
            labels.add(label)
            metadata = _mapping_value(raw_point["metadata"], "point metadata")
            raw_observations = raw_point["observations"]
            if not isinstance(raw_observations, list):
                raise TensorBundleError(
                    f"observations for point {label!r} must be a JSON list"
                )
            observations: list[TensorObservation] = []
            keys: set[tuple[str, int]] = set()
            for expected_order, raw_observation in enumerate(raw_observations):
                observation = _observation_from_manifest(
                    root,
                    run_id,
                    index,
                    expected_order,
                    raw_observation,
                    cache_tensors=cache_tensors,
                )
                if observation.key in keys:
                    raise TensorBundleError(
                        f"point {label!r} has duplicate observation "
                        f"{observation.probe_name!r}[{observation.invocation_index}]"
                    )
                keys.add(observation.key)
                observations.append(observation)
            points.append(
                TensorPoint(
                    run_id=run_id,
                    index=index,
                    label=label,
                    timestamp=float(raw_point["timestamp"]),
                    metadata=MappingProxyType(metadata),
                    observations=tuple(observations),
                )
            )

        provenance = _mapping_value(manifest["provenance"], "provenance")
        run_metadata = _mapping_value(manifest["run_metadata"], "run_metadata")
        finished_at = manifest["finished_at"]
        return cls(
            run_id=run_id,
            name=name,
            execution=execution,
            rank=rank,
            created_at=float(manifest["created_at"]),
            finished_at=float(finished_at) if finished_at is not None else None,
            complete=bool(manifest["complete"]),
            default_payload=default_payload,
            points=tuple(points),
            group_id=group_id,
            world_size=world_size,
            provenance=MappingProxyType(provenance),
            run_metadata=MappingProxyType(run_metadata),
            bundle_dir=root,
        )


@dataclass(frozen=True)
class _PendingObservation:
    order: int
    probe_name: str
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
    probe_name: str
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
        _validate_synchronize_target(synchronize)
        self.synchronize = synchronize
        self.rank = _default_rank() if rank is None else int(rank)
        self.group_id = group_id
        self.world_size = (
            _default_world_size() if world_size is None else int(world_size)
        )
        _validate_group_identity(self.rank, self.group_id, self.world_size)
        metadata = _json_value(dict(run_metadata or {}), "$.run_metadata")
        provenance = _json_value(runtime_provenance(), "$.provenance")
        assert isinstance(metadata, dict)
        assert isinstance(provenance, dict)
        self.run_metadata = MappingProxyType(metadata)
        self._provenance: dict[str, Any] = provenance
        self._run_id = uuid.uuid4().hex
        self._created_at = time.time()
        self._finished_at: float | None = None
        self._points: list[TensorPoint] = []
        self._result: TensorRun | None = None
        self._closed = False
        self._active_point: _ActivePoint | None = None
        self._capture_slots: list[_CaptureSlot] = []
        self._capture_invocation_counts: dict[str, int] = {}
        self._device: torch.device | None = None
        self._session_probe: TensorProbe | None = None
        self._last_recorded_replay_index: int | None = None

        if self.execution == "cuda_graph":
            self._device = _resolve_cuda_device(device)
            self._session_probe = TensorProbe(
                f"{self.name}.__tensor_run__",
                [RecordTensor()],
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
        name: str,
        tensor: torch.Tensor,
        *,
        payload: PayloadKind | None = None,
    ) -> torch.Tensor:
        """Return tensor unchanged while recording one named observation."""

        self._ensure_open()
        if not name:
            raise ValueError("observation name must be non-empty")
        selected_payload = (
            self.default_payload if payload is None else validate_payload_kind(payload)
        )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("observe tensor must be a torch.Tensor")

        if self.execution == "eager":
            if self._active_point is None:
                return tensor
            source = self._validate_tensor(tensor)
            invocation_index = self._next_eager_invocation(name)
            staging = torch.empty(
                tuple(source.shape),
                dtype=source.dtype,
                device="cpu",
                pin_memory=True,
            )
            staging.copy_(source.detach(), non_blocking=True)
            self._active_point.pending.append(
                _PendingObservation(
                    order=len(self._active_point.pending),
                    probe_name=name,
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

        if not torch.cuda.is_current_stream_capturing():
            return tensor
        self._validate_tensor(tensor)
        assert self._session_probe is not None
        result = self._session_probe(tensor)
        invocation_index = self._capture_invocation_counts.get(name, 0)
        self._capture_invocation_counts[name] = invocation_index + 1
        self._capture_slots.append(
            _CaptureSlot(
                order=len(self._capture_slots),
                probe_name=name,
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
        name: str,
        tensor: torch.Tensor,
        *,
        payload: PayloadKind | None = None,
        strict: bool = False,
    ) -> RemovableHandle | None:
        """Register an autograd hook that records a named gradient observation."""

        self._ensure_open()
        if not tensor.requires_grad:
            if strict:
                raise RuntimeError(
                    "cannot watch gradients for a tensor that does not require grad"
                )
            return None

        def hook(grad: torch.Tensor | None) -> torch.Tensor | None:
            if grad is None:
                return None
            self.observe(name, grad, payload=payload)
            return grad

        return tensor.register_hook(hook)

    @contextmanager
    def point(
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
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "TensorRecorder.point() cannot run during CUDA graph capture; "
                "capture observe() calls first, then wrap graph.replay()"
            )
        target = self.synchronize if synchronize is None else synchronize
        _validate_synchronize_target(target)
        serializable_metadata = _json_value(
            dict(metadata or {}),
            "$.point.metadata",
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

    def snapshot_run(self) -> TensorRun:
        return self._build_run(complete=self._result is not None)

    def finish(self) -> TensorRun:
        if self._result is not None:
            return self._result
        if self._active_point is not None:
            raise TensorDebugError("cannot finish while a tensor point is active")
        self._finished_at = time.time()
        self._result = self._build_run(complete=True)
        self._write_manifest(complete=True)
        return self._result

    @property
    def result(self) -> TensorRun:
        if self._result is None:
            raise TensorDebugError("tensor recorder has not been finished")
        return self._result

    def close(self) -> None:
        """Release native resources after the captured graph can no longer replay."""

        if self._closed:
            return
        if self._active_point is not None:
            raise TensorDebugError("cannot close while a tensor point is active")
        if self._session_probe is not None:
            self._session_probe.close()
            self._session_probe = None
        self._closed = True

    def __enter__(self) -> "TensorRecorder":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self.finish()
        finally:
            self.close()

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
            _synchronize_results(self._device, active.synchronize)
        self._commit_point(active.label, active.metadata, active.pending)
        self._active_point = None

    def _abort_point(self, active: _ActivePoint) -> None:
        if self.execution == "eager" and active.pending and self._device is not None:
            try:
                _synchronize_results(self._device, active.synchronize)
            except Exception:
                pass
        active.pending.clear()
        self._active_point = None

    def _collect_cuda_graph_observations(
        self,
        active: _ActivePoint,
    ) -> list[_PendingObservation]:
        if self._session_probe is None or self._device is None:
            raise TensorDebugError("CUDA Graph tensor recorder has no native session")
        if not self._capture_slots:
            raise TensorDebugError(
                "CUDA Graph tensor recorder captured no observations"
            )
        _synchronize_results(self._device, active.synchronize)
        snapshots = self._session_probe.snapshots(synchronize=False)
        if len(snapshots) != len(self._capture_slots):
            raise TensorDebugError(
                "captured tensor slot count does not match recorded snapshots: "
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
                "TensorRecorder.point() did not observe a CUDA Graph replay"
            )
        if (
            self._last_recorded_replay_index is not None
            and replay_index <= self._last_recorded_replay_index
        ):
            raise TensorDebugError(
                "TensorRecorder.point() did not observe a new CUDA Graph replay"
            )
        pending: list[_PendingObservation] = []
        for slot, snapshot in zip(self._capture_slots, snapshots):
            if snapshot.invocation_index != slot.order:
                raise TensorDebugError(
                    "captured tensor invocation ordering changed unexpectedly"
                )
            if (
                tuple(snapshot.shape or ()) != slot.shape
                or snapshot.dtype != slot.dtype
            ):
                raise TensorDebugError(
                    f"captured tensor metadata changed for {slot.probe_name!r}"
                )
            pending.append(
                _PendingObservation(
                    order=slot.order,
                    probe_name=slot.probe_name,
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
        self._last_recorded_replay_index = replay_index
        return pending

    def _commit_point(
        self,
        label: str,
        metadata: Mapping[str, Any],
        pending: Sequence[_PendingObservation],
    ) -> TensorPoint:
        index = len(self._points)
        observations: list[TensorObservation] = []
        seen: set[tuple[str, int]] = set()
        for expected_order, item in enumerate(pending):
            if item.order != expected_order:
                raise TensorDebugError("tensor observation order must be contiguous")
            key = (item.probe_name, item.invocation_index)
            if key in seen:
                raise TensorDebugError(
                    f"duplicate tensor observation {item.probe_name!r}"
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
                    run_id=self._run_id,
                    point_index=index,
                    order=expected_order,
                    probe_name=item.probe_name,
                    invocation_index=item.invocation_index,
                    replay_index=item.replay_index,
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
        )
        self._points.append(point)
        self._record_device_provenance()
        self._write_manifest(complete=False)
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

    def _write_manifest(self, *, complete: bool) -> None:
        if self.bundle_dir is None:
            return
        payload = {
            "schema": BUNDLE_SCHEMA,
            "run_id": self._run_id,
            "name": self.name,
            "execution": self.execution,
            "rank": self.rank,
            "group_id": self.group_id,
            "world_size": self.world_size,
            "created_at": self._created_at,
            "finished_at": self._finished_at,
            "complete": complete,
            "default_payload": self.default_payload,
            "provenance": self._provenance,
            "run_metadata": dict(self.run_metadata),
            "points": [
                _point_manifest(point, self.bundle_dir) for point in self._points
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

    def _build_run(self, *, complete: bool) -> TensorRun:
        return TensorRun(
            run_id=self._run_id,
            name=self.name,
            execution=self.execution,
            rank=self.rank,
            created_at=self._created_at,
            finished_at=self._finished_at,
            complete=complete,
            default_payload=self.default_payload,
            points=tuple(self._points),
            group_id=self.group_id,
            world_size=self.world_size,
            provenance=MappingProxyType(dict(self._provenance)),
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
                "probe_name": item.probe_name,
                "invocation_index": item.invocation_index,
                "replay_index": item.replay_index,
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
        "metadata": dict(point.metadata),
        "observations": observations,
    }


def _observation_from_manifest(
    root: Path,
    run_id: str,
    point_index: int,
    expected_order: int,
    raw: Any,
    *,
    cache_tensors: bool,
) -> TensorObservation:
    if not isinstance(raw, Mapping):
        raise TensorBundleError(
            f"tensor observation {expected_order} must be a JSON object"
        )
    _require_exact_fields(
        raw,
        _OBSERVATION_FIELDS,
        f"tensor observation {expected_order}",
    )
    order = int(raw["order"])
    if order != expected_order:
        raise TensorBundleError(
            "tensor observation order must be contiguous and ordered"
        )
    probe_name = _nonempty_string(raw["probe_name"], "probe_name")
    invocation_index = int(raw["invocation_index"])
    if invocation_index < 0:
        raise TensorBundleError("invocation_index must be non-negative")
    replay_index = _optional_int(raw["replay_index"], "replay_index")
    if replay_index is not None and replay_index < 0:
        raise TensorBundleError("replay_index must be non-negative")
    shape = _integer_tuple(raw["shape"], "shape", non_negative=True)
    stride = _integer_tuple(raw["stride"], "stride", non_negative=True)
    if len(stride) != len(shape):
        raise TensorBundleError("tensor stride rank must equal shape rank")
    dtype_name = str(raw["dtype"])
    try:
        dtype = _NAME_DTYPES[dtype_name]
    except KeyError as exc:
        raise TensorBundleError(f"unsupported tensor dtype {dtype_name!r}") from exc
    source_device = _nonempty_string(raw["source_device"], "source_device")
    nbytes = int(raw["nbytes"])
    expected_nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if nbytes != expected_nbytes:
        raise TensorBundleError(
            f"tensor observation {probe_name!r} has nbytes={nbytes}; "
            f"expected {expected_nbytes}"
        )
    try:
        payload = validate_payload_kind(str(raw["payload"]))
    except ValueError as exc:
        raise TensorBundleError(f"invalid observation payload: {exc}") from exc
    digest = str(raw["sha256"])
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
                f"tensor blob for {probe_name!r}[{invocation_index}] does not exist"
            )
        if blob_path.stat().st_size != nbytes:
            raise TensorBundleError(
                f"tensor blob for {probe_name!r}[{invocation_index}] "
                "has an unexpected size"
            )
    elif raw["blob_file"] is not None:
        raise TensorBundleError("summary-only observations must not reference a blob")

    return TensorObservation(
        run_id=run_id,
        point_index=point_index,
        order=order,
        probe_name=probe_name,
        invocation_index=invocation_index,
        replay_index=replay_index,
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


def _validate_synchronize_target(synchronize: SynchronizeTarget) -> None:
    if not isinstance(synchronize, (bool, torch.cuda.Stream, torch.device)):
        raise TypeError(
            "synchronize must be a bool, torch.cuda.Stream, or torch.device"
        )


def _synchronize_results(
    recorder_device: torch.device,
    synchronize: SynchronizeTarget,
) -> None:
    _validate_synchronize_target(synchronize)
    if isinstance(synchronize, bool):
        if not synchronize:
            return
        target_device = recorder_device
        stream = None
    elif isinstance(synchronize, torch.cuda.Stream):
        target_device = torch.device(synchronize.device)
        stream = synchronize
    else:
        target_device = synchronize
        stream = None
    if target_device.type != "cuda":
        raise ValueError("synchronize target must identify a CUDA device")
    if target_device.index is None:
        target_device = torch.device("cuda", torch.cuda.current_device())
    if target_device != recorder_device:
        raise ValueError(
            f"synchronize target {target_device} does not match recorder device "
            f"{recorder_device}"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "cannot synchronize TensorRecorder results during CUDA graph capture"
        )
    if stream is not None:
        stream.synchronize()
    else:
        torch.cuda.synchronize(target_device)


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


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    context: str,
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
    raise TensorBundleError(f"{context} has invalid fields: {', '.join(details)}")


def _json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TensorBundleError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TensorBundleError(
                    f"{path} contains non-string mapping key {key!r}"
                )
            output[key] = _json_value(item, f"{path}.{key}")
        return output
    raise TensorBundleError(f"{path} contains unsupported value {type(value).__name__}")


def _mapping_value(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TensorBundleError(f"{context} must be a JSON object")
    result = _json_value(dict(value), f"$.{context}")
    assert isinstance(result, dict)
    return result


def _integer_tuple(
    value: Any,
    context: str,
    *,
    non_negative: bool,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TensorBundleError(f"tensor {context} must be a JSON list")
    result = tuple(int(item) for item in value)
    if non_negative and any(item < 0 for item in result):
        raise TensorBundleError(f"tensor {context} values must be non-negative")
    return result


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TensorBundleError(f"{context} must be a non-empty string")
    return value


def _optional_nonempty_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, context)


def _optional_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TensorBundleError(f"{context} must be an integer or null")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TensorBundleError(f"{context} must be an integer or null") from exc


def _validate_group_identity(
    rank: int | None,
    group_id: str | None,
    world_size: int | None,
) -> None:
    if group_id is not None and not group_id:
        raise ValueError("group_id must be non-empty when provided")
    if world_size is not None and world_size < 1:
        raise ValueError("world_size must be >= 1")
    if rank is not None and rank < 0:
        raise ValueError("rank must be non-negative")
    if rank is not None and world_size is not None and rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")


def _default_rank() -> int | None:
    value = os.environ.get("RANK")
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    try:
        import torch.distributed as distributed
    except ImportError:
        return None
    if not distributed.is_available() or not distributed.is_initialized():
        return None
    return int(distributed.get_rank())


def _default_world_size() -> int | None:
    value = os.environ.get("WORLD_SIZE")
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    try:
        import torch.distributed as distributed
    except ImportError:
        return None
    if not distributed.is_available() or not distributed.is_initialized():
        return None
    return int(distributed.get_world_size())
