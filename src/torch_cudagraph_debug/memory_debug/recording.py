"""Memory snapshot collection, ownership, and persistence."""

from __future__ import annotations

import gzip
import json
import math
import time
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

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
    JSONValue,
    _freeze_json,
    _thaw_json,
)

from ._collector import (
    DeviceSelector,
    SnapshotProvider,
    SynchronizeTarget,
    _MemoryCollector,
)
from ._pool_identity import (
    MemoryObservationKey,
    MemoryPoolKey,
    PoolId,
    StreamId,
    normalize_device_index,
    normalize_pool_id,
    normalize_stream,
)
from .aggregation import summarize_allocator_scopes, summarize_pools
from .allocator_snapshot import AllocatorSnapshotData
from .errors import MemoryBundleError, MemoryDebugError, MemoryOwnershipError
from .stats import AllocatorScope, MemoryStats

if TYPE_CHECKING:
    from .attribution import MemoryAttributionOptions, MemoryLifetimeOptions
    from .reports import (
        MemoryAllocationLifetimeAnalysis,
        MemoryPointComparison,
        MemoryTimeline,
    )

BUNDLE_SCHEMA = "torch-cudagraph-debug/memory-run"
BUNDLE_FORMAT_VERSION = 1
_MEMORY_STATS_FIELDS = tuple(item.name for item in fields(MemoryStats))
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "run_id",
        "name",
        "rank",
        "group_id",
        "world_size",
        "created_at",
        "finished_at",
        "complete",
        "provenance",
        "run_metadata",
        "points",
    }
)
_POINT_FIELDS = frozenset(
    {
        "index",
        "label",
        "timestamp",
        "metadata",
        "boundary_marker",
        "snapshot_file",
        "warnings",
        "observations",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {"order", "device_index", "pool_id", "stream", *_MEMORY_STATS_FIELDS}
)


@dataclass(frozen=True)
class MemoryObservation:
    """One device/pool/stream allocator state captured at a boundary."""

    order: int
    device_index: int
    pool_id: PoolId
    stream: StreamId
    stats: MemoryStats

    def __post_init__(self) -> None:
        if type(self.order) is not int or self.order < 0:
            raise ValueError("memory observation order must be non-negative")
        object.__setattr__(
            self, "device_index", normalize_device_index(self.device_index)
        )
        object.__setattr__(self, "pool_id", normalize_pool_id(self.pool_id))
        object.__setattr__(self, "stream", normalize_stream(self.stream))
        if not isinstance(self.stats, MemoryStats):
            raise TypeError("memory observation stats must be MemoryStats")

    @property
    def key(self) -> MemoryObservationKey:
        return MemoryObservationKey(self.device_index, self.pool_id, self.stream)

    def descriptor(self) -> dict[str, object]:
        return {
            "order": self.order,
            "device_index": self.device_index,
            "pool_id": list(self.pool_id),
            "stream": self.stream,
            "stats": self.stats.to_dict(),
        }


@dataclass(frozen=True)
class MemoryPoint:
    """One immutable labeled allocator snapshot owned by a run."""

    run_id: str
    index: int
    label: str
    timestamp: float
    metadata: Mapping[str, FrozenJSONValue]
    boundary_marker: str
    observations: tuple[MemoryObservation, ...]
    warnings: tuple[str, ...] = ()
    _snapshot_path: Path | None = field(default=None, repr=False, compare=False)
    _snapshot_cache: dict[str, FrozenJSONValue] = field(
        default_factory=dict, repr=False, compare=False
    )
    _cache_snapshots: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.run_id or not self.label:
            raise ValueError("memory point run_id and label must be non-empty")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("memory point index must be non-negative")
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not math.isfinite(float(self.timestamp))
        ):
            raise ValueError("memory point timestamp must be finite")
        if not self.boundary_marker:
            raise ValueError("memory point boundary_marker must be non-empty")
        if [item.order for item in self.observations] != list(
            range(len(self.observations))
        ):
            raise ValueError("memory observations must be contiguous and ordered")
        keys = [item.key for item in self.observations]
        if len(keys) != len(set(keys)):
            raise ValueError("memory observations must have unique keys")
        object.__setattr__(
            self,
            "metadata",
            cast(Mapping[str, FrozenJSONValue], _freeze_json(self.metadata)),
        )
        for key, value in tuple(self._snapshot_cache.items()):
            self._snapshot_cache[key] = _freeze_json(
                cast(JSONValue | FrozenJSONValue, value)
            )

    @cached_property
    def by_key(self) -> Mapping[MemoryObservationKey, MemoryObservation]:
        return MappingProxyType({item.key: item for item in self.observations})

    @cached_property
    def observation_stats(self) -> Mapping[MemoryObservationKey, MemoryStats]:
        return MappingProxyType(
            {key: observation.stats for key, observation in self.by_key.items()}
        )

    @cached_property
    def pool_stats(self) -> Mapping[MemoryPoolKey, MemoryStats]:
        return MappingProxyType(summarize_pools(self.observation_stats))

    @cached_property
    def allocator_scope_stats(self) -> Mapping[AllocatorScope, MemoryStats]:
        return MappingProxyType(summarize_allocator_scopes(self.pool_stats))

    def observation(
        self, device_index: int, pool_id: Any, stream: Any
    ) -> MemoryObservation:
        key = MemoryObservationKey(device_index, pool_id, stream)
        try:
            return self.by_key[key]
        except KeyError as exc:
            raise KeyError(
                f"memory observation {key.label} does not exist at point {self.label!r}"
            ) from exc

    @property
    def allocator_settings(self) -> Mapping[str, FrozenJSONValue]:
        raw = self.raw_snapshot()
        if not isinstance(raw, Mapping):
            return MappingProxyType({})
        settings = raw.get("allocator_settings", MappingProxyType({}))
        if not isinstance(settings, Mapping):
            return MappingProxyType({})
        return cast(Mapping[str, FrozenJSONValue], settings)

    def raw_snapshot(self) -> FrozenJSONValue:
        """Load and return a recursively immutable allocator snapshot."""

        cached = self._snapshot_cache.get("snapshot") if self._cache_snapshots else None
        if cached is not None:
            return cached
        if self._snapshot_path is None:
            raise MemoryBundleError(f"point {self.label!r} has no snapshot payload")
        try:
            with gzip.open(self._snapshot_path, "rt", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            raise MemoryBundleError(
                f"could not load snapshot for point {self.label!r}: {exc}"
            ) from exc
        snapshot = strict_json_loads(
            text,
            error_type=MemoryBundleError,
            context=f"snapshot for point {self.label!r}",
        )
        if not isinstance(snapshot, (Mapping, list)):
            raise MemoryBundleError(
                f"snapshot for point {self.label!r} has an invalid root type"
            )
        frozen = _freeze_json(cast(JSONValue, snapshot))
        if self._cache_snapshots:
            self._snapshot_cache["snapshot"] = frozen
        return frozen

    def descriptor(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "index": self.index,
            "label": self.label,
            "timestamp": self.timestamp,
            "metadata": _thaw_json(cast(FrozenJSONValue, self.metadata)),
            "boundary_marker": self.boundary_marker,
            "observation_count": len(self.observations),
            "warnings": list(self.warnings),
        }


MemoryPointReference: TypeAlias = str | int | MemoryPoint


@dataclass(frozen=True)
class MemoryLifetimeSelection:
    """Select all, active-at, or born-between allocation generations."""

    mode: Literal["all", "active_at", "born_between"]
    start: MemoryPointReference | None = None
    end: MemoryPointReference | None = None

    def __post_init__(self) -> None:
        if self.mode == "all":
            if self.start is not None or self.end is not None:
                raise ValueError("all selection does not accept point references")
        elif self.mode == "active_at":
            if self.start is None or self.end is not None:
                raise ValueError("active_at selection requires exactly one point")
        elif self.mode == "born_between":
            if self.start is None or self.end is None:
                raise ValueError("born_between selection requires two points")
        else:
            raise ValueError(f"unsupported lifetime selection mode {self.mode!r}")

    @classmethod
    def all(cls) -> "MemoryLifetimeSelection":
        return cls("all")

    @classmethod
    def active_at(cls, point: MemoryPointReference) -> "MemoryLifetimeSelection":
        return cls("active_at", start=point)

    @classmethod
    def born_between(
        cls, start: MemoryPointReference, end: MemoryPointReference
    ) -> "MemoryLifetimeSelection":
        return cls("born_between", start=start, end=end)


@dataclass(frozen=True)
class MemoryRun:
    """Immutable sequence of allocator snapshot points."""

    run_id: str
    name: str
    rank: int | None
    created_at: float
    finished_at: float | None
    complete: bool
    points: tuple[MemoryPoint, ...]
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
        if not self.run_id or not self.name:
            raise ValueError("memory run run_id and name must be non-empty")
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
        object.__setattr__(
            self,
            "provenance",
            cast(Mapping[str, FrozenJSONValue], _freeze_json(self.provenance)),
        )
        object.__setattr__(
            self,
            "run_metadata",
            cast(Mapping[str, FrozenJSONValue], _freeze_json(self.run_metadata)),
        )
        labels: set[str] = set()
        for expected_index, point in enumerate(self.points):
            if point.run_id != self.run_id or point.index != expected_index:
                raise ValueError("memory run points must be owned and ordered")
            if point.label in labels:
                raise ValueError(f"duplicate memory point label {point.label!r}")
            labels.add(point.label)

    def descriptor(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "rank": self.rank,
            "group_id": self.group_id,
            "world_size": self.world_size,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "complete": self.complete,
            "provenance": _thaw_json(cast(FrozenJSONValue, self.provenance)),
            "run_metadata": _thaw_json(cast(FrozenJSONValue, self.run_metadata)),
        }

    def point(self, ref: str | int | MemoryPoint) -> MemoryPoint:
        if isinstance(ref, MemoryPoint):
            if ref.run_id != self.run_id:
                raise MemoryOwnershipError(
                    f"point {ref.label!r} belongs to run {ref.run_id}, not {self.run_id}"
                )
            if ref.index < 0 or ref.index >= len(self.points):
                raise MemoryOwnershipError(
                    f"point {ref.label!r} is not present in run {self.run_id}"
                )
            owned = self.points[ref.index]
            if owned.label != ref.label:
                raise MemoryOwnershipError(
                    f"point {ref.label!r} is not present in run {self.run_id}"
                )
            return owned
        if isinstance(ref, int):
            try:
                return self.points[ref]
            except IndexError as exc:
                raise IndexError(f"memory point index {ref} does not exist") from exc
        for point in self.points:
            if point.label == ref:
                return point
        raise KeyError(f"memory point {ref!r} does not exist")

    def __getitem__(self, ref: str | int) -> MemoryPoint:
        return self.point(ref)

    def between(
        self, start: str | int | MemoryPoint, end: str | int | MemoryPoint
    ) -> "MemoryRange":
        start_point = self.point(start)
        end_point = self.point(end)
        if end_point.index <= start_point.index:
            raise ValueError("end point must come after start point")
        return MemoryRange(self, start_point, end_point)

    def compare(
        self,
        reference: str | int | MemoryPoint,
        candidate: str | int | MemoryPoint,
        *,
        attribution: MemoryAttributionOptions | None = None,
    ) -> MemoryPointComparison:
        from .attribution import MemoryAttributionOptions
        from .comparison import _compare_same_run

        memory_range = self.between(reference, candidate)
        return _compare_same_run(
            self,
            memory_range.start,
            memory_range.end,
            attribution or MemoryAttributionOptions(),
        )

    def timeline(
        self,
        *,
        attribution: MemoryAttributionOptions | None = None,
    ) -> MemoryTimeline:
        from .attribution import MemoryAttributionOptions
        from .timeline import _build_timeline

        return _build_timeline(self, attribution or MemoryAttributionOptions())

    def lifetimes(
        self,
        selection: MemoryLifetimeSelection | None = None,
        *,
        through: MemoryPointReference | None = None,
        options: MemoryLifetimeOptions | None = None,
    ) -> MemoryAllocationLifetimeAnalysis:
        """Analyze allocation cohorts selected by an explicit lifetime query."""

        from .attribution import MemoryLifetimeOptions
        from .lifetimes import analyze_allocation_lifetimes

        if not self.points:
            raise MemoryDebugError("allocation lifetimes require at least one point")
        selected_query = selection or MemoryLifetimeSelection.all()
        active_point = None
        birth_range = None
        if selected_query.mode == "active_at":
            assert selected_query.start is not None
            active_point = self.point(selected_query.start)
        elif selected_query.mode == "born_between":
            assert selected_query.start is not None and selected_query.end is not None
            birth_range = self.between(selected_query.start, selected_query.end)
        start = active_point or (birth_range.start if birth_range else self.points[0])
        end = self.point(through) if through is not None else self.points[-1]
        if end.index < start.index:
            raise ValueError("through point must not come before the lifetime anchor")
        if birth_range is not None and end.index < birth_range.end.index:
            raise ValueError("through point must not come before born_between end")
        return analyze_allocation_lifetimes(
            self,
            start=start,
            end=end,
            active_at=active_point,
            born_between=(
                (birth_range.start, birth_range.end) if birth_range else None
            ),
            options=options or MemoryLifetimeOptions(),
        )

    @classmethod
    def load(
        cls,
        bundle_dir: str | Path,
        *,
        cache_snapshots: bool = True,
    ) -> "MemoryRun":
        root = Path(bundle_dir).resolve()
        manifest_path = root / "manifest.json"
        try:
            manifest_text = manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MemoryBundleError(
                f"could not read memory bundle {root}: {exc}"
            ) from exc
        manifest = strict_json_loads(
            manifest_text,
            error_type=MemoryBundleError,
            context=f"memory bundle {root}",
        )
        if not isinstance(manifest, Mapping):
            raise MemoryBundleError("memory bundle manifest must be a JSON object")
        if manifest.get("schema") != BUNDLE_SCHEMA:
            raise MemoryBundleError(
                f"unsupported memory bundle schema {manifest.get('schema')!r}"
            )
        if manifest.get("format_version") != BUNDLE_FORMAT_VERSION:
            raise MemoryBundleError(
                "unsupported memory bundle format_version "
                f"{manifest.get('format_version')!r}"
            )
        require_exact_fields(
            manifest,
            _MANIFEST_FIELDS,
            "memory bundle manifest",
            error_type=MemoryBundleError,
        )

        run_id = require_nonempty_string(
            manifest["run_id"], "run_id", error_type=MemoryBundleError
        )
        name = require_nonempty_string(
            manifest["name"], "name", error_type=MemoryBundleError
        )

        raw_points = manifest.get("points")
        if not isinstance(raw_points, list):
            raise MemoryBundleError("memory bundle points must be a JSON list")
        points: list[MemoryPoint] = []
        labels: set[str] = set()
        for expected_index, raw in enumerate(raw_points):
            if not isinstance(raw, Mapping):
                raise MemoryBundleError(
                    f"memory point {expected_index} must be a JSON object"
                )
            require_exact_fields(
                raw,
                _POINT_FIELDS,
                f"memory point {expected_index}",
                error_type=MemoryBundleError,
            )
            index = require_nonnegative_int(
                raw["index"], "memory point index", error_type=MemoryBundleError
            )
            label = require_nonempty_string(
                raw["label"], "memory point label", error_type=MemoryBundleError
            )
            if index != expected_index:
                raise MemoryBundleError(
                    "memory bundle point indices must be contiguous and ordered"
                )
            if not label or label in labels:
                raise MemoryBundleError(
                    f"memory bundle has an empty or duplicate label {label!r}"
                )
            labels.add(label)
            snapshot_path = _safe_bundle_path(
                root, raw["snapshot_file"], expected_index=expected_index
            )
            if not snapshot_path.is_file():
                raise MemoryBundleError(
                    f"snapshot file for point {label!r} does not exist"
                )
            observations: list[MemoryObservation] = []
            keys: set[MemoryObservationKey] = set()
            raw_observations = raw["observations"]
            if not isinstance(raw_observations, list):
                raise MemoryBundleError(
                    f"observations for point {label!r} must be a JSON list"
                )
            for expected_order, observation_row in enumerate(raw_observations):
                observation = _observation_from_manifest(
                    expected_order,
                    observation_row,
                )
                if observation.key in keys:
                    raise MemoryBundleError(
                        f"point {label!r} has duplicate observation {observation.key!r}"
                    )
                keys.add(observation.key)
                observations.append(observation)
            metadata = require_json_mapping(
                raw["metadata"],
                f"metadata for point {label!r}",
                error_type=MemoryBundleError,
            )
            boundary_marker = require_nonempty_string(
                raw["boundary_marker"],
                f"boundary_marker for point {label!r}",
                error_type=MemoryBundleError,
            )
            warnings = raw["warnings"]
            if not isinstance(warnings, list) or not all(
                isinstance(item, str) for item in warnings
            ):
                raise MemoryBundleError(
                    f"warnings for point {label!r} must be a JSON string list"
                )
            try:
                point = MemoryPoint(
                    run_id=run_id,
                    index=index,
                    label=label,
                    timestamp=require_finite_number(
                        raw["timestamp"],
                        f"timestamp for point {label!r}",
                        error_type=MemoryBundleError,
                    ),
                    metadata=MappingProxyType(metadata),
                    boundary_marker=boundary_marker,
                    observations=tuple(observations),
                    warnings=tuple(warnings),
                    _snapshot_path=snapshot_path,
                    _cache_snapshots=cache_snapshots,
                )
            except (TypeError, ValueError) as exc:
                raise MemoryBundleError(
                    f"invalid memory point {label!r}: {exc}"
                ) from exc
            points.append(point)

        rank = require_optional_int(
            manifest["rank"], "rank", error_type=MemoryBundleError
        )
        group_id = require_optional_nonempty_string(
            manifest["group_id"], "group_id", error_type=MemoryBundleError
        )
        world_size = require_optional_int(
            manifest["world_size"], "world_size", error_type=MemoryBundleError
        )
        try:
            validate_group_identity(rank, group_id, world_size)
        except (TypeError, ValueError) as exc:
            raise MemoryBundleError(f"invalid memory run identity: {exc}") from exc
        provenance = require_json_mapping(
            manifest["provenance"], "provenance", error_type=MemoryBundleError
        )
        run_metadata = require_json_mapping(
            manifest["run_metadata"], "run_metadata", error_type=MemoryBundleError
        )
        finished_value = manifest["finished_at"]
        try:
            return cls(
                run_id=run_id,
                name=name,
                rank=rank,
                created_at=require_finite_number(
                    manifest["created_at"],
                    "created_at",
                    error_type=MemoryBundleError,
                ),
                finished_at=(
                    require_finite_number(
                        finished_value,
                        "finished_at",
                        error_type=MemoryBundleError,
                    )
                    if finished_value is not None
                    else None
                ),
                complete=require_bool(
                    manifest["complete"], "complete", error_type=MemoryBundleError
                ),
                points=tuple(points),
                group_id=group_id,
                world_size=world_size,
                provenance=MappingProxyType(provenance),
                run_metadata=MappingProxyType(run_metadata),
                bundle_dir=root,
            )
        except (TypeError, ValueError) as exc:
            raise MemoryBundleError(f"invalid memory run: {exc}") from exc


@dataclass(frozen=True)
class MemoryRange:
    """An ordered pair of points owned by one run."""

    run: MemoryRun
    start: MemoryPoint
    end: MemoryPoint

    def compare(
        self, *, attribution: MemoryAttributionOptions | None = None
    ) -> MemoryPointComparison:
        return self.run.compare(
            self.start,
            self.end,
            attribution=attribution,
        )


class MemoryRecorder:
    """Collect private PyTorch allocator snapshots and produce a MemoryRun."""

    def __init__(
        self,
        *,
        name: str = "run",
        bundle_dir: str | Path | None = None,
        rank: int | None = None,
        devices: DeviceSelector = None,
        synchronize: SynchronizeTarget = True,
        group_id: str | None = None,
        world_size: int | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        self.name = name
        self.bundle_dir = Path(bundle_dir).resolve() if bundle_dir is not None else None
        self.rank, self.world_size = resolve_distributed_identity(rank, world_size)
        self.group_id = group_id
        validate_group_identity(self.rank, self.group_id, self.world_size)
        serializable_metadata = json_value(
            dict(run_metadata or {}),
            "$.run_metadata",
            error_type=MemoryBundleError,
        )
        assert isinstance(serializable_metadata, dict)
        serializable_provenance = json_value(
            runtime_provenance(), "$.provenance", error_type=MemoryBundleError
        )
        assert isinstance(serializable_provenance, dict)
        self.run_metadata = MappingProxyType(serializable_metadata)
        self._provenance: dict[str, Any] = serializable_provenance
        self.synchronize = synchronize
        self._device_selector = devices
        self._collector = _MemoryCollector(
            devices=devices,
            synchronize=synchronize,
        )
        self._run_id = uuid.uuid4().hex
        self._created_at = time.time()
        self._finished_at: float | None = None
        self._points: list[MemoryPoint] = []
        self._result: MemoryRun | None = None
        self._context_active = False

        if self.bundle_dir is not None:
            if self.bundle_dir.exists() and any(self.bundle_dir.iterdir()):
                raise FileExistsError(
                    f"memory bundle directory is not empty: {self.bundle_dir}"
                )
            (self.bundle_dir / "snapshots").mkdir(parents=True, exist_ok=True)
            self._write_manifest(complete=False)

    @classmethod
    def _from_snapshot_provider(
        cls,
        provider: SnapshotProvider,
        **kwargs: Any,
    ) -> "MemoryRecorder":
        recorder = cls(**kwargs)
        recorder._collector = _MemoryCollector(
            devices=recorder._device_selector,
            synchronize=recorder.synchronize,
            snapshot_provider=provider,
        )
        return recorder

    def record_point(
        self,
        label: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        synchronize: SynchronizeTarget | None = None,
    ) -> MemoryPoint:
        if self._result is not None:
            raise MemoryDebugError(
                "cannot record a point on a finished memory recorder"
            )
        if not label:
            raise ValueError("memory point label must be non-empty")
        if any(point.label == label for point in self._points):
            raise ValueError(f"memory point label {label!r} already exists")

        index = len(self._points)
        marker = f"torch-cudagraph-debug:memory-point:{self._run_id}:{index}:{label}"
        capture = self._collector.capture(marker, synchronize=synchronize)
        snapshot = capture.raw_snapshot
        self._record_device_provenance()
        serializable_snapshot = json_value(
            snapshot, "$.snapshot", error_type=MemoryBundleError
        )
        if not isinstance(serializable_snapshot, (dict, list)):
            raise MemoryBundleError("allocator snapshot root must be an object or list")
        serializable_metadata = json_value(
            dict(metadata or {}), "$.metadata", error_type=MemoryBundleError
        )
        assert isinstance(serializable_metadata, dict)

        observations = tuple(
            MemoryObservation(
                order=order,
                device_index=key.device_index,
                pool_id=key.pool_id,
                stream=key.stream,
                stats=stats,
            )
            for order, (key, stats) in enumerate(capture.summaries)
        )
        snapshot_path: Path | None = None
        cache: dict[str, FrozenJSONValue] = {}
        if self.bundle_dir is not None:
            snapshot_path = self._write_snapshot(index, serializable_snapshot)
        else:
            cache["snapshot"] = _freeze_json(serializable_snapshot)

        point = MemoryPoint(
            run_id=self._run_id,
            index=index,
            label=label,
            timestamp=capture.timestamp,
            metadata=cast(
                Mapping[str, FrozenJSONValue], _freeze_json(serializable_metadata)
            ),
            boundary_marker=capture.boundary_marker,
            observations=observations,
            warnings=capture.warnings,
            _snapshot_path=snapshot_path,
            _snapshot_cache=cache,
        )
        self._write_manifest(complete=False, points=(*self._points, point))
        self._points.append(point)
        return point

    @property
    def devices(self) -> tuple[int, ...] | None:
        """Selected device indices, resolved lazily by the first point."""

        return self._collector.devices

    def preview(self) -> MemoryRun:
        """Return an immutable nonterminal view of collected points."""

        if self._result is not None:
            return self._result
        return self._build_run(complete=False)

    def finish(self) -> MemoryRun:
        """Finish collection and return the immutable run; idempotent."""

        if self._context_active:
            raise MemoryDebugError(
                "cannot finish a memory recorder inside its context manager"
            )
        return self._finish()

    def _finish(self) -> MemoryRun:
        if self._result is not None:
            return self._result
        finished_at = time.time()
        candidate = self._build_run(complete=True, finished_at=finished_at)
        self._write_manifest(complete=True, finished_at=finished_at)
        self._finished_at = finished_at
        self._result = candidate
        return candidate

    def _abort(self) -> MemoryRun:
        if self._result is not None:
            return self._result
        finished_at = time.time()
        candidate = self._build_run(complete=False, finished_at=finished_at)
        self._write_manifest(complete=False, finished_at=finished_at)
        self._finished_at = finished_at
        self._result = candidate
        return candidate

    def _build_run(
        self, *, complete: bool, finished_at: float | None = None
    ) -> MemoryRun:
        return MemoryRun(
            run_id=self._run_id,
            name=self.name,
            rank=self.rank,
            created_at=self._created_at,
            finished_at=self._finished_at if finished_at is None else finished_at,
            complete=complete,
            points=tuple(self._points),
            group_id=self.group_id,
            world_size=self.world_size,
            provenance=MappingProxyType(dict(self._provenance)),
            run_metadata=self.run_metadata,
            bundle_dir=self.bundle_dir,
        )

    @property
    def result(self) -> MemoryRun:
        if self._result is None:
            raise MemoryDebugError("memory recorder has not been finished")
        return self._result

    def __enter__(self) -> "MemoryRecorder":
        if self._context_active:
            raise MemoryDebugError("memory recorder context cannot be re-entered")
        if self._result is not None:
            raise MemoryDebugError("cannot enter a finished memory recorder")
        self._context_active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._context_active:
            raise MemoryDebugError("memory recorder context is not active")
        try:
            if exc_type is None:
                self._finish()
            else:
                try:
                    self._abort()
                except Exception as cleanup_error:
                    warnings.warn(
                        "could not persist aborted memory run: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        finally:
            self._context_active = False

    def _write_snapshot(self, index: int, snapshot: AllocatorSnapshotData) -> Path:
        assert self.bundle_dir is not None
        path = self.bundle_dir / "snapshots" / f"{index:04d}.json.gz"
        temporary = path.with_name(path.name + ".tmp")
        try:
            with gzip.open(
                temporary,
                "wt",
                encoding="utf-8",
                compresslevel=1,
            ) as handle:
                json.dump(
                    snapshot,
                    handle,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            temporary.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            raise MemoryBundleError(
                f"could not persist allocator snapshot {index}: {exc}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def _write_manifest(
        self,
        *,
        complete: bool,
        points: tuple[MemoryPoint, ...] | None = None,
        finished_at: float | None = None,
    ) -> None:
        if self.bundle_dir is None:
            return
        payload = {
            "schema": BUNDLE_SCHEMA,
            "format_version": BUNDLE_FORMAT_VERSION,
            "run_id": self._run_id,
            "name": self.name,
            "rank": self.rank,
            "group_id": self.group_id,
            "world_size": self.world_size,
            "created_at": self._created_at,
            "finished_at": self._finished_at if finished_at is None else finished_at,
            "complete": complete,
            "provenance": self._provenance,
            "run_metadata": _thaw_json(cast(FrozenJSONValue, self.run_metadata)),
            "points": [
                _point_manifest(point, self.bundle_dir)
                for point in (self._points if points is None else points)
            ],
        }
        path = self.bundle_dir / "manifest.json"
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            raise MemoryBundleError(
                f"could not persist memory bundle manifest: {exc}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)

    def _record_device_provenance(self) -> None:
        if self._collector.uses_snapshot_provider or "devices" in self._provenance:
            return
        devices = []
        for device_index in self._collector.devices or ():
            device = initialized_device_provenance(f"cuda:{device_index}")
            if device is not None:
                devices.append(device)
        if devices:
            self._provenance["devices"] = devices


def _point_manifest(point: MemoryPoint, root: Path) -> dict[str, object]:
    if point._snapshot_path is None:
        raise MemoryBundleError(f"point {point.label!r} is not persisted")
    return {
        "index": point.index,
        "label": point.label,
        "timestamp": point.timestamp,
        "metadata": _thaw_json(cast(FrozenJSONValue, point.metadata)),
        "boundary_marker": point.boundary_marker,
        "snapshot_file": str(point._snapshot_path.relative_to(root)),
        "warnings": list(point.warnings),
        "observations": [
            _observation_manifest(observation) for observation in point.observations
        ],
    }


def _observation_manifest(
    observation: MemoryObservation,
) -> dict[str, object]:
    return {
        "order": observation.order,
        "device_index": observation.device_index,
        "pool_id": list(observation.pool_id),
        "stream": observation.stream,
        **{name: getattr(observation.stats, name) for name in _MEMORY_STATS_FIELDS},
    }


def _observation_from_manifest(
    expected_order: int,
    row: Any,
) -> MemoryObservation:
    if not isinstance(row, Mapping):
        raise MemoryBundleError(
            f"memory observation {expected_order} must be a JSON object"
        )
    require_exact_fields(
        row,
        _OBSERVATION_FIELDS,
        f"memory observation {expected_order}",
        error_type=MemoryBundleError,
    )
    order = require_nonnegative_int(
        row["order"], "memory observation order", error_type=MemoryBundleError
    )
    if order != expected_order:
        raise MemoryBundleError(
            "memory observation order must be contiguous and ordered"
        )
    try:
        return MemoryObservation(
            order=order,
            device_index=require_nonnegative_int(
                row["device_index"],
                "memory observation device_index",
                error_type=MemoryBundleError,
            ),
            pool_id=normalize_pool_id(row["pool_id"]),
            stream=normalize_stream(row["stream"]),
            stats=MemoryStats.from_dict(row),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MemoryBundleError(
            f"invalid memory observation {expected_order}: {exc}"
        ) from exc


def _safe_bundle_path(root: Path, raw_path: Any, *, expected_index: int) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise MemoryBundleError("memory point is missing snapshot_file")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise MemoryBundleError("snapshot_file must be relative to the bundle")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MemoryBundleError(
            f"snapshot_file escapes the memory bundle: {raw_path!r}"
        ) from exc
    expected = f"snapshots/{expected_index:04d}.json.gz"
    if raw_path != expected:
        raise MemoryBundleError(f"snapshot_file must be {expected!r}")
    return resolved
