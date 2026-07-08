"""Memory snapshot collection, ownership, and persistence."""

from __future__ import annotations

import gzip
import json
import math
import time
import uuid
import warnings
import zlib
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
from .allocator_snapshot import normalize_snapshot
from .errors import MemoryBundleError, MemoryDebugError, MemoryOwnershipError
from .events import (
    HISTORY_WINDOW_STATUSES,
    HistoryWindowStatus,
    _DeviceEventEvidence,
    _extract_point_event_evidence,
    _PointEventEvidence,
)
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
        "boundary_recorded",
        "state_file",
        "event_file",
        "history",
        "warnings",
        "observations",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {"order", "device_index", "pool_id", "stream", *_MEMORY_STATS_FIELDS}
)
_HISTORY_FIELDS = frozenset(
    {"device_index", "status", "warnings", "event_count", "trace_index_offset"}
)
_EVENT_FIELDS = frozenset({"start_index", "end_index", "devices"})
_EVENT_DEVICE_FIELDS = frozenset({"device_index", "trace_index_offset", "entries"})


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
class _DeviceHistoryStatus:
    device_index: int
    status: HistoryWindowStatus
    warnings: tuple[str, ...]
    event_count: int
    trace_index_offset: int


@dataclass(frozen=True)
class MemoryPoint:
    """One immutable labeled allocator state owned by a run."""

    run_id: str
    index: int
    label: str
    timestamp: float
    metadata: Mapping[str, FrozenJSONValue]
    boundary_marker: str
    observations: tuple[MemoryObservation, ...]
    warnings: tuple[str, ...] = ()
    _boundary_recorded: bool = field(default=True, repr=False, compare=False)
    _history_statuses: tuple[_DeviceHistoryStatus, ...] = field(
        default=(), repr=False, compare=False
    )
    _state_path: Path | None = field(default=None, repr=False, compare=False)
    _event_path: Path | None = field(default=None, repr=False, compare=False)
    _state_cache: dict[str, FrozenJSONValue] = field(
        default_factory=dict, repr=False, compare=False
    )
    _event_cache: dict[str, _PointEventEvidence] = field(
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
        if type(self._boundary_recorded) is not bool:
            raise TypeError("memory point boundary status must be boolean")
        if [item.order for item in self.observations] != list(
            range(len(self.observations))
        ):
            raise ValueError("memory observations must be contiguous and ordered")
        keys = [item.key for item in self.observations]
        if len(keys) != len(set(keys)):
            raise ValueError("memory observations must have unique keys")
        history_devices = [item.device_index for item in self._history_statuses]
        if len(history_devices) != len(set(history_devices)):
            raise ValueError("memory history statuses must have unique devices")
        if self.index == 0 and self._history_statuses:
            raise ValueError("the first memory point cannot own event history")
        object.__setattr__(
            self,
            "metadata",
            cast(Mapping[str, FrozenJSONValue], _freeze_json(self.metadata)),
        )
        for key, value in tuple(self._state_cache.items()):
            self._state_cache[key] = _freeze_json(
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
        state = self.allocator_state()
        settings = state.get("allocator_settings", MappingProxyType({}))
        if not isinstance(settings, Mapping):
            return MappingProxyType({})
        return cast(Mapping[str, FrozenJSONValue], settings)

    def allocator_state(self) -> Mapping[str, FrozenJSONValue]:
        """Load the recursively immutable allocator state at this point."""

        cached = self._state_cache.get("state") if self._cache_snapshots else None
        if cached is not None:
            return cast(Mapping[str, FrozenJSONValue], cached)
        if self._state_path is None:
            raise MemoryBundleError(f"point {self.label!r} has no state payload")
        state = _read_gzip_json(
            self._state_path,
            context=f"state for point {self.label!r}",
        )
        if not isinstance(state, Mapping):
            raise MemoryBundleError(
                f"state for point {self.label!r} must be a JSON object"
            )
        if "device_traces" in state:
            raise MemoryBundleError(
                f"state for point {self.label!r} must not contain device_traces"
            )
        try:
            normalize_snapshot(state)
        except (TypeError, ValueError) as exc:
            raise MemoryBundleError(
                f"state for point {self.label!r} has invalid allocator schema: {exc}"
            ) from exc
        frozen = _freeze_json(cast(JSONValue, state))
        if self._cache_snapshots:
            self._state_cache["state"] = frozen
        return cast(Mapping[str, FrozenJSONValue], frozen)

    def _event_evidence(self) -> _PointEventEvidence | None:
        if self.index == 0:
            return None
        cached = self._event_cache.get("event")
        if cached is not None:
            return cached
        if self._event_path is None:
            raise MemoryBundleError(f"point {self.label!r} has no event payload")
        payload = _read_gzip_json(
            self._event_path,
            context=f"event evidence for point {self.label!r}",
        )
        evidence = _event_evidence_from_payload(payload, point=self)
        if self._cache_snapshots:
            self._event_cache["event"] = evidence
        return evidence

    def _event_windows(self) -> tuple[Any, ...]:
        evidence = self._event_evidence()
        if evidence is None:
            return ()
        try:
            return evidence.windows()
        except (TypeError, ValueError) as exc:
            if self._event_path is not None:
                raise MemoryBundleError(
                    f"event evidence for point {self.label!r} has invalid allocator schema: {exc}"
                ) from exc
            raise

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
            state_path = _safe_bundle_payload_path(
                root,
                raw["state_file"],
                field="state_file",
                expected=f"states/{expected_index:04d}.json.gz",
            )
            if not state_path.is_file():
                raise MemoryBundleError(
                    f"state file for point {label!r} does not exist"
                )
            if expected_index == 0:
                if raw["event_file"] is not None or raw["history"] is not None:
                    raise MemoryBundleError(
                        "the first memory point must not have event evidence"
                    )
                event_path = None
                history_statuses: tuple[_DeviceHistoryStatus, ...] = ()
            else:
                event_path = _safe_bundle_payload_path(
                    root,
                    raw["event_file"],
                    field="event_file",
                    expected=f"events/{expected_index - 1:04d}-{expected_index:04d}.json.gz",
                )
                if not event_path.is_file():
                    raise MemoryBundleError(
                        f"event file for point {label!r} does not exist"
                    )
                history_statuses = _history_statuses_from_manifest(
                    raw["history"], point_label=label
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
                    _boundary_recorded=require_bool(
                        raw["boundary_recorded"],
                        f"boundary_recorded for point {label!r}",
                        error_type=MemoryBundleError,
                    ),
                    _history_statuses=history_statuses,
                    _state_path=state_path,
                    _event_path=event_path,
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
            (self.bundle_dir / "states").mkdir(parents=True, exist_ok=True)
            (self.bundle_dir / "events").mkdir(parents=True, exist_ok=True)
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
        if not isinstance(serializable_snapshot, dict):
            raise MemoryBundleError("allocator snapshot root must be an object")
        state_payload = dict(serializable_snapshot)
        state_payload.pop("device_traces", None)
        serializable_metadata = json_value(
            dict(metadata or {}), "$.metadata", error_type=MemoryBundleError
        )
        assert isinstance(serializable_metadata, dict)

        event_evidence: _PointEventEvidence | None = None
        history_statuses: tuple[_DeviceHistoryStatus, ...] = ()
        if self._points:
            previous = self._points[-1]
            event_evidence = _extract_point_event_evidence(
                snapshot,
                devices=capture.devices,
                previous_boundary_recorded=previous._boundary_recorded,
                current_boundary_recorded=capture.boundary_recorded,
                start_marker=previous.boundary_marker,
                end_marker=capture.boundary_marker,
                start_label=previous.label,
                end_label=label,
                start_index=previous.index,
                end_index=index,
            )
            history_statuses = tuple(
                _DeviceHistoryStatus(
                    device_index=device.device_index,
                    status=device.status,
                    warnings=device.warnings,
                    event_count=device.event_count,
                    trace_index_offset=device.trace_index_offset,
                )
                for device in event_evidence.devices
            )

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
        state_path: Path | None = None
        event_path: Path | None = None
        state_cache: dict[str, FrozenJSONValue] = {}
        event_cache: dict[str, _PointEventEvidence] = {}
        if self.bundle_dir is not None:
            state_path = self._write_state(index, state_payload)
            if event_evidence is not None:
                event_path = self._write_event(event_evidence)
        else:
            state_cache["state"] = _freeze_json(state_payload)
            if event_evidence is not None:
                event_cache["event"] = event_evidence
        del snapshot, serializable_snapshot

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
            _boundary_recorded=capture.boundary_recorded,
            _history_statuses=history_statuses,
            _state_path=state_path,
            _event_path=event_path,
            _state_cache=state_cache,
            _event_cache=event_cache,
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

    def _write_state(self, index: int, state: Mapping[str, Any]) -> Path:
        assert self.bundle_dir is not None
        path = self.bundle_dir / "states" / f"{index:04d}.json.gz"
        self._write_gzip_payload(path, state, context=f"allocator state {index}")
        return path

    def _write_event(self, evidence: _PointEventEvidence) -> Path:
        assert self.bundle_dir is not None
        path = (
            self.bundle_dir
            / "events"
            / f"{evidence.start_index:04d}-{evidence.end_index:04d}.json.gz"
        )
        self._write_gzip_payload(
            path,
            _event_evidence_payload(evidence),
            context=(
                f"allocator event evidence {evidence.start_index}-{evidence.end_index}"
            ),
        )
        return path

    @staticmethod
    def _write_gzip_payload(
        path: Path, payload: Mapping[str, Any], *, context: str
    ) -> None:
        temporary = path.with_name(path.name + ".tmp")
        try:
            with gzip.open(
                temporary,
                "wt",
                encoding="utf-8",
                compresslevel=1,
            ) as handle:
                json.dump(
                    payload,
                    handle,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            temporary.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            raise MemoryBundleError(f"could not persist {context}: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)

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
    if point._state_path is None:
        raise MemoryBundleError(f"point {point.label!r} is not persisted")
    if point.index > 0 and point._event_path is None:
        raise MemoryBundleError(
            f"point {point.label!r} has no persisted event evidence"
        )
    return {
        "index": point.index,
        "label": point.label,
        "timestamp": point.timestamp,
        "metadata": _thaw_json(cast(FrozenJSONValue, point.metadata)),
        "boundary_marker": point.boundary_marker,
        "boundary_recorded": point._boundary_recorded,
        "state_file": str(point._state_path.relative_to(root)),
        "event_file": (
            str(point._event_path.relative_to(root))
            if point._event_path is not None
            else None
        ),
        "history": (
            [
                {
                    "device_index": item.device_index,
                    "status": item.status,
                    "warnings": list(item.warnings),
                    "event_count": item.event_count,
                    "trace_index_offset": item.trace_index_offset,
                }
                for item in point._history_statuses
            ]
            if point.index > 0
            else None
        ),
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


def _history_statuses_from_manifest(
    raw_history: Any, *, point_label: str
) -> tuple[_DeviceHistoryStatus, ...]:
    if not isinstance(raw_history, list):
        raise MemoryBundleError(
            f"history for point {point_label!r} must be a JSON list"
        )
    statuses = []
    previous_device = -1
    for index, row in enumerate(raw_history):
        if not isinstance(row, Mapping):
            raise MemoryBundleError(
                f"history entry {index} for point {point_label!r} must be an object"
            )
        require_exact_fields(
            row,
            _HISTORY_FIELDS,
            f"history entry {index} for point {point_label!r}",
            error_type=MemoryBundleError,
        )
        device_index = require_nonnegative_int(
            row["device_index"],
            f"history device_index for point {point_label!r}",
            error_type=MemoryBundleError,
        )
        if device_index <= previous_device:
            raise MemoryBundleError(
                f"history devices for point {point_label!r} must be unique and ordered"
            )
        previous_device = device_index
        status = row["status"]
        if not isinstance(status, str) or status not in HISTORY_WINDOW_STATUSES:
            raise MemoryBundleError(
                f"invalid history status for point {point_label!r}: {status!r}"
            )
        raw_warnings = row["warnings"]
        if not isinstance(raw_warnings, list) or not all(
            isinstance(item, str) for item in raw_warnings
        ):
            raise MemoryBundleError(
                f"history warnings for point {point_label!r} must be strings"
            )
        event_count = require_nonnegative_int(
            row["event_count"],
            f"history event_count for point {point_label!r}",
            error_type=MemoryBundleError,
        )
        trace_index_offset = require_nonnegative_int(
            row["trace_index_offset"],
            f"history trace_index_offset for point {point_label!r}",
            error_type=MemoryBundleError,
        )
        if status != "complete" and event_count:
            raise MemoryBundleError(
                f"incomplete history for point {point_label!r} cannot contain events"
            )
        statuses.append(
            _DeviceHistoryStatus(
                device_index=device_index,
                status=cast(HistoryWindowStatus, status),
                warnings=tuple(raw_warnings),
                event_count=event_count,
                trace_index_offset=trace_index_offset,
            )
        )
    return tuple(statuses)


def _event_evidence_payload(evidence: _PointEventEvidence) -> dict[str, object]:
    return {
        "start_index": evidence.start_index,
        "end_index": evidence.end_index,
        "devices": [
            {
                "device_index": device.device_index,
                "trace_index_offset": device.trace_index_offset,
                "entries": [dict(entry) for entry in device.entries],
            }
            for device in evidence.devices
        ],
    }


def _event_evidence_from_payload(
    payload: Any, *, point: MemoryPoint
) -> _PointEventEvidence:
    if not isinstance(payload, Mapping):
        raise MemoryBundleError(
            f"event evidence for point {point.label!r} must be a JSON object"
        )
    require_exact_fields(
        payload,
        _EVENT_FIELDS,
        f"event evidence for point {point.label!r}",
        error_type=MemoryBundleError,
    )
    start_index = require_nonnegative_int(
        payload["start_index"], "event start_index", error_type=MemoryBundleError
    )
    end_index = require_nonnegative_int(
        payload["end_index"], "event end_index", error_type=MemoryBundleError
    )
    if start_index != point.index - 1 or end_index != point.index:
        raise MemoryBundleError(
            f"event evidence indices do not match point {point.label!r}"
        )
    raw_devices = payload["devices"]
    if not isinstance(raw_devices, list) or len(raw_devices) != len(
        point._history_statuses
    ):
        raise MemoryBundleError(
            f"event devices do not match history for point {point.label!r}"
        )
    devices = []
    for index, (row, status) in enumerate(zip(raw_devices, point._history_statuses)):
        if not isinstance(row, Mapping):
            raise MemoryBundleError(
                f"event device {index} for point {point.label!r} must be an object"
            )
        require_exact_fields(
            row,
            _EVENT_DEVICE_FIELDS,
            f"event device {index} for point {point.label!r}",
            error_type=MemoryBundleError,
        )
        device_index = require_nonnegative_int(
            row["device_index"], "event device_index", error_type=MemoryBundleError
        )
        trace_index_offset = require_nonnegative_int(
            row["trace_index_offset"],
            "event trace_index_offset",
            error_type=MemoryBundleError,
        )
        raw_entries = row["entries"]
        if not isinstance(raw_entries, list) or not all(
            isinstance(entry, Mapping) for entry in raw_entries
        ):
            raise MemoryBundleError(
                f"event entries for point {point.label!r} must be objects"
            )
        if (
            device_index != status.device_index
            or trace_index_offset != status.trace_index_offset
            or len(raw_entries) != status.event_count
        ):
            raise MemoryBundleError(
                f"event payload does not match history for point {point.label!r}"
            )
        devices.append(
            _DeviceEventEvidence(
                device_index=device_index,
                status=status.status,
                warnings=status.warnings,
                entries=tuple(dict(entry) for entry in raw_entries),
                trace_index_offset=trace_index_offset,
            )
        )
    return _PointEventEvidence(
        start_index=start_index,
        end_index=end_index,
        devices=tuple(devices),
    )


def _read_gzip_json(path: Path, *, context: str) -> Any:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            text = handle.read()
    except (OSError, EOFError, zlib.error, UnicodeDecodeError) as exc:
        raise MemoryBundleError(f"could not load {context}: {exc}") from exc
    return strict_json_loads(
        text,
        error_type=MemoryBundleError,
        context=context,
    )


def _safe_bundle_payload_path(
    root: Path, raw_path: Any, *, field: str, expected: str
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise MemoryBundleError(f"memory point is missing {field}")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise MemoryBundleError(f"{field} must be relative to the bundle")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MemoryBundleError(
            f"{field} escapes the memory bundle: {raw_path!r}"
        ) from exc
    if raw_path != expected:
        raise MemoryBundleError(f"{field} must be {expected!r}")
    return resolved
