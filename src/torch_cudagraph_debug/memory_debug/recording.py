"""Memory snapshot collection, ownership, and persistence."""

from __future__ import annotations

import gzip
import json
import math
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from torch_cudagraph_debug._provenance import (
    initialized_device_provenance,
    runtime_provenance,
)

from ._pool_identity import PoolId
from ._collector import SnapshotProvider, SynchronizeTarget, _MemoryCollector
from .aggregation import summarize_allocator_scopes, summarize_pools
from .allocator_snapshot import (
    AllocatorSnapshotData,
    MemoryObservationKey,
    normalize_pool_id,
    normalize_stream,
)
from .errors import MemoryBundleError, MemoryDebugError, MemoryOwnershipError
from .stats import MemoryStats

if TYPE_CHECKING:
    from .attribution import MemoryAttributionOptions
    from .reports import (
        MemoryAllocationLifetimeAnalysis,
        MemoryPointComparison,
        MemoryTimeline,
    )

BUNDLE_SCHEMA = "torch-cudagraph-debug/memory-run"
_MEMORY_STATS_FIELDS = tuple(item.name for item in fields(MemoryStats))
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
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
_OBSERVATION_FIELDS = frozenset({"order", "pool_id", "stream", *_MEMORY_STATS_FIELDS})


@dataclass(frozen=True)
class MemoryObservation:
    """One pool/stream allocator state captured by a snapshot or point."""

    order: int
    pool_id: PoolId
    stream: Any
    stats: MemoryStats

    @property
    def key(self) -> MemoryObservationKey:
        return MemoryObservationKey(self.pool_id, self.stream)

    def descriptor(self) -> dict[str, object]:
        return {
            "order": self.order,
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
    metadata: Mapping[str, Any]
    boundary_marker: str
    observations: tuple[MemoryObservation, ...]
    warnings: tuple[str, ...] = ()
    _snapshot_path: Path | None = field(default=None, repr=False, compare=False)
    _snapshot_cache: dict[str, AllocatorSnapshotData] = field(
        default_factory=dict, repr=False, compare=False
    )
    _cache_snapshots: bool = field(default=True, repr=False, compare=False)

    @cached_property
    def by_key(self) -> Mapping[MemoryObservationKey, MemoryObservation]:
        return MappingProxyType({item.key: item for item in self.observations})

    @cached_property
    def observation_stats(self) -> Mapping[MemoryObservationKey, MemoryStats]:
        return MappingProxyType(
            {key: observation.stats for key, observation in self.by_key.items()}
        )

    @cached_property
    def pool_stats(self) -> Mapping[PoolId, MemoryStats]:
        return MappingProxyType(summarize_pools(self.observation_stats))

    @cached_property
    def allocator_scope_stats(self) -> Mapping[str, MemoryStats]:
        return MappingProxyType(summarize_allocator_scopes(self.pool_stats))

    def observation(self, pool_id: Any, stream: Any) -> MemoryObservation:
        key = MemoryObservationKey(normalize_pool_id(pool_id), normalize_stream(stream))
        try:
            return self.by_key[key]
        except KeyError as exc:
            raise KeyError(
                f"memory observation pool={key.pool_id!r} stream={key.stream!r} "
                f"does not exist at point {self.label!r}"
            ) from exc

    def raw_snapshot(self) -> AllocatorSnapshotData:
        """Load the full private PyTorch snapshot, lazily for persisted runs."""

        cached = self._snapshot_cache.get("snapshot") if self._cache_snapshots else None
        if cached is not None:
            return cached
        if self._snapshot_path is None:
            raise MemoryBundleError(f"point {self.label!r} has no snapshot payload")
        try:
            with gzip.open(self._snapshot_path, "rt", encoding="utf-8") as handle:
                snapshot = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryBundleError(
                f"could not load snapshot for point {self.label!r}: {exc}"
            ) from exc
        if not isinstance(snapshot, (Mapping, list)):
            raise MemoryBundleError(
                f"snapshot for point {self.label!r} has an invalid root type"
            )
        if self._cache_snapshots:
            self._snapshot_cache["snapshot"] = snapshot
        return snapshot

    def descriptor(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "index": self.index,
            "label": self.label,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
        }


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
    provenance: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    run_metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    bundle_dir: Path | None = field(default=None, compare=False)

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
            "provenance": dict(self.provenance),
            "run_metadata": dict(self.run_metadata),
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
        at: str | int | MemoryPoint | None = None,
        *,
        born_between: (
            tuple[str | int | MemoryPoint, str | int | MemoryPoint] | None
        ) = None,
        through: str | int | MemoryPoint | None = None,
        attribution: MemoryAttributionOptions | None = None,
    ) -> MemoryAllocationLifetimeAnalysis:
        """Analyze allocation cohorts across points in this run."""

        from .attribution import MemoryAttributionOptions
        from .lifetimes import analyze_allocation_lifetimes

        if not self.points:
            raise MemoryDebugError("allocation lifetimes require at least one point")
        if at is not None and born_between is not None:
            raise ValueError("at and born_between are mutually exclusive")
        active_at = self.point(at) if at is not None else None
        birth_range = (
            self.between(born_between[0], born_between[1])
            if born_between is not None
            else None
        )
        start = active_at or (birth_range.start if birth_range else self.points[0])
        end = self.point(through) if through is not None else self.points[-1]
        if end.index < start.index:
            raise ValueError("through point must not come before the lifetime anchor")
        if birth_range is not None and end.index < birth_range.end.index:
            raise ValueError("through point must not come before born_between end")
        options = attribution or MemoryAttributionOptions(
            stacks=True,
            events=True,
            lifetimes=True,
            on_missing="warn",
            stack_depth=4,
            limit=20,
        )
        return analyze_allocation_lifetimes(
            self,
            start=start,
            end=end,
            active_at=active_at,
            born_between=(
                (birth_range.start, birth_range.end) if birth_range else None
            ),
            options=options,
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
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryBundleError(
                f"could not read memory bundle {root}: {exc}"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise MemoryBundleError("memory bundle manifest must be a JSON object")
        if manifest.get("schema") != BUNDLE_SCHEMA:
            raise MemoryBundleError(
                f"unsupported memory bundle schema {manifest.get('schema')!r}"
            )
        _require_exact_fields(manifest, _MANIFEST_FIELDS, "memory bundle manifest")

        run_id = str(manifest["run_id"])
        name = str(manifest["name"])
        if not run_id or not name:
            raise MemoryBundleError("memory bundle is missing run_id or name")

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
            _require_exact_fields(
                raw,
                _POINT_FIELDS,
                f"memory point {expected_index}",
            )
            index = int(raw["index"])
            label = str(raw["label"])
            if index != expected_index:
                raise MemoryBundleError(
                    "memory bundle point indices must be contiguous and ordered"
                )
            if not label or label in labels:
                raise MemoryBundleError(
                    f"memory bundle has an empty or duplicate label {label!r}"
                )
            labels.add(label)
            snapshot_path = _safe_bundle_path(root, raw.get("snapshot_file"))
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
            metadata = raw["metadata"]
            if not isinstance(metadata, Mapping):
                raise MemoryBundleError(
                    f"metadata for point {label!r} must be a JSON object"
                )
            boundary_marker = raw["boundary_marker"]
            if not isinstance(boundary_marker, str):
                raise MemoryBundleError(
                    f"boundary_marker for point {label!r} must be a string"
                )
            warnings = raw["warnings"]
            if not isinstance(warnings, list) or not all(
                isinstance(item, str) for item in warnings
            ):
                raise MemoryBundleError(
                    f"warnings for point {label!r} must be a JSON string list"
                )
            points.append(
                MemoryPoint(
                    run_id=run_id,
                    index=index,
                    label=label,
                    timestamp=float(raw["timestamp"]),
                    metadata=MappingProxyType(dict(metadata)),
                    boundary_marker=boundary_marker,
                    observations=tuple(observations),
                    warnings=tuple(warnings),
                    _snapshot_path=snapshot_path,
                    _cache_snapshots=cache_snapshots,
                )
            )

        rank_value = manifest["rank"]
        rank = int(rank_value) if rank_value is not None else None
        group_value = manifest["group_id"]
        group_id = str(group_value) if group_value is not None else None
        world_size_value = manifest["world_size"]
        world_size = int(world_size_value) if world_size_value is not None else None
        provenance = manifest["provenance"]
        run_metadata = manifest["run_metadata"]
        if not isinstance(provenance, Mapping):
            raise MemoryBundleError("memory bundle provenance must be a JSON object")
        if not isinstance(run_metadata, Mapping):
            raise MemoryBundleError("memory bundle run_metadata must be a JSON object")
        finished_value = manifest["finished_at"]
        return cls(
            run_id=run_id,
            name=name,
            rank=rank,
            created_at=float(manifest["created_at"]),
            finished_at=(float(finished_value) if finished_value is not None else None),
            complete=bool(manifest["complete"]),
            points=tuple(points),
            group_id=group_id,
            world_size=world_size,
            provenance=MappingProxyType(dict(provenance)),
            run_metadata=MappingProxyType(dict(run_metadata)),
            bundle_dir=root,
        )


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
        synchronize: SynchronizeTarget = True,
        group_id: str | None = None,
        world_size: int | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        self.name = name
        self.bundle_dir = Path(bundle_dir).resolve() if bundle_dir is not None else None
        self.rank = _default_rank() if rank is None else int(rank)
        self.group_id = group_id
        if self.group_id is not None and not self.group_id:
            raise ValueError("group_id must be non-empty when provided")
        self.world_size = (
            _default_world_size() if world_size is None else int(world_size)
        )
        if self.world_size is not None and self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if (
            self.rank is not None
            and self.world_size is not None
            and not 0 <= self.rank < self.world_size
        ):
            raise ValueError("rank must be in [0, world_size)")
        serializable_metadata = _json_value(dict(run_metadata or {}), "$.run_metadata")
        assert isinstance(serializable_metadata, dict)
        serializable_provenance = _json_value(runtime_provenance(), "$.provenance")
        assert isinstance(serializable_provenance, dict)
        self.run_metadata = MappingProxyType(serializable_metadata)
        self._provenance: dict[str, Any] = serializable_provenance
        self.synchronize = synchronize
        self._collector = _MemoryCollector(synchronize=synchronize)
        self._run_id = uuid.uuid4().hex
        self._created_at = time.time()
        self._finished_at: float | None = None
        self._points: list[MemoryPoint] = []
        self._result: MemoryRun | None = None

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
        serializable_snapshot = _json_value(snapshot, "$.snapshot")
        if not isinstance(serializable_snapshot, (dict, list)):
            raise MemoryBundleError("allocator snapshot root must be an object or list")
        serializable_metadata = _json_value(dict(metadata or {}), "$.metadata")
        assert isinstance(serializable_metadata, dict)

        observations = tuple(
            MemoryObservation(
                order=order,
                pool_id=key.pool_id,
                stream=key.stream,
                stats=stats,
            )
            for order, (key, stats) in enumerate(capture.summaries)
        )
        snapshot_path: Path | None = None
        cache: dict[str, AllocatorSnapshotData] = {}
        if self.bundle_dir is not None:
            snapshot_path = self._write_snapshot(index, serializable_snapshot)
        else:
            cache["snapshot"] = serializable_snapshot

        point = MemoryPoint(
            run_id=self._run_id,
            index=index,
            label=label,
            timestamp=capture.timestamp,
            metadata=MappingProxyType(serializable_metadata),
            boundary_marker=capture.boundary_marker,
            observations=observations,
            warnings=capture.warnings,
            _snapshot_path=snapshot_path,
            _snapshot_cache=cache,
        )
        self._points.append(point)
        self._write_manifest(complete=False)
        return point

    def snapshot_run(self) -> MemoryRun:
        """Return an immutable view of points collected so far."""

        return self._build_run(complete=self._result is not None)

    def finish(self) -> MemoryRun:
        """Finish collection and return the immutable run; idempotent."""

        if self._result is not None:
            return self._result
        self._finished_at = time.time()
        self._result = self._build_run(complete=True)
        self._write_manifest(complete=True)
        return self._result

    def _build_run(self, *, complete: bool) -> MemoryRun:
        return MemoryRun(
            run_id=self._run_id,
            name=self.name,
            rank=self.rank,
            created_at=self._created_at,
            finished_at=self._finished_at,
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
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.finish()

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

    def _write_manifest(self, *, complete: bool) -> None:
        if self.bundle_dir is None:
            return
        payload = {
            "schema": BUNDLE_SCHEMA,
            "run_id": self._run_id,
            "name": self.name,
            "rank": self.rank,
            "group_id": self.group_id,
            "world_size": self.world_size,
            "created_at": self._created_at,
            "finished_at": self._finished_at,
            "complete": complete,
            "provenance": self._provenance,
            "run_metadata": dict(self.run_metadata),
            "points": [
                _point_manifest(point, self.bundle_dir) for point in self._points
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
        if self._collector.uses_snapshot_provider or "device" in self._provenance:
            return
        device = initialized_device_provenance()
        if device is not None:
            self._provenance["device"] = device


def _point_manifest(point: MemoryPoint, root: Path) -> dict[str, object]:
    if point._snapshot_path is None:
        raise MemoryBundleError(f"point {point.label!r} is not persisted")
    return {
        "index": point.index,
        "label": point.label,
        "timestamp": point.timestamp,
        "metadata": dict(point.metadata),
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
    _require_exact_fields(
        row,
        _OBSERVATION_FIELDS,
        f"memory observation {expected_order}",
    )
    order = int(row["order"])
    if order != expected_order:
        raise MemoryBundleError(
            "memory observation order must be contiguous and ordered"
        )
    return MemoryObservation(
        order=order,
        pool_id=normalize_pool_id(row["pool_id"]),
        stream=normalize_stream(row["stream"]),
        stats=MemoryStats.from_dict(row),
    )


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
    raise MemoryBundleError(f"{context} has invalid fields: {', '.join(details)}")


def _safe_bundle_path(root: Path, raw_path: Any) -> Path:
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
    if resolved.suffixes[-2:] != [".json", ".gz"]:
        raise MemoryBundleError(f"snapshot_file must end in .json.gz: {raw_path!r}")
    return resolved


def _json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MemoryBundleError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise MemoryBundleError(
                    f"{path} contains non-string mapping key {key!r}"
                )
            output[key] = _json_value(item, f"{path}.{key}")
        return output
    raise MemoryBundleError(f"{path} contains unsupported value {type(value).__name__}")


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
