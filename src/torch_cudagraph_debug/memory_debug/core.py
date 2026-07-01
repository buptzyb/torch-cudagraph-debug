"""Memory snapshot collection, ownership, persistence, and comparison."""

from __future__ import annotations

import gzip
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .errors import (
    MemoryBundleError,
    MemoryDebugError,
    MemoryHistoryError,
    MemoryOwnershipError,
)
from .events import (
    extract_event_window,
    extract_event_window_from_snapshot,
    summarize_allocator_events,
)
from .lifetimes import analyze_allocation_lifetimes
from .models import (
    LifecycleDelta,
    MemoryStats,
    MemoryStatsDelta,
    PointAllocatorScopeState,
    PointPoolState,
    PointPoolStreamState,
    PoolComparison,
    PoolId,
    PoolStreamComparison,
)
from .provenance import initialized_device_provenance, runtime_provenance
from .reports import (
    AllocationLifetimeReport,
    MemoryComparison,
    MemoryTimeline,
    PhaseComparison,
)
from .stacks import (
    AllocationStackCoverage,
    AllocationStackDelta,
    _AllocationStackIndex,
    _build_allocation_stack_index,
    _compare_mapped_stack_indexes,
    _compare_stack_indexes,
)
from .summary import (
    DEFAULT_POOL_ID,
    GroupKey,
    SnapshotInput,
    compare_lifecycle,
    normalize_pool_id,
    normalize_snapshot,
    normalize_trace_entries,
    pool_id_label,
    stream_label,
    summarize_segments,
    trace_device_indices,
)
from .totals import (
    ALLOCATOR_SCOPES,
    compare_allocator_scopes,
    summarize_allocator_scopes,
    summarize_pools,
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
        "groups",
    }
)
_GROUP_FIELDS = frozenset({"pool_id", "stream", *_MEMORY_STATS_FIELDS})
SnapshotProvider = Callable[[str], SnapshotInput]
MissingPolicy = Literal["warn", "error"]


@dataclass(frozen=True)
class AttributionOptions:
    """Optional allocator-history attribution for memory analysis."""

    stacks: bool = False
    events: bool = False
    lifetimes: bool = False
    on_missing: MissingPolicy = "warn"
    stack_depth: int = 2
    limit: int = 20

    def __post_init__(self) -> None:
        if self.on_missing not in {"warn", "error"}:
            raise ValueError("on_missing must be 'warn' or 'error'")
        if self.stack_depth < 1:
            raise ValueError("stack_depth must be >= 1")
        if self.limit < 1:
            raise ValueError("limit must be >= 1")


@dataclass(frozen=True)
class MemoryPoint:
    """One immutable labeled allocator snapshot owned by a run."""

    run_id: str
    index: int
    label: str
    timestamp: float
    metadata: Mapping[str, Any]
    boundary_marker: str
    groups: Mapping[GroupKey, MemoryStats]
    warnings: tuple[str, ...] = ()
    _snapshot_path: Path | None = field(default=None, repr=False, compare=False)
    _snapshot_cache: dict[str, SnapshotInput] = field(
        default_factory=dict, repr=False, compare=False
    )
    _cache_snapshots: bool = field(default=True, repr=False, compare=False)

    @cached_property
    def pools(self) -> Mapping[PoolId, MemoryStats]:
        return MappingProxyType(summarize_pools(self.groups))

    @cached_property
    def totals(self) -> Mapping[str, MemoryStats]:
        return MappingProxyType(summarize_allocator_scopes(self.pools))

    def raw_snapshot(self) -> SnapshotInput:
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
        self, before: str | int | MemoryPoint, after: str | int | MemoryPoint
    ) -> "MemoryRange":
        start = self.point(before)
        end = self.point(after)
        if end.index <= start.index:
            raise ValueError("after point must come after before point")
        return MemoryRange(self, start, end)

    def compare(
        self,
        before: str | int | MemoryPoint,
        after: str | int | MemoryPoint,
        *,
        attribution: AttributionOptions | None = None,
    ) -> MemoryComparison:
        memory_range = self.between(before, after)
        return _compare_same_run(
            self,
            memory_range.before,
            memory_range.after,
            attribution or AttributionOptions(),
        )

    def timeline(
        self,
        *,
        attribution: AttributionOptions | None = None,
    ) -> MemoryTimeline:
        return _build_timeline(self, attribution or AttributionOptions())

    def lifetimes(
        self,
        at: str | int | MemoryPoint | None = None,
        *,
        born_between: (
            tuple[str | int | MemoryPoint, str | int | MemoryPoint] | None
        ) = None,
        through: str | int | MemoryPoint | None = None,
        attribution: AttributionOptions | None = None,
    ) -> AllocationLifetimeReport:
        """Analyze allocation cohorts across points in this run."""

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
        start = active_at or (birth_range.before if birth_range else self.points[0])
        end = self.point(through) if through is not None else self.points[-1]
        if end.index < start.index:
            raise ValueError("through point must not come before the lifetime anchor")
        if birth_range is not None and end.index < birth_range.after.index:
            raise ValueError("through point must not come before born_between end")
        options = attribution or AttributionOptions(
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
                (birth_range.before, birth_range.after) if birth_range else None
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
            groups: dict[GroupKey, MemoryStats] = {}
            raw_groups = raw["groups"]
            if not isinstance(raw_groups, list):
                raise MemoryBundleError(
                    f"groups for point {label!r} must be a JSON list"
                )
            for group_row in raw_groups:
                if not isinstance(group_row, Mapping):
                    raise MemoryBundleError(
                        f"group summary for point {label!r} is invalid"
                    )
                _require_exact_fields(
                    group_row,
                    _GROUP_FIELDS,
                    f"group summary for point {label!r}",
                )
                key, stats = _group_from_manifest(group_row)
                if key in groups:
                    raise MemoryBundleError(
                        f"point {label!r} has duplicate group {key!r}"
                    )
                groups[key] = stats
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
                    groups=MappingProxyType(groups),
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
    before: MemoryPoint
    after: MemoryPoint

    def compare(
        self, *, attribution: AttributionOptions | None = None
    ) -> MemoryComparison:
        return self.run.compare(
            self.before,
            self.after,
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
        synchronize: bool = True,
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
        self.synchronize = bool(synchronize)
        self._run_id = uuid.uuid4().hex
        self._created_at = time.time()
        self._finished_at: float | None = None
        self._points: list[MemoryPoint] = []
        self._snapshot_provider: SnapshotProvider | None = None
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
        recorder._snapshot_provider = provider
        return recorder

    def mark(
        self,
        label: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryPoint:
        if self._result is not None:
            raise MemoryDebugError("cannot mark a finished memory recorder")
        if not label:
            raise ValueError("memory point label must be non-empty")
        if any(point.label == label for point in self._points):
            raise ValueError(f"memory point label {label!r} already exists")

        index = len(self._points)
        marker = f"torch-cudagraph-debug:memory-point:{self._run_id}:{index}:{label}"
        warnings: list[str] = []
        snapshot = _snapshot_envelope(self._capture_snapshot(marker, warnings))
        self._record_device_provenance()
        serializable_snapshot = _json_value(snapshot, "$.snapshot")
        if not isinstance(serializable_snapshot, (dict, list)):
            raise MemoryBundleError("allocator snapshot root must be an object or list")
        serializable_metadata = _json_value(dict(metadata or {}), "$.metadata")
        assert isinstance(serializable_metadata, dict)

        segments = normalize_snapshot(serializable_snapshot)
        groups = summarize_segments(segments)
        snapshot_path: Path | None = None
        cache: dict[str, SnapshotInput] = {}
        if self.bundle_dir is not None:
            snapshot_path = self._write_snapshot(index, serializable_snapshot)
        else:
            cache["snapshot"] = serializable_snapshot

        point = MemoryPoint(
            run_id=self._run_id,
            index=index,
            label=label,
            timestamp=time.time(),
            metadata=MappingProxyType(serializable_metadata),
            boundary_marker=marker,
            groups=MappingProxyType(dict(groups)),
            warnings=tuple(warnings),
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

    def _capture_snapshot(self, marker: str, warnings: list[str]) -> SnapshotInput:
        if self._snapshot_provider is not None:
            return self._snapshot_provider(marker)

        import torch

        capturing = _current_stream_is_capturing(torch)
        if self.synchronize and torch.cuda.is_available() and not capturing:
            torch.cuda.synchronize()

        memory = torch.cuda.memory
        get_metadata = getattr(memory, "_get_memory_metadata", None)
        set_metadata = getattr(memory, "_set_memory_metadata", None)
        previous_metadata: str | None = None
        marker_set = False
        if callable(get_metadata) and callable(set_metadata):
            try:
                previous_metadata = get_metadata()
                set_metadata(marker)
                marker_set = True
            except Exception as exc:
                warnings.append(
                    "could not set allocator metadata marker: "
                    f"{type(exc).__name__}: {exc}"
                )
        else:
            warnings.append(
                "PyTorch memory metadata APIs are unavailable; "
                "event boundaries cannot be marked"
            )
        try:
            return memory._snapshot()
        finally:
            if marker_set and callable(set_metadata):
                try:
                    set_metadata(previous_metadata or "")
                except Exception as exc:
                    warnings.append(
                        "could not restore allocator metadata: "
                        f"{type(exc).__name__}: {exc}"
                    )

    def _write_snapshot(self, index: int, snapshot: SnapshotInput) -> Path:
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
        if self._snapshot_provider is not None or "device" in self._provenance:
            return
        device = initialized_device_provenance()
        if device is not None:
            self._provenance["device"] = device


@dataclass(frozen=True)
class _PointSnapshotView:
    point: MemoryPoint
    raw: SnapshotInput
    segments: tuple[Mapping[str, Any], ...]
    stacks: _AllocationStackIndex | None


def _load_point_snapshot(
    point: MemoryPoint,
    options: AttributionOptions,
) -> _PointSnapshotView:
    raw = point.raw_snapshot()
    segments = normalize_snapshot(raw)
    return _PointSnapshotView(
        point=point,
        raw=raw,
        segments=segments,
        stacks=(
            _build_allocation_stack_index(
                segments,
                stack_depth=options.stack_depth,
            )
            if options.stacks
            else None
        ),
    )


def compare_points(
    before: MemoryPoint,
    after: MemoryPoint,
    *,
    pool_mapping: Mapping[Any, Any] | None = None,
    attribution: AttributionOptions | None = None,
) -> MemoryComparison:
    """Compare points from independent runs without address/event identity."""

    return _compare_points_impl(
        before,
        after,
        pool_mapping=pool_mapping,
        options=attribution or AttributionOptions(),
    )


def _compare_points_impl(
    before: MemoryPoint,
    after: MemoryPoint,
    *,
    pool_mapping: Mapping[Any, Any] | None,
    options: AttributionOptions,
    before_stack_index: _AllocationStackIndex | None = None,
    after_stack_index: _AllocationStackIndex | None = None,
) -> MemoryComparison:
    if before.run_id == after.run_id:
        raise MemoryOwnershipError(
            "compare_points requires independent runs; use MemoryRun.compare "
            "for points from the same run"
        )
    if options.events:
        raise MemoryDebugError(
            "allocator events cannot be compared across independent runs"
        )
    if options.lifetimes:
        raise MemoryDebugError(
            "allocation lifetimes cannot be compared across independent runs"
        )

    before_pools = before.pools
    after_pools = after.pools
    mapping = _validate_pool_mapping(before_pools, after_pools, pool_mapping)
    pools: list[PoolComparison] = []
    matched_before: set[PoolId] = set()
    matched_after: set[PoolId] = set()
    for before_pool, (after_pool, match) in mapping.items():
        matched_before.add(before_pool)
        matched_after.add(after_pool)
        before_stats = before_pools[before_pool]
        after_stats = after_pools[after_pool]
        pools.append(
            PoolComparison(
                before_pool_id=before_pool,
                after_pool_id=after_pool,
                match=match,
                before=before_stats,
                after=after_stats,
                delta=MemoryStatsDelta.between(before_stats, after_stats),
                lifecycle=None,
            )
        )
    for pool_id in set(before_pools) - matched_before:
        before_stats = before_pools[pool_id]
        after_stats = MemoryStats()
        pools.append(
            PoolComparison(
                before_pool_id=pool_id,
                after_pool_id=None,
                match="before_only",
                before=before_stats,
                after=after_stats,
                delta=MemoryStatsDelta.between(before_stats, after_stats),
                lifecycle=None,
            )
        )
    for pool_id in set(after_pools) - matched_after:
        before_stats = MemoryStats()
        after_stats = after_pools[pool_id]
        pools.append(
            PoolComparison(
                before_pool_id=None,
                after_pool_id=pool_id,
                match="after_only",
                before=before_stats,
                after=after_stats,
                delta=MemoryStatsDelta.between(before_stats, after_stats),
                lifecycle=None,
            )
        )

    pool_streams = _unmatched_cross_run_groups(before, after)
    warnings = [*before.warnings, *after.warnings]
    unmatched_private = (set(before_pools) - matched_before - {DEFAULT_POOL_ID}) or (
        set(after_pools) - matched_after - {DEFAULT_POOL_ID}
    )
    if unmatched_private:
        warnings.append(
            "private pools are unmatched across runs unless pool_mapping "
            "explicitly pairs them"
        )

    before_coverage: AllocationStackCoverage | None = None
    after_coverage: AllocationStackCoverage | None = None
    stack_deltas: tuple[AllocationStackDelta, ...] = ()
    if options.stacks:
        before_index = before_stack_index or _build_allocation_stack_index(
            normalize_snapshot(before.raw_snapshot()),
            stack_depth=options.stack_depth,
        )
        after_index = after_stack_index or _build_allocation_stack_index(
            normalize_snapshot(after.raw_snapshot()),
            stack_depth=options.stack_depth,
        )
        before_coverage = before_index.coverage
        after_coverage = after_index.coverage
        _check_stack_coverage(
            before_coverage,
            after_coverage,
            options,
            warnings,
        )
        stack_deltas = _compare_mapped_stack_indexes(
            before_index,
            after_index,
            mapping,
            top=options.limit,
        )
        if unmatched_private:
            warnings.append(
                "allocation stacks for unmatched private pools are omitted "
                "from cross-run stack deltas"
            )

    return MemoryComparison(
        before=before,
        after=after,
        totals=compare_allocator_scopes(
            summarize_allocator_scopes(before_pools),
            summarize_allocator_scopes(after_pools),
        ),
        pools=tuple(sorted(pools, key=_pool_comparison_sort_key)),
        pool_streams=pool_streams,
        allocation_stacks=stack_deltas,
        before_stack_coverage=before_coverage,
        after_stack_coverage=after_coverage,
        lifecycle_available=False,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def compare_phases(
    baseline: MemoryRange,
    candidate: MemoryRange,
    *,
    pool_mapping: Mapping[Any, Any] | None = None,
    attribution: AttributionOptions | None = None,
) -> PhaseComparison:
    """Decompose candidate-vs-baseline phase memory using four points."""

    options = attribution or AttributionOptions()
    baseline_before_view = _load_point_snapshot(baseline.before, options)
    baseline_after_view = _load_point_snapshot(baseline.after, options)
    baseline_growth = _compare_same_run_views(
        baseline.run,
        baseline_before_view,
        baseline_after_view,
        options,
    )
    baseline_before_stacks = baseline_before_view.stacks
    baseline_after_stacks = baseline_after_view.stacks
    del baseline_before_view, baseline_after_view

    candidate_before_view = _load_point_snapshot(candidate.before, options)
    candidate_after_view = _load_point_snapshot(candidate.after, options)
    candidate_growth = _compare_same_run_views(
        candidate.run,
        candidate_before_view,
        candidate_after_view,
        options,
    )
    candidate_before_stacks = candidate_before_view.stacks
    candidate_after_stacks = candidate_after_view.stacks
    del candidate_before_view, candidate_after_view

    cross_options = AttributionOptions(
        stacks=options.stacks,
        events=False,
        lifetimes=False,
        on_missing=options.on_missing,
        stack_depth=options.stack_depth,
        limit=options.limit,
    )
    start_delta = _compare_points_impl(
        baseline.before,
        candidate.before,
        pool_mapping=pool_mapping,
        options=cross_options,
        before_stack_index=baseline_before_stacks,
        after_stack_index=candidate_before_stacks,
    )
    end_delta = _compare_points_impl(
        baseline.after,
        candidate.after,
        pool_mapping=pool_mapping,
        options=cross_options,
        before_stack_index=baseline_after_stacks,
        after_stack_index=candidate_after_stacks,
    )
    decomposition = _phase_decomposition(
        baseline_growth,
        candidate_growth,
        start_delta,
        end_delta,
    )
    total_decomposition = _scope_phase_decomposition(
        baseline_growth,
        candidate_growth,
        start_delta,
        end_delta,
    )
    return PhaseComparison(
        baseline_name=baseline.run.name,
        candidate_name=candidate.run.name,
        baseline_growth=baseline_growth,
        candidate_growth=candidate_growth,
        start_delta=start_delta,
        end_delta=end_delta,
        total_decomposition=total_decomposition,
        decomposition=decomposition,
    )


def _compare_same_run(
    run: MemoryRun,
    before: MemoryPoint,
    after: MemoryPoint,
    options: AttributionOptions,
) -> MemoryComparison:
    return _compare_same_run_views(
        run,
        _load_point_snapshot(before, options),
        _load_point_snapshot(after, options),
        options,
    )


def _compare_same_run_views(
    run: MemoryRun,
    before_view: _PointSnapshotView,
    after_view: _PointSnapshotView,
    options: AttributionOptions,
) -> MemoryComparison:
    before = before_view.point
    after = after_view.point
    after_snapshot = after_view.raw
    before_segments = before_view.segments
    after_segments = after_view.segments
    before_pools = before.pools
    after_pools = after.pools

    group_lifecycle = compare_lifecycle(before_segments, after_segments)
    lifecycle_by_pool: dict[PoolId, list[LifecycleDelta]] = {}
    for key, lifecycle in group_lifecycle.items():
        lifecycle_by_pool.setdefault(key.pool_id, []).append(lifecycle)
    pool_lifecycle = {
        pool_id: LifecycleDelta.combine(values)
        for pool_id, values in lifecycle_by_pool.items()
    }

    pools = tuple(
        PoolComparison(
            before_pool_id=pool_id,
            after_pool_id=pool_id,
            match="same_run",
            before=before_pools.get(pool_id, MemoryStats()),
            after=after_pools.get(pool_id, MemoryStats()),
            delta=MemoryStatsDelta.between(
                before_pools.get(pool_id, MemoryStats()),
                after_pools.get(pool_id, MemoryStats()),
            ),
            lifecycle=pool_lifecycle.get(pool_id, LifecycleDelta()),
        )
        for pool_id in sorted(set(before_pools) | set(after_pools), key=pool_id_label)
    )
    pool_streams = tuple(
        PoolStreamComparison(
            before_pool_id=key.pool_id,
            before_stream=key.stream,
            after_pool_id=key.pool_id,
            after_stream=key.stream,
            match="same_run",
            before=before.groups.get(key, MemoryStats()),
            after=after.groups.get(key, MemoryStats()),
            delta=MemoryStatsDelta.between(
                before.groups.get(key, MemoryStats()),
                after.groups.get(key, MemoryStats()),
            ),
            lifecycle=group_lifecycle.get(key, LifecycleDelta()),
        )
        for key in sorted(
            set(before.groups) | set(after.groups),
            key=lambda item: (pool_id_label(item.pool_id), stream_label(item.stream)),
        )
    )

    warnings = [*before.warnings, *after.warnings]
    before_coverage: AllocationStackCoverage | None = None
    after_coverage: AllocationStackCoverage | None = None
    stack_deltas: tuple[AllocationStackDelta, ...] = ()
    stack_detail_deltas: tuple[AllocationStackDelta, ...] = ()
    if options.stacks:
        before_index = before_view.stacks
        after_index = after_view.stacks
        assert before_index is not None and after_index is not None
        before_coverage = before_index.coverage
        after_coverage = after_index.coverage
        _check_stack_coverage(
            before_coverage,
            after_coverage,
            options,
            warnings,
        )
        stack_deltas = _compare_stack_indexes(
            before_index,
            after_index,
            by_stream=False,
            top=options.limit,
        )
        stack_detail_deltas = _compare_stack_indexes(
            before_index,
            after_index,
            by_stream=True,
            top=options.limit,
        )

    allocator_events = ()
    events_available = False
    events_complete = True
    if options.events:
        entries = []
        events_available = True
        event_devices = _segment_device_indices(before_segments, after_segments)
        selected = run.points[before.index : after.index + 1]
        for previous, current in zip(selected, selected[1:]):
            current_snapshot = (
                after_snapshot if current.index == after.index else current.raw_snapshot()
            )
            interval_devices = tuple(
                sorted(set(event_devices) | set(trace_device_indices(current_snapshot)))
            )
            if interval_devices:
                windows = tuple(
                    extract_event_window_from_snapshot(
                        current_snapshot,
                        device_index=device,
                        start_marker=previous.boundary_marker,
                        end_marker=current.boundary_marker,
                        start_label=f"{previous.label} on device {device}",
                    )
                    for device in interval_devices
                )
            else:
                windows = (
                    extract_event_window(
                        normalize_trace_entries(current_snapshot),
                        start_marker=previous.boundary_marker,
                        end_marker=current.boundary_marker,
                        start_label=previous.label,
                    ),
                )
            for window in windows:
                entries.extend(window.entries)
                events_available = events_available and window.available
                events_complete = events_complete and window.complete
                warnings.extend(window.warnings)
        if not events_available or not events_complete:
            _missing(
                "allocator event history is unavailable or incomplete; "
                "enable torch.cuda.memory._record_memory_history(...) "
                "before the phase",
                options,
                warnings,
            )
        allocator_events = summarize_allocator_events(
            entries,
            before_segments=before_segments,
            after_segments=after_segments,
            stack_depth=options.stack_depth,
            top=options.limit,
        )

    allocation_lifetimes = None
    if options.lifetimes:
        lifetime_options = replace(options, lifetimes=False)
        allocation_lifetimes = analyze_allocation_lifetimes(
            run,
            start=before,
            end=after,
            active_at=None,
            born_between=None,
            options=lifetime_options,
        )
        warnings.extend(allocation_lifetimes.warnings)

    return MemoryComparison(
        before=before,
        after=after,
        totals=compare_allocator_scopes(before.totals, after.totals),
        pools=pools,
        pool_streams=pool_streams,
        allocation_stacks=stack_deltas,
        allocation_stacks_by_stream=stack_detail_deltas,
        before_stack_coverage=before_coverage,
        after_stack_coverage=after_coverage,
        allocator_events=allocator_events,
        allocation_lifetimes=allocation_lifetimes,
        events_available=events_available,
        events_complete=events_complete,
        lifecycle_available=True,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _segment_device_indices(
    *segment_groups: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    devices = {
        int(segment["device"])
        for segments in segment_groups
        for segment in segments
        if segment.get("device") is not None
    }
    return tuple(sorted(devices))


def _build_timeline(run: MemoryRun, options: AttributionOptions) -> MemoryTimeline:
    total_rows: list[PointAllocatorScopeState] = []
    pool_rows: list[PointPoolState] = []
    group_rows: list[PointPoolStreamState] = []
    previous_pools: Mapping[PoolId, MemoryStats] = {}
    previous_totals = summarize_allocator_scopes(previous_pools)
    previous_groups: Mapping[GroupKey, MemoryStats] = {}
    for point in run.points:
        current_pools = point.pools
        current_totals = point.totals
        for scope in ALLOCATOR_SCOPES:
            stats = current_totals[scope]
            total_rows.append(
                PointAllocatorScopeState(
                    point_index=point.index,
                    point_label=point.label,
                    scope=scope,
                    stats=stats,
                    delta=MemoryStatsDelta.between(previous_totals[scope], stats),
                )
            )
        for pool_id in sorted(
            set(previous_pools) | set(current_pools), key=pool_id_label
        ):
            before_stats = previous_pools.get(pool_id, MemoryStats())
            stats = current_pools.get(pool_id, MemoryStats())
            pool_rows.append(
                PointPoolState(
                    point_index=point.index,
                    point_label=point.label,
                    pool_id=pool_id,
                    stats=stats,
                    delta=MemoryStatsDelta.between(before_stats, stats),
                )
            )
        for key in sorted(
            set(previous_groups) | set(point.groups),
            key=lambda item: (pool_id_label(item.pool_id), stream_label(item.stream)),
        ):
            before_stats = previous_groups.get(key, MemoryStats())
            stats = point.groups.get(key, MemoryStats())
            group_rows.append(
                PointPoolStreamState(
                    point_index=point.index,
                    point_label=point.label,
                    pool_id=key.pool_id,
                    stream=key.stream,
                    stats=stats,
                    delta=MemoryStatsDelta.between(before_stats, stats),
                )
            )
        previous_pools = current_pools
        previous_totals = current_totals
        previous_groups = point.groups

    allocation_lifetimes = None
    adjacent_options = replace(options, lifetimes=False)
    if options.lifetimes and run.points:
        allocation_lifetimes = analyze_allocation_lifetimes(
            run,
            start=run.points[0],
            end=run.points[-1],
            active_at=None,
            born_between=None,
            options=adjacent_options,
        )
    adjacent = (
        _build_adjacent_comparisons(run, adjacent_options)
        if options.stacks or options.events
        else ()
    )
    return MemoryTimeline(
        run=run,
        totals=tuple(total_rows),
        pools=tuple(pool_rows),
        pool_streams=tuple(group_rows),
        adjacent=adjacent,
        allocation_lifetimes=allocation_lifetimes,
    )


def _build_adjacent_comparisons(
    run: MemoryRun,
    options: AttributionOptions,
) -> tuple[MemoryComparison, ...]:
    if len(run.points) < 2:
        return ()
    rows = []
    previous = _load_point_snapshot(run.points[0], options)
    for point in run.points[1:]:
        current = _load_point_snapshot(point, options)
        rows.append(_compare_same_run_views(run, previous, current, options))
        previous = current
    return tuple(rows)


def _validate_pool_mapping(
    before_pools: Mapping[PoolId, MemoryStats],
    after_pools: Mapping[PoolId, MemoryStats],
    pool_mapping: Mapping[Any, Any] | None,
) -> dict[PoolId, tuple[PoolId, Literal["default", "mapped"]]]:
    result: dict[PoolId, tuple[PoolId, Literal["default", "mapped"]]] = {}
    if DEFAULT_POOL_ID in before_pools and DEFAULT_POOL_ID in after_pools:
        result[DEFAULT_POOL_ID] = (DEFAULT_POOL_ID, "default")

    targets: set[PoolId] = {target for target, _match in result.values()}
    for raw_before, raw_after in (pool_mapping or {}).items():
        before_pool = normalize_pool_id(raw_before)
        after_pool = normalize_pool_id(raw_after)
        if before_pool == DEFAULT_POOL_ID or after_pool == DEFAULT_POOL_ID:
            if before_pool != DEFAULT_POOL_ID or after_pool != DEFAULT_POOL_ID:
                raise ValueError("the default pool may only map to the default pool")
            continue
        if before_pool not in before_pools:
            raise ValueError(
                f"mapped source pool {pool_id_label(before_pool)} does not exist"
            )
        if after_pool not in after_pools:
            raise ValueError(
                f"mapped target pool {pool_id_label(after_pool)} does not exist"
            )
        if before_pool in result:
            raise ValueError(
                f"source pool {pool_id_label(before_pool)} is mapped more than once"
            )
        if after_pool in targets:
            raise ValueError(
                f"target pool {pool_id_label(after_pool)} is mapped more than once"
            )
        result[before_pool] = (after_pool, "mapped")
        targets.add(after_pool)
    return result


def _unmatched_cross_run_groups(
    before: MemoryPoint,
    after: MemoryPoint,
) -> tuple[PoolStreamComparison, ...]:
    rows: list[PoolStreamComparison] = []
    for key, before_stats in before.groups.items():
        after_stats = MemoryStats()
        rows.append(
            PoolStreamComparison(
                before_pool_id=key.pool_id,
                before_stream=key.stream,
                after_pool_id=None,
                after_stream=None,
                match="before_only",
                before=before_stats,
                after=after_stats,
                delta=MemoryStatsDelta.between(before_stats, after_stats),
                lifecycle=None,
            )
        )
    for key, after_stats in after.groups.items():
        before_stats = MemoryStats()
        rows.append(
            PoolStreamComparison(
                before_pool_id=None,
                before_stream=None,
                after_pool_id=key.pool_id,
                after_stream=key.stream,
                match="after_only",
                before=before_stats,
                after=after_stats,
                delta=MemoryStatsDelta.between(before_stats, after_stats),
                lifecycle=None,
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                pool_id_label(item.after_pool_id or item.before_pool_id or ()),
                stream_label(
                    item.after_stream
                    if item.after_pool_id is not None
                    else item.before_stream
                ),
                item.match,
            ),
        )
    )


def _check_stack_coverage(
    before: AllocationStackCoverage,
    after: AllocationStackCoverage,
    options: AttributionOptions,
    warnings: list[str],
) -> None:
    if before.unattributed_bytes or after.unattributed_bytes:
        _missing(
            "allocation stack coverage is incomplete: "
            f"before={before.ratio:.1%}, after={after.ratio:.1%}; "
            "enable torch.cuda.memory._record_memory_history(...) "
            "before the allocations",
            options,
            warnings,
        )


def _missing(
    message: str,
    options: AttributionOptions,
    warnings: list[str],
) -> None:
    if options.on_missing == "error":
        raise MemoryHistoryError(message)
    warnings.append(message)


def _phase_decomposition(
    baseline_growth: MemoryComparison,
    candidate_growth: MemoryComparison,
    start_delta: MemoryComparison,
    end_delta: MemoryComparison,
) -> tuple[dict[str, object], ...]:
    baseline_by_pool = {
        item.before_pool_id: item
        for item in baseline_growth.pools
        if item.before_pool_id is not None
    }
    candidate_by_pool = {
        item.before_pool_id: item
        for item in candidate_growth.pools
        if item.before_pool_id is not None
    }
    end_by_pair = {
        (item.before_pool_id, item.after_pool_id): item for item in end_delta.pools
    }
    rows: list[dict[str, object]] = []
    for start in start_delta.pools:
        if start.match not in {"default", "mapped"}:
            continue
        baseline_pool = start.before_pool_id
        candidate_pool = start.after_pool_id
        if baseline_pool is None or candidate_pool is None:
            continue
        baseline = baseline_by_pool.get(baseline_pool)
        candidate = candidate_by_pool.get(candidate_pool)
        end = end_by_pair.get((baseline_pool, candidate_pool))
        if baseline is None or candidate is None or end is None:
            continue
        for metric in (
            "reserved_bytes",
            "allocated_bytes",
            "active_bytes",
            "inactive_bytes",
            "requested_bytes",
        ):
            baseline_change = int(getattr(baseline.delta, metric))
            candidate_change = int(getattr(candidate.delta, metric))
            start_value = int(getattr(start.delta, metric))
            end_value = int(getattr(end.delta, metric))
            rows.append(
                {
                    "baseline_pool_id": pool_id_label(baseline_pool),
                    "candidate_pool_id": pool_id_label(candidate_pool),
                    "pool": (
                        f"{pool_id_label(baseline_pool)} -> "
                        f"{pool_id_label(candidate_pool)}"
                    ),
                    "metric": metric,
                    "start_delta_bytes": start_value,
                    "baseline_growth_bytes": baseline_change,
                    "candidate_growth_bytes": candidate_change,
                    "growth_delta_bytes": candidate_change - baseline_change,
                    "end_delta_bytes": end_value,
                    "identity_holds": (
                        end_value == start_value + candidate_change - baseline_change
                    ),
                }
            )
    return tuple(rows)


def _scope_phase_decomposition(
    baseline_growth: MemoryComparison,
    candidate_growth: MemoryComparison,
    start_delta: MemoryComparison,
    end_delta: MemoryComparison,
) -> tuple[dict[str, object], ...]:
    comparisons = (
        {item.scope: item for item in baseline_growth.totals},
        {item.scope: item for item in candidate_growth.totals},
        {item.scope: item for item in start_delta.totals},
        {item.scope: item for item in end_delta.totals},
    )
    rows = []
    for scope in ALLOCATOR_SCOPES:
        baseline, candidate, start, end = (
            comparison[scope] for comparison in comparisons
        )
        for metric in (
            "reserved_bytes",
            "allocated_bytes",
            "active_bytes",
            "inactive_bytes",
            "requested_bytes",
        ):
            baseline_change = int(getattr(baseline.delta, metric))
            candidate_change = int(getattr(candidate.delta, metric))
            start_value = int(getattr(start.delta, metric))
            end_value = int(getattr(end.delta, metric))
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "start_delta_bytes": start_value,
                    "baseline_growth_bytes": baseline_change,
                    "candidate_growth_bytes": candidate_change,
                    "growth_delta_bytes": candidate_change - baseline_change,
                    "end_delta_bytes": end_value,
                    "identity_holds": (
                        end_value == start_value + candidate_change - baseline_change
                    ),
                }
            )
    return tuple(rows)


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
        "groups": [
            _group_manifest(key, stats) for key, stats in point.groups.items()
        ],
    }


def _group_manifest(key: GroupKey, stats: MemoryStats) -> dict[str, object]:
    return {
        "pool_id": list(key.pool_id),
        "stream": key.stream,
        **{name: getattr(stats, name) for name in _MEMORY_STATS_FIELDS},
    }


def _group_from_manifest(
    row: Mapping[str, Any],
) -> tuple[GroupKey, MemoryStats]:
    return (
        GroupKey(
            pool_id=normalize_pool_id(row["pool_id"]),
            stream=row["stream"],
        ),
        MemoryStats.from_dict(row),
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


def _snapshot_envelope(snapshot: SnapshotInput) -> SnapshotInput:
    if isinstance(snapshot, Mapping):
        envelope = dict(snapshot)
    else:
        envelope = {"segments": list(snapshot)}
    envelope.setdefault("device_traces", [])
    envelope.setdefault("external_annotations", [])
    envelope.setdefault("allocator_settings", {})
    return envelope


def _current_stream_is_capturing(torch_module: Any) -> bool:
    is_capturing = getattr(torch_module.cuda, "is_current_stream_capturing", None)
    if not callable(is_capturing):
        return False
    try:
        return bool(is_capturing())
    except Exception:
        return False


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


def _pool_comparison_sort_key(
    item: PoolComparison,
) -> tuple[str, str, str]:
    return (
        pool_id_label(item.before_pool_id or ()),
        pool_id_label(item.after_pool_id or ()),
        item.match,
    )
