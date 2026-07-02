from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from torch_cudagraph_debug.memory_debug import (
    MemoryBundleError,
    MemoryDebugError,
    MemoryOwnershipError,
    MemoryRecorder,
    MemoryRun,
)

from ._helpers import make_run, segment, snapshot


def test_recorder_finish_is_idempotent_and_freezes_collection() -> None:
    pending = [snapshot(segment(active=10))]
    recorder = MemoryRecorder._from_snapshot_provider(
        lambda marker: pending.pop(0),
        name="sample",
    )
    point = recorder.record_point("start", metadata={"step": 1})
    view = recorder.snapshot_run()

    assert view.complete is False
    assert view.point("start").run_id == point.run_id
    with pytest.raises(MemoryDebugError, match="not been finished"):
        _ = recorder.result

    run = recorder.finish()
    assert run.complete is True
    assert run is recorder.finish()
    assert recorder.result is run
    assert run["start"].metadata == {"step": 1}
    with pytest.raises(MemoryDebugError, match="finished"):
        recorder.record_point("late")


def test_context_manager_exposes_result_after_exit() -> None:
    pending = [snapshot(segment(active=10))]
    with MemoryRecorder._from_snapshot_provider(
        lambda marker: pending.pop(0)
    ) as recorder:
        recorder.record_point("inside")
    assert recorder.result.complete is True


def test_labels_must_be_nonempty_and_unique() -> None:
    pending = [snapshot(), snapshot()]
    recorder = MemoryRecorder._from_snapshot_provider(lambda marker: pending.pop(0))
    with pytest.raises(ValueError, match="non-empty"):
        recorder.record_point("")
    recorder.record_point("same")
    with pytest.raises(ValueError, match="already exists"):
        recorder.record_point("same")


def test_bundle_is_gzip_json_and_load_is_lazy(tmp_path: Path) -> None:
    bundle = tmp_path / "run.tcgd-memory"
    run = make_run(
        [
            snapshot(segment(active=10)),
            snapshot(
                segment(active=10),
                segment(
                    active=20,
                    total=32,
                    pool=(0, 3),
                    stream=7,
                    address=9000,
                ),
            ),
        ],
        name="json-run",
        bundle_dir=bundle,
        labels=("before", "after"),
    )

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "torch-cudagraph-debug/memory-run"
    assert manifest["complete"] is True
    assert manifest["name"] == "json-run"
    assert set(manifest["points"][0]["observations"][0]) == {
        "order",
        "pool_id",
        "stream",
        "reserved_bytes",
        "allocated_bytes",
        "active_bytes",
        "requested_bytes",
        "segment_count",
        "block_count",
        "largest_inactive_block_bytes",
    }
    snapshot_path = bundle / manifest["points"][1]["snapshot_file"]
    assert snapshot_path.name.endswith(".json.gz")
    with gzip.open(snapshot_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["segments"][1]["segment_pool_id"] == [0, 3]

    loaded = MemoryRun.load(bundle)
    assert loaded.complete is True
    observation = loaded["after"].observation((0, 3), 7)
    assert observation.order == 1
    assert observation.stats.active_bytes == 20
    assert loaded["after"].observation_stats[observation.key] == observation.stats
    assert loaded["after"]._snapshot_cache == {}
    assert loaded["after"].raw_snapshot()["segments"][1]["total_size"] == 32
    assert loaded["after"]._snapshot_cache
    comparison = loaded.compare("before", "after")
    assert comparison.lifecycle_available is True
    assert any(item.candidate_pool_id == (0, 3) for item in comparison.pool_comparisons)
    assert (
        run.compare("before", "after").pool_comparison_rows()
        == comparison.pool_comparison_rows()
    )


def test_bundle_round_trips_group_identity_and_provenance(tmp_path: Path) -> None:
    bundle = tmp_path / "rank-1.tcgd-memory"
    make_run(
        [snapshot(segment(active=10))],
        name="distributed-run",
        bundle_dir=bundle,
        labels=("point",),
        rank=1,
        group_id="job-123",
        world_size=4,
        run_metadata={"source_revision": "abc123", "scenario": "candidate"},
    )

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rank"] == 1
    assert manifest["group_id"] == "job-123"
    assert manifest["world_size"] == 4
    assert manifest["provenance"]["producer"]["name"] == "torch-cudagraph-debug"
    assert manifest["provenance"]["runtime"]["torch"]
    assert manifest["run_metadata"] == {
        "source_revision": "abc123",
        "scenario": "candidate",
    }

    loaded = MemoryRun.load(bundle)
    assert loaded.rank == 1
    assert loaded.group_id == "job-123"
    assert loaded.world_size == 4
    assert loaded.provenance["runtime"]["torch"]
    assert loaded.run_metadata["source_revision"] == "abc123"


def test_load_rejects_unrecognized_bundle_schema(tmp_path: Path) -> None:
    bundle = tmp_path / "wrong-schema.tcgd-memory"
    make_run(
        [snapshot(segment(active=1))],
        bundle_dir=bundle,
        labels=("point",),
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "other-project/memory-run"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MemoryBundleError, match="unsupported memory bundle schema"):
        MemoryRun.load(bundle)


def test_load_rejects_missing_manifest_fields(tmp_path: Path) -> None:
    bundle = tmp_path / "missing-field.tcgd-memory"
    make_run(
        [snapshot(segment(active=1))],
        bundle_dir=bundle,
        labels=("point",),
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("run_metadata")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MemoryBundleError, match=r"invalid fields.*run_metadata"):
        MemoryRun.load(bundle)


def test_load_rejects_unknown_point_fields(tmp_path: Path) -> None:
    bundle = tmp_path / "unknown-point-field.tcgd-memory"
    make_run(
        [snapshot(segment(active=1))],
        bundle_dir=bundle,
        labels=("point",),
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["points"][0]["unexpected_field"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MemoryBundleError, match=r"invalid fields.*unexpected_field"):
        MemoryRun.load(bundle)


def test_load_rejects_unknown_observation_fields(tmp_path: Path) -> None:
    bundle = tmp_path / "unknown-group-field.tcgd-memory"
    make_run(
        [snapshot(segment(active=1))],
        bundle_dir=bundle,
        labels=("point",),
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["points"][0]["observations"][0]["unexpected_field"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MemoryBundleError, match=r"invalid fields.*unexpected_field"):
        MemoryRun.load(bundle)


def test_recorder_validates_group_identity_and_user_metadata() -> None:
    with pytest.raises(ValueError, match="group_id"):
        MemoryRecorder(group_id="")
    with pytest.raises(ValueError, match="world_size"):
        MemoryRecorder(world_size=0)
    with pytest.raises(ValueError, match="rank"):
        MemoryRecorder(rank=2, world_size=2)
    with pytest.raises(MemoryBundleError, match=r"\$\.run_metadata\.bad"):
        MemoryRecorder(run_metadata={"bad": object()})


def test_bundle_rejects_unsupported_values_with_path() -> None:
    recorder = MemoryRecorder._from_snapshot_provider(
        lambda marker: {
            "segments": [],
            "device_traces": [],
            "bad": object(),
        }
    )
    with pytest.raises(MemoryBundleError, match=r"\$\.snapshot\.bad.*unsupported"):
        recorder.record_point("bad")


def test_load_rejects_snapshot_path_traversal(tmp_path: Path) -> None:
    bundle = tmp_path / "run"
    make_run(
        [snapshot(segment(active=1))],
        bundle_dir=bundle,
        labels=("point",),
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["points"][0]["snapshot_file"] = "../outside.json.gz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MemoryBundleError, match="escapes"):
        MemoryRun.load(bundle)


def test_run_rejects_foreign_points() -> None:
    reference = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        name="reference",
        labels=("before", "after"),
    )
    candidate = make_run(
        [snapshot(segment(active=30)), snapshot(segment(active=40))],
        name="candidate",
        labels=("before", "after"),
    )

    with pytest.raises(MemoryOwnershipError, match="belongs to run"):
        reference.compare(reference["before"], candidate["after"])
    with pytest.raises(MemoryOwnershipError, match="belongs to run"):
        reference.between(candidate["before"], reference["after"])


def test_memory_run_load_does_not_call_recorder_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "run"
    make_run(
        [snapshot(segment(active=1))],
        bundle_dir=bundle,
        labels=("point",),
    )

    def fail_init(*args: object, **kwargs: object) -> None:
        raise AssertionError("MemoryRun.load must not construct a recorder")

    monkeypatch.setattr(MemoryRecorder, "__init__", fail_init)
    loaded = MemoryRun.load(bundle)
    assert loaded["point"].label == "point"
