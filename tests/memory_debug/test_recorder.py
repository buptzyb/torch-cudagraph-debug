from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryBundleError,
    MemoryDebugError,
    MemoryOwnershipError,
    MemoryPoolKey,
    MemoryRecorder,
    MemoryRun,
)

from ._helpers import event, make_history_run, make_run, segment, snapshot


def test_recorder_finish_is_idempotent_and_freezes_collection() -> None:
    pending = [snapshot(segment(active=10))]
    recorder = MemoryRecorder._from_snapshot_provider(
        lambda marker: pending.pop(0),
        name="sample",
    )
    point = recorder.record_point("start", metadata={"step": 1})
    view = recorder.preview()
    descriptor = point.descriptor()
    assert descriptor["boundary_marker"] == point.boundary_marker
    assert descriptor["observation_count"] == len(point.observations)

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
        "device_index",
        "pool_id",
        "stream",
        "reserved_bytes",
        "allocated_bytes",
        "active_bytes",
        "requested_bytes",
        "segment_count",
        "block_count",
        "inactive_block_count",
        "largest_inactive_block_bytes",
        "expandable_segment_count",
        "expandable_reserved_bytes",
    }
    point_manifest = manifest["points"][1]
    state_path = bundle / point_manifest["state_file"]
    event_path = bundle / point_manifest["event_file"]
    assert state_path == bundle / "states" / "0001.json.gz"
    assert event_path == bundle / "events" / "0000-0001.json.gz"
    assert point_manifest["boundary_recorded"] is True
    assert point_manifest["history"][0]["status"] == "disabled"
    with gzip.open(state_path, "rt", encoding="utf-8") as handle:
        state_payload = json.load(handle)
    assert "device_traces" not in state_payload
    assert state_payload["segments"][1]["segment_pool_id"] == [0, 3]
    with gzip.open(event_path, "rt", encoding="utf-8") as handle:
        event_payload = json.load(handle)
    assert event_payload == {
        "start_index": 0,
        "end_index": 1,
        "devices": [{"device_index": 0, "trace_index_offset": 0, "entries": []}],
    }

    loaded = MemoryRun.load(bundle)
    assert loaded.complete is True
    observation = loaded["after"].observation(0, (0, 3), 7)
    assert observation.order == 1
    assert observation.stats.active_bytes == 20
    assert loaded["after"].observation_stats[observation.key] == observation.stats
    assert loaded["after"]._state_cache == {}
    assert loaded["after"]._event_cache == {}
    assert loaded["after"].allocator_state()["segments"][1]["total_size"] == 32
    assert loaded["after"]._state_cache
    assert loaded["after"]._event_cache == {}
    comparison = loaded.compare("before", "after")
    assert comparison.lifecycle_available is True
    assert any(
        item.candidate_key == MemoryPoolKey(0, (0, 3))
        for item in comparison.pool_comparisons
    )
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


def test_schema_drift_missing_fields_warn_once_per_field() -> None:
    def drifted_segment(address: int) -> dict[str, object]:
        return {
            "address": address,
            "stream": 0,
            "segment_type": "large",
            "total_size": 100,
            "allocated_size": 100,
            "active_size": 100,
            # missing: device, segment_pool_id, requested_size
            "blocks": [
                {
                    "address": address,
                    "size": 100,
                    "state": "active_allocated",
                    "frames": [],
                    # missing: requested_size
                }
            ],
        }

    raw = {
        "segments": [drifted_segment(1000), drifted_segment(2000)],
        "device_traces": [],
        "external_annotations": [],
        "allocator_settings": {},
    }
    run = make_run([raw], labels=("point",))

    joined = "\n".join(run.points[0].warnings)
    assert "2 segment(s) missing 'device' (treated as device 0)" in joined
    assert (
        "2 segment(s) missing 'segment_pool_id' (treated as the default pool)" in joined
    )
    assert "2 segment(s) missing 'requested_size' (treated as 0)" in joined
    assert "2 block(s) missing 'requested_size' (treated as 0)" in joined


def test_complete_snapshot_produces_no_schema_warnings() -> None:
    run = make_run([snapshot(segment(active=10))], labels=("point",))
    assert run.points[0].warnings == ()


def test_truncated_state_payload_raises_bundle_error(tmp_path: Path) -> None:
    bundle = tmp_path / "truncated.tcgd-memory"
    make_run(
        [snapshot(segment(active=10))],
        bundle_dir=bundle,
        labels=("point",),
    )
    payload = bundle / "states" / "0000.json.gz"
    data = payload.read_bytes()
    payload.write_bytes(data[: len(data) // 2])

    run = MemoryRun.load(bundle, cache_snapshots=False)
    with pytest.raises(MemoryBundleError, match="could not load state"):
        run.points[0].allocator_state()


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


def test_load_rejects_state_path_traversal(tmp_path: Path) -> None:
    bundle = tmp_path / "run"
    make_run(
        [snapshot(segment(active=1))],
        bundle_dir=bundle,
        labels=("point",),
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["points"][0]["state_file"] = "../outside.json.gz"
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


def test_context_exception_persists_incomplete_terminal_run(tmp_path: Path) -> None:
    bundle = tmp_path / "aborted.tcgd-memory"
    pending = [snapshot(segment(active=10))]
    recorder = MemoryRecorder._from_snapshot_provider(
        lambda marker: pending.pop(0),
        name="aborted",
        bundle_dir=bundle,
    )

    with pytest.raises(RuntimeError, match="boom"):
        with recorder:
            recorder.record_point("inside")
            raise RuntimeError("boom")

    assert recorder.result.complete is False
    assert recorder.result.finished_at is not None
    assert recorder.preview() is recorder.result
    assert recorder.finish() is recorder.result
    loaded = MemoryRun.load(bundle)
    assert loaded.complete is False
    assert [point.label for point in loaded.points] == ["inside"]
    with pytest.raises(MemoryDebugError, match="finished"):
        recorder.record_point("late")


def test_manifest_failures_do_not_commit_memory_recorder_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = [snapshot(segment(active=10)), snapshot(segment(active=20))]
    recorder = MemoryRecorder._from_snapshot_provider(
        lambda marker: pending.pop(0),
        bundle_dir=tmp_path / "transaction.tcgd-memory",
    )
    original = recorder._write_manifest
    failed_point = False
    failed_finish = False

    def flaky(**kwargs: object) -> None:
        nonlocal failed_point, failed_finish
        complete = kwargs["complete"]
        if not complete and not failed_point:
            failed_point = True
            raise MemoryBundleError("injected manifest failure")
        if complete and not failed_finish:
            failed_finish = True
            raise MemoryBundleError("injected manifest failure")
        original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(recorder, "_write_manifest", flaky)
    with pytest.raises(MemoryBundleError, match="injected"):
        recorder.record_point("point")
    assert recorder.preview().points == ()

    recorder.record_point("point")
    with pytest.raises(MemoryBundleError, match="injected"):
        recorder.finish()
    assert recorder._result is None
    assert recorder.finish().complete is True


def test_memory_bundle_load_rejects_lossy_scalar_coercions(tmp_path: Path) -> None:
    bundle = tmp_path / "strict.tcgd-memory"
    make_run(
        [snapshot(segment(active=1))],
        bundle_dir=bundle,
        labels=("point",),
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["complete"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MemoryBundleError, match="boolean"):
        MemoryRun.load(bundle)

    manifest["complete"] = True
    manifest["points"][0]["index"] = 0.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MemoryBundleError, match="integer"):
        MemoryRun.load(bundle)

    manifest["points"][0]["index"] = 0
    manifest["created_at"] = float("nan")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MemoryBundleError, match="non-finite"):
        MemoryRun.load(bundle)


def test_memory_recorder_rejects_bad_environment_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "not-an-integer")

    with pytest.raises(ValueError, match="environment variable WORLD_SIZE"):
        MemoryRecorder()


def test_memory_recorder_context_rejects_reentry_and_explicit_finish() -> None:
    recorder = MemoryRecorder._from_snapshot_provider(lambda marker: snapshot())

    with recorder:
        with pytest.raises(MemoryDebugError, match="re-entered"):
            recorder.__enter__()
        with pytest.raises(MemoryDebugError, match="inside its context"):
            recorder.finish()


def test_memory_run_point_and_snapshot_data_are_deeply_immutable() -> None:
    source = {"nested": {"values": [1, 2]}}
    recorder = MemoryRecorder._from_snapshot_provider(
        lambda marker: snapshot(segment(active=10)),
        run_metadata=source,
    )
    point = recorder.record_point("point", metadata=source)
    run = recorder.finish()

    source["nested"]["values"].append(3)
    assert run.run_metadata["nested"]["values"] == (1, 2)
    assert point.metadata["nested"]["values"] == (1, 2)
    raw = point.allocator_state()
    assert isinstance(raw["segments"], tuple)
    with pytest.raises(TypeError):
        raw["segments"][0]["total_size"] = 0


def test_event_payload_is_preserved_and_loaded_independently(tmp_path: Path) -> None:
    allocation = event("alloc", address=1000, size=64)
    allocation["future_allocator_field"] = {"nested": [1, 2]}
    bundle = tmp_path / "event-evidence.tcgd-memory"
    make_history_run(
        [
            ([segment(active=0, address=1000)], []),
            ([segment(active=64, address=1000)], [allocation]),
        ],
        labels=("before", "after"),
        bundle_dir=bundle,
    )

    event_path = bundle / "events" / "0000-0001.json.gz"
    with gzip.open(event_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["devices"][0]["entries"][0]["future_allocator_field"] == {
        "nested": [1, 2]
    }

    data = event_path.read_bytes()
    event_path.write_bytes(data[: len(data) // 2])
    run = MemoryRun.load(bundle, cache_snapshots=False)
    assert run.compare("before", "after").candidate.pool_stats
    with pytest.raises(MemoryBundleError, match="could not load event evidence"):
        run.compare(
            "before",
            "after",
            attribution=MemoryAttributionOptions(events=True),
        )


def test_invalid_allocator_schema_in_payloads_raises_bundle_error(
    tmp_path: Path,
) -> None:
    allocation = event("alloc", address=1000, size=64)
    bundle = tmp_path / "invalid-allocator-schema.tcgd-memory"
    make_history_run(
        [
            ([segment(active=0, address=1000)], []),
            ([segment(active=64, address=1000)], [allocation]),
        ],
        labels=("before", "after"),
        bundle_dir=bundle,
    )

    state_path = bundle / "states" / "0001.json.gz"
    with gzip.open(state_path, "rt", encoding="utf-8") as handle:
        state_payload = json.load(handle)
    state_payload["segments"][0]["total_size"] = "invalid"
    with gzip.open(state_path, "wt", encoding="utf-8") as handle:
        json.dump(state_payload, handle)

    run = MemoryRun.load(bundle, cache_snapshots=False)
    with pytest.raises(MemoryBundleError, match="invalid allocator schema"):
        run["after"].allocator_state()

    state_payload["segments"][0]["total_size"] = 64
    with gzip.open(state_path, "wt", encoding="utf-8") as handle:
        json.dump(state_payload, handle)
    event_path = bundle / "events" / "0000-0001.json.gz"
    with gzip.open(event_path, "rt", encoding="utf-8") as handle:
        event_payload = json.load(handle)
    event_payload["devices"][0]["entries"][0]["size"] = "invalid"
    with gzip.open(event_path, "wt", encoding="utf-8") as handle:
        json.dump(event_payload, handle)

    run = MemoryRun.load(bundle, cache_snapshots=False)
    with pytest.raises(MemoryBundleError, match="invalid allocator schema"):
        run.compare(
            "before",
            "after",
            attribution=MemoryAttributionOptions(events=True),
        )
