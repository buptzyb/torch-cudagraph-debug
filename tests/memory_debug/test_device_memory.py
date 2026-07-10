"""Device-wide CUDA Runtime memory sampling, persistence, comparison, and reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from torch_cudagraph_debug.memory_debug import (
    DeviceMemorySample,
    MemoryAllocatorScopePhaseDecomposition,
    MemoryBundleError,
    MemoryDevicePhaseDecomposition,
    MemoryPhaseComponents,
    MemoryPoolKey,
    MemoryPoolPhaseDecomposition,
    MemoryProbe,
    MemoryRecorder,
    MemoryRun,
    compare_phases,
    compare_snapshots,
)
from torch_cudagraph_debug.memory_debug.aggregation import summarize_devices

from ._helpers import make_run, segment, snapshot


def _patch_torch_collection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    capturing: object = False,
) -> None:
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", capturing, raising=False
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(
        torch.cuda.memory, "_snapshot", lambda: snapshot(segment(active=10))
    )
    monkeypatch.setattr(torch.cuda.memory, "_get_memory_metadata", lambda: "")
    monkeypatch.setattr(torch.cuda.memory, "_set_memory_metadata", lambda value: None)


def test_phase_decompositions_validate_metric_domains() -> None:
    allocator_components = MemoryPhaseComponents("reserved_bytes", 0, 0, 0, 0)
    device_components = MemoryPhaseComponents("used_bytes", 0, 0, 0, 0)

    MemoryAllocatorScopePhaseDecomposition("all", allocator_components)
    pool_key = MemoryPoolKey(0, (0, 0))
    MemoryPoolPhaseDecomposition(pool_key, pool_key, allocator_components)
    MemoryDevicePhaseDecomposition(0, 0, device_components)
    with pytest.raises(ValueError, match="allocator metric"):
        MemoryAllocatorScopePhaseDecomposition("all", device_components)
    with pytest.raises(ValueError, match="pool decomposition"):
        MemoryPoolPhaseDecomposition(pool_key, pool_key, device_components)
    with pytest.raises(ValueError, match="device memory metric"):
        MemoryDevicePhaseDecomposition(0, 0, allocator_components)


def test_device_memory_sample_validates_and_derives_used() -> None:
    sample = DeviceMemorySample(free_bytes=30, total_bytes=100)
    assert sample.used_bytes == 70
    assert sample.to_dict() == {
        "free_bytes": 30,
        "total_bytes": 100,
        "used_bytes": 70,
    }
    with pytest.raises(ValueError, match="non-negative integer"):
        DeviceMemorySample(free_bytes=-1, total_bytes=100)
    with pytest.raises(ValueError, match="non-negative integer"):
        DeviceMemorySample(free_bytes=True, total_bytes=100)
    with pytest.raises(ValueError, match="must not exceed total_bytes"):
        DeviceMemorySample(free_bytes=101, total_bytes=100)
    with pytest.raises(ValueError, match="non-negative integer"):
        DeviceMemorySample.from_dict({"free_bytes": 1.0, "total_bytes": 100})


def test_recorder_provider_seam_populates_point_device_memory() -> None:
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        device_memory=[{0: (90, 100)}, {0: (80, 100)}],
    )
    assert dict(run.points[0].device_memory) == {0: DeviceMemorySample(90, 100)}
    assert dict(run.points[1].device_memory) == {0: DeviceMemorySample(80, 100)}


def test_recorder_without_device_provider_reports_nothing() -> None:
    run = make_run([snapshot(segment(active=10))])
    assert dict(run.points[0].device_memory) == {}
    assert not any("device memory" in warning for warning in run.points[0].warnings)


def test_probe_provider_seam_populates_snapshot_device_memory() -> None:
    values = iter([snapshot(segment(active=10))])
    probe = MemoryProbe._from_snapshot_provider(
        lambda marker: next(values),
        device_memory_provider=lambda device: (90, 100),
    )
    taken = probe.snapshot()
    assert dict(taken.device_memory) == {0: DeviceMemorySample(90, 100)}


def test_provider_failure_becomes_a_per_device_warning() -> None:
    values = iter([snapshot(segment(active=10))])
    probe = MemoryProbe._from_snapshot_provider(
        lambda marker: next(values),
        device_memory_provider=lambda device: (_ for _ in ()).throw(
            RuntimeError("provider failed")
        ),
    )
    taken = probe.snapshot()
    assert dict(taken.device_memory) == {}
    assert any(
        "could not sample device memory for device 0" in warning
        for warning in taken.warnings
    )


def test_multi_device_samples_cover_every_selected_device() -> None:
    run = make_run(
        [
            snapshot(
                segment(active=10, device=1, address=5000),
                segment(active=10, device=0),
            )
        ],
        device_memory=[{1: (50, 200), 0: (90, 100)}],
    )
    point = run.points[0]
    assert list(point.device_memory) == [0, 1]
    assert point.device_memory[1] == DeviceMemorySample(50, 200)


def test_torch_path_samples_selected_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_torch_collection(monkeypatch, capturing=lambda: False)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device=None: (90, 100), raising=False
    )
    point = MemoryRecorder().record_point("outside_capture")
    assert dict(point.device_memory) == {0: DeviceMemorySample(90, 100)}


def test_torch_path_samples_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_torch_collection(monkeypatch, capturing=lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device=None: (90, 100), raising=False
    )
    point = MemoryRecorder().record_point("inside_capture")
    assert dict(point.device_memory) == {0: DeviceMemorySample(90, 100)}
    assert not any(
        "device memory sampling was skipped" in warning for warning in point.warnings
    )


def test_torch_path_samples_when_capture_state_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_capture_state() -> bool:
        raise RuntimeError("no current stream")

    _patch_torch_collection(monkeypatch, capturing=broken_capture_state)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device=None: (90, 100), raising=False
    )
    point = MemoryRecorder().record_point("unknown_state")
    assert dict(point.device_memory) == {0: DeviceMemorySample(90, 100)}
    assert any(
        "requested allocator synchronization was skipped" in warning
        for warning in point.warnings
    )


def test_torch_path_warns_when_capture_sampling_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_torch_collection(monkeypatch, capturing=lambda: True)

    def fail_mem_get_info(device: object = None) -> tuple[int, int]:
        raise RuntimeError("query failed during capture")

    monkeypatch.setattr(torch.cuda, "mem_get_info", fail_mem_get_info, raising=False)
    point = MemoryRecorder().record_point("inside_capture")
    assert dict(point.device_memory) == {}
    assert any(
        "could not sample device memory for device 0" in warning
        for warning in point.warnings
    )


def test_torch_path_keeps_healthy_devices_when_one_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_torch_collection(monkeypatch, capturing=lambda: False)

    def flaky_mem_get_info(device: object = None) -> tuple[int, int]:
        if device == 1:
            raise RuntimeError("driver rejected the query")
        return (90, 100)

    monkeypatch.setattr(torch.cuda, "mem_get_info", flaky_mem_get_info, raising=False)
    point = MemoryRecorder(devices=(0, 1)).record_point("partial")
    assert dict(point.device_memory) == {0: DeviceMemorySample(90, 100)}
    assert any(
        "could not sample device memory for device 1" in warning
        for warning in point.warnings
    )


def test_device_memory_is_immutable_and_in_descriptors() -> None:
    run = make_run(
        [snapshot(segment(active=10))],
        device_memory=[{0: (90, 100)}],
    )
    point = run.points[0]
    with pytest.raises(TypeError):
        point.device_memory[0] = DeviceMemorySample(0, 0)  # type: ignore[index]
    assert point.descriptor()["device_memory"] == {
        "0": {"free_bytes": 90, "total_bytes": 100, "used_bytes": 10}
    }


def _device_memory_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "device-memory.tcgd-memory"
    make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        bundle_dir=bundle,
        labels=("start", "end"),
        device_memory=[{0: (90, 100)}, {0: (80, 100)}],
    )
    return bundle


def test_bundle_round_trip_preserves_device_memory(tmp_path: Path) -> None:
    bundle = _device_memory_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["points"][0]["device_memory"] == {
        "0": {"free_bytes": 90, "total_bytes": 100}
    }
    loaded = MemoryRun.load(bundle)
    assert dict(loaded.points[0].device_memory) == {0: DeviceMemorySample(90, 100)}
    assert dict(loaded.points[1].device_memory) == {0: DeviceMemorySample(80, 100)}


def _rewrite_manifest(bundle: Path, mutate) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_load_rejects_unknown_device_memory_entry_fields(tmp_path: Path) -> None:
    bundle = _device_memory_bundle(tmp_path)

    def mutate(manifest: dict) -> None:
        manifest["points"][0]["device_memory"]["0"]["used_bytes"] = 10

    _rewrite_manifest(bundle, mutate)
    with pytest.raises(MemoryBundleError, match=r"invalid fields.*used_bytes"):
        MemoryRun.load(bundle)


def test_load_rejects_non_integer_device_memory_keys(tmp_path: Path) -> None:
    bundle = _device_memory_bundle(tmp_path)

    def mutate(manifest: dict) -> None:
        entry = manifest["points"][0]["device_memory"].pop("0")
        manifest["points"][0]["device_memory"]["cuda:0"] = entry

    _rewrite_manifest(bundle, mutate)
    with pytest.raises(MemoryBundleError, match="non-negative integer strings"):
        MemoryRun.load(bundle)


def test_load_rejects_inconsistent_device_memory_values(tmp_path: Path) -> None:
    bundle = _device_memory_bundle(tmp_path)

    def mutate(manifest: dict) -> None:
        manifest["points"][0]["device_memory"]["0"]["free_bytes"] = 101

    _rewrite_manifest(bundle, mutate)
    with pytest.raises(MemoryBundleError, match="invalid device_memory entry '0'"):
        MemoryRun.load(bundle)


def test_load_rejects_missing_device_memory_field(tmp_path: Path) -> None:
    bundle = _device_memory_bundle(tmp_path)

    def mutate(manifest: dict) -> None:
        del manifest["points"][0]["device_memory"]

    _rewrite_manifest(bundle, mutate)
    with pytest.raises(MemoryBundleError, match=r"invalid fields.*device_memory"):
        MemoryRun.load(bundle)


def test_summarize_devices_groups_pool_stats_by_device() -> None:
    run = make_run(
        [
            snapshot(
                segment(active=10, device=0),
                segment(active=20, device=0, pool=(1, 0), address=3000),
                segment(active=40, device=1, address=5000),
            )
        ]
    )
    by_device = summarize_devices(run.points[0].pool_stats)
    assert list(by_device) == [0, 1]
    assert by_device[0].reserved_bytes == 30
    assert by_device[1].reserved_bytes == 40


def test_same_run_comparison_reports_cuda_allocator_residual_growth() -> None:
    run = make_run(
        [
            snapshot(segment(active=100, total=100)),
            snapshot(segment(active=100, total=100)),
        ],
        labels=("before", "after"),
        device_memory=[{0: (900, 1000)}, {0: (700, 1000)}],
    )
    result = run.compare("before", "after")
    (row,) = result.device_comparisons
    assert row.reference_device_index == row.candidate_device_index == 0
    assert row.reference_allocator_reserved_bytes == 100
    assert row.candidate_allocator_reserved_bytes == 100
    assert row.reference_cuda_allocator_residual_bytes == 0
    assert row.candidate_cuda_allocator_residual_bytes == 200
    assert row.delta_used_bytes == 200
    assert row.delta_allocator_reserved_bytes == 0
    assert row.delta_cuda_allocator_residual_bytes == 200
    assert row.changed

    text = result.to_text()
    assert (
        "CUDA scope: device-wide, includes other processes; "
        "residual = CUDA used - allocator reserved" in text
    )
    assert "residual: 0 B -> 200 B (+200 B)" in text

    payload = result.to_dict()
    assert (
        payload["device_comparisons"][0]["delta"]["cuda_allocator_residual_bytes"]
        == 200
    )


def test_comparison_reports_cuda_visible_capacity_change() -> None:
    run = make_run(
        [
            snapshot(segment(active=100, total=100)),
            snapshot(segment(active=100, total=100)),
        ],
        labels=("before", "after"),
        device_memory=[{0: (900, 1000)}, {0: (800, 1100)}],
    )
    result = run.compare("before", "after")
    (row,) = result.device_comparisons
    assert row.delta_used_bytes == 200
    assert row.delta_free_bytes == -100
    assert row.delta_total_bytes == 100
    assert "CUDA total: 1000 B -> 1.07 KiB (+100 B)" in result.to_text()


def test_phase_comparison_decomposes_complete_device_samples(
    tmp_path: Path,
) -> None:
    baseline = make_run(
        [snapshot(segment(active=100)), snapshot(segment(active=100))],
        name="baseline",
        labels=("start", "end"),
        device_memory=[{0: (900, 1000)}, {0: (880, 1000)}],
    )
    candidate = make_run(
        [snapshot(segment(active=100)), snapshot(segment(active=100))],
        name="candidate",
        labels=("start", "end"),
        device_memory=[{0: (850, 1000)}, {0: (800, 1000)}],
    )
    report = compare_phases(
        baseline.between("start", "end"),
        candidate.between("start", "end"),
    )
    by_metric = {row.components.metric: row for row in report.device_decomposition}
    used = by_metric["used_bytes"].components
    assert used.start_gap_bytes == 50
    assert used.baseline_change_bytes == 20
    assert used.candidate_change_bytes == 50
    assert used.change_gap_bytes == 30
    assert used.end_gap_bytes == 80
    assert used.identity_holds
    assert by_metric["total_bytes"].components.end_gap_bytes == 0
    assert "device[0] -> device[0] used_bytes" in report.to_text()
    changed_text = report.to_text(include_unchanged=False)
    assert "device[0] -> device[0] used_bytes" in changed_text
    assert "device[0] -> device[0] total_bytes" not in changed_text

    paths = report.write(tmp_path / "phase")
    assert paths["devices"].name == "devices.csv"
    assert paths["device_decomposition"].name == "device_decomposition.csv"
    assert "Device Decomposition" in paths["html"].read_text(encoding="utf-8")


def test_phase_comparison_omits_incomplete_device_equations() -> None:
    baseline = make_run(
        [snapshot(segment(active=100)), snapshot(segment(active=100))],
        name="baseline",
        labels=("start", "end"),
        device_memory=[{0: (900, 1000)}, {}],
    )
    candidate = make_run(
        [snapshot(segment(active=100)), snapshot(segment(active=100))],
        name="candidate",
        labels=("start", "end"),
        device_memory=[{0: (850, 1000)}, {0: (800, 1000)}],
    )
    report = compare_phases(
        baseline.between("start", "end"),
        candidate.between("start", "end"),
    )
    assert report.device_decomposition == ()
    assert "no complete changed device equations" in report.to_text()


def test_independent_comparison_marks_missing_side() -> None:
    reference_values = iter(
        [
            snapshot(
                segment(active=10, device=0),
                segment(active=10, device=1, address=5000),
            )
        ]
    )
    reference_probe = MemoryProbe._from_snapshot_provider(
        lambda marker: next(reference_values),
        device_memory_provider=lambda device: (90, 100),
    )
    candidate_values = iter([snapshot(segment(active=10, device=0))])
    candidate_probe = MemoryProbe._from_snapshot_provider(
        lambda marker: next(candidate_values),
        device_memory_provider=lambda device: (80, 100),
    )
    result = compare_snapshots(reference_probe.snapshot(), candidate_probe.snapshot())
    by_device = {
        row.reference_device_index
        if row.reference_device_index is not None
        else row.candidate_device_index: row
        for row in result.device_comparisons
    }
    assert set(by_device) == {0, 1}
    assert by_device[0].delta_used_bytes == 10
    assert by_device[1].candidate is None
    assert by_device[1].delta_used_bytes is None
    assert by_device[1].delta_cuda_allocator_residual_bytes is None
    assert by_device[1].changed

    text = result.to_text()
    assert "device[1] [reference_only]" in text
    assert "CUDA used: 10 B -> n/a" in text


@pytest.mark.parametrize(
    "device_memory",
    [
        [{0: (900, 1000)}, {}],
        [{}, {0: (900, 1000)}],
    ],
)
def test_paired_device_comparison_treats_sample_availability_as_a_change(
    device_memory: list[dict[int, tuple[int, int]]],
) -> None:
    run = make_run(
        [snapshot(segment(active=100)), snapshot(segment(active=100))],
        labels=("before", "after"),
        device_memory=device_memory,
    )

    result = run.compare("before", "after")
    (row,) = result.device_comparisons

    assert (row.reference is None) != (row.candidate is None)
    assert row.allocator_delta.changed is False
    assert row.changed is True
    assert "device[0]" in result.to_text(include_unchanged=False)


def test_unsampled_endpoints_render_allocator_only_tree() -> None:
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        labels=("before", "after"),
    )
    result = run.compare("before", "after")
    (row,) = result.device_comparisons
    assert row.reference is None and row.candidate is None
    assert row.reference_allocator.reserved_bytes == 10
    assert row.candidate_allocator.reserved_bytes == 20
    text = result.to_text()
    assert "CUDA scope:" not in text
    assert "CUDA used:" not in text
    assert "residual" not in text
    assert "  device[0]\n    allocator:" in text
    assert result.to_dict()["device_comparisons"][0]["reference"] is None


def _device_csv_columns() -> str:
    from torch_cudagraph_debug.memory_debug import MemoryStats

    stats_keys = list(MemoryStats().to_dict())
    columns = ["reference_device_index", "candidate_device_index", "match"]
    for prefix in ("reference", "candidate"):
        columns.extend(f"{prefix}_{name}_bytes" for name in ("free", "total", "used"))
        columns.append(f"{prefix}_cuda_allocator_residual_bytes")
        columns.extend(f"{prefix}_allocator_{key}" for key in stats_keys)
    columns.extend(f"delta_{name}_bytes" for name in ("used", "free", "total"))
    columns.append("delta_cuda_allocator_residual_bytes")
    columns.extend(f"delta_allocator_{key}" for key in stats_keys)
    return ",".join(columns)


def test_comparison_write_emits_devices_csv(tmp_path: Path) -> None:
    sampled = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        labels=("before", "after"),
        device_memory=[{0: (90, 100)}, {0: (70, 100)}],
    ).compare("before", "after")
    output = tmp_path / "report"
    paths = sampled.write(output)
    assert paths["devices"].name == "devices.csv"
    header = (output / "devices.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == _device_csv_columns()
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "<h2>Devices</h2>" in html
    assert "CUDA scope: device-wide" in html

    # Pool-only devices still produce device rows, so the CSV persists even
    # without CUDA Runtime samples; the sample columns stay empty.
    unsampled = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        labels=("before", "after"),
    ).compare("before", "after")
    unsampled_paths = unsampled.write(output, overwrite=True)
    assert unsampled_paths["devices"].name == "devices.csv"
    rows = (output / "devices.csv").read_text(encoding="utf-8").splitlines()
    assert rows[1].startswith("0,0,same_run,,,")


def test_timeline_device_entries_never_fabricate_deltas() -> None:
    run = make_run(
        [
            snapshot(segment(active=10)),
            snapshot(segment(active=20)),
            snapshot(segment(active=30)),
        ],
        labels=("first", "second", "third"),
        device_memory=[{0: (900, 1000)}, {}, {0: (600, 1000)}],
    )
    timeline = run.timeline()
    entries = {entry.point_label: entry for entry in timeline.device_entries}
    assert set(entries) == {"first", "second", "third"}
    assert entries["first"].delta_used_bytes is None
    assert entries["second"].sample is None
    assert entries["second"].delta_used_bytes is None
    # The previous point carries no sample, so the third point must not
    # report a delta computed against the first point or a fabricated zero.
    assert entries["third"].delta_used_bytes is None


def test_timeline_device_entries_track_cuda_allocator_residual_growth(
    tmp_path: Path,
) -> None:
    run = make_run(
        [
            snapshot(segment(active=100, total=100)),
            snapshot(segment(active=100, total=200)),
        ],
        labels=("before", "after"),
        device_memory=[{0: (900, 1000)}, {0: (600, 1000)}],
    )
    timeline = run.timeline()
    first, second = timeline.device_entries
    assert first.allocator_reserved_bytes == 100
    assert first.cuda_allocator_residual_bytes == 0
    assert second.allocator_reserved_bytes == 200
    assert second.delta_used_bytes == 300
    assert second.delta_allocator_reserved_bytes == 100
    assert second.delta_cuda_allocator_residual_bytes == 200

    text = timeline.to_text()
    assert "CUDA used: 400 B (+300 B)" in text
    assert "residual: 200 B (+200 B)" in text
    entry_payload = timeline.to_dict()["device_entries"][1]
    assert entry_payload["sample_delta"] == {
        "used_bytes": 300,
        "free_bytes": -300,
        "total_bytes": 0,
        "cuda_allocator_residual_bytes": 200,
    }
    assert entry_payload["allocator_delta"]["reserved_bytes"] == 100
    output = tmp_path / "timeline"
    paths = timeline.write(output)
    assert paths["devices"].name == "devices.csv"


def test_device_mapping_pairs_cross_device_default_pools() -> None:
    reference_probe = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(segment(active=10, device=0))
    )
    candidate_probe = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(segment(active=30, device=1))
    )

    result = compare_snapshots(
        reference_probe.snapshot(),
        candidate_probe.snapshot(),
        device_mapping={0: 1},
    )

    (device,) = result.device_comparisons
    assert (device.reference_device_index, device.candidate_device_index) == (0, 1)
    assert device.match == "mapped"
    (pool,) = result.pool_comparisons
    assert pool.reference_key == MemoryPoolKey(0, (0, 0))
    assert pool.candidate_key == MemoryPoolKey(1, (0, 0))
    assert pool.match == "default"


def test_pool_mapping_derives_device_pair_and_rejects_conflicts() -> None:
    reference_probe = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(segment(active=10, device=0))
    )
    candidate_probe = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(
            segment(active=20, device=1),
            segment(active=30, device=2, address=5000),
        )
    )
    reference = reference_probe.snapshot()
    candidate = candidate_probe.snapshot()

    derived = compare_snapshots(
        reference,
        candidate,
        pool_mapping={MemoryPoolKey(0, (0, 0)): MemoryPoolKey(1, (0, 0))},
    )
    assert derived.device_comparisons[0].match == "pool_mapping"
    assert (
        derived.device_comparisons[0].reference_device_index,
        derived.device_comparisons[0].candidate_device_index,
    ) == (0, 1)

    with pytest.raises(ValueError, match="conflicting with device"):
        compare_snapshots(
            reference,
            candidate,
            device_mapping={0: 1},
            pool_mapping={MemoryPoolKey(0, (0, 0)): MemoryPoolKey(2, (0, 0))},
        )


def test_device_mapping_is_one_to_one() -> None:
    reference_probe = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(
            segment(active=10, device=0),
            segment(active=20, device=1, address=5000),
        )
    )
    candidate_probe = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(segment(active=30, device=2))
    )
    with pytest.raises(ValueError, match="candidate device 2"):
        compare_snapshots(
            reference_probe.snapshot(),
            candidate_probe.snapshot(),
            device_mapping={0: 2, 1: 2},
        )


def test_phase_device_mapping_retains_both_device_indices() -> None:
    baseline = make_run(
        [
            snapshot(segment(active=10, device=0)),
            snapshot(segment(active=10, device=0)),
        ],
        name="baseline",
        labels=("start", "end"),
        device_memory=[{0: (90, 100)}, {0: (80, 100)}],
    )
    candidate = make_run(
        [
            snapshot(segment(active=10, device=1)),
            snapshot(segment(active=10, device=1)),
        ],
        name="candidate",
        labels=("start", "end"),
        device_memory=[{1: (70, 100)}, {1: (50, 100)}],
    )

    report = compare_phases(
        baseline.between("start", "end"),
        candidate.between("start", "end"),
        device_mapping={0: 1},
    )

    assert report.device_decomposition
    assert {
        (row.baseline_device_index, row.candidate_device_index)
        for row in report.device_decomposition
    } == {(0, 1)}


def test_cuda_allocator_residual_is_signed_without_warning() -> None:
    run = make_run(
        [
            snapshot(segment(active=100, total=100)),
            snapshot(segment(active=80, total=80)),
        ],
        labels=("before", "after"),
        device_memory=[{0: (950, 1000)}, {0: (960, 1000)}],
    )

    result = run.compare("before", "after")
    (row,) = result.device_comparisons
    assert row.reference_cuda_allocator_residual_bytes == -50
    assert row.candidate_cuda_allocator_residual_bytes == -40
    assert row.delta_cuda_allocator_residual_bytes == 10
    assert "residual: -50 B -> -40 B (+10 B)" in result.to_text()
    assert not any("residual" in warning for warning in result.warnings)


def test_cuda_allocator_residual_can_decrease_with_external_usage() -> None:
    run = make_run(
        [
            snapshot(segment(active=100, total=100)),
            snapshot(segment(active=100, total=100)),
        ],
        labels=("before", "after"),
        device_memory=[{0: (700, 1000)}, {0: (800, 1000)}],
    )
    (row,) = run.compare("before", "after").device_comparisons
    assert row.reference_cuda_allocator_residual_bytes == 200
    assert row.candidate_cuda_allocator_residual_bytes == 100
    assert row.delta_cuda_allocator_residual_bytes == -100


def test_phase_components_validate_metric_and_integer_fields() -> None:
    with pytest.raises(ValueError, match="unknown phase metric"):
        MemoryPhaseComponents("unknown", 0, 0, 0, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="start_gap_bytes"):
        MemoryPhaseComponents("used_bytes", True, 0, 0, 0)
    with pytest.raises(ValueError, match="baseline_device_index"):
        MemoryDevicePhaseDecomposition(
            True, 0, MemoryPhaseComponents("used_bytes", 0, 0, 0, 0)
        )
