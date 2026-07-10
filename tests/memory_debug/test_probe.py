from __future__ import annotations

from dataclasses import replace

import pytest

from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryDebugError,
    MemoryHistoryBoundaryError,
    MemoryOwnershipError,
    MemoryPoolKey,
    MemoryProbe,
    compare_snapshots,
)

from ._helpers import event, segment, snapshot


def test_probe_snapshot_discovers_all_allocator_scopes() -> None:
    values = iter(
        [
            snapshot(
                segment(active=10, pool=(0, 0), stream=7),
                segment(active=20, pool=(1, 0), stream=7, address=2000),
            )
        ]
    )
    probe = MemoryProbe._from_snapshot_provider(lambda marker: next(values))

    captured = probe.snapshot()

    assert captured.snapshot_index == 0
    assert captured.probe_name == "memory"
    assert len(captured.observations) == 2
    assert captured.pool_stats[MemoryPoolKey(0, (0, 0))].allocated_bytes == 10
    assert captured.pool_stats[MemoryPoolKey(0, (1, 0))].allocated_bytes == 20
    assert captured.raw_snapshot()["segments"]
    assert not hasattr(captured, "run_id")


def test_same_probe_compare_supports_event_attribution() -> None:
    markers: list[str] = []

    def provider(marker: str) -> dict[str, object]:
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(segment(active=10))
        return snapshot(
            segment(active=10),
            segment(active=4, address=1010),
            traces=[
                [
                    event("snapshot", marker=markers[0]),
                    event("alloc", address=1010, size=4),
                    event("snapshot", marker=markers[1]),
                ]
            ],
        )

    probe = MemoryProbe._from_snapshot_provider(provider)
    before = probe.snapshot()
    after = probe.snapshot()

    comparison = probe.compare(
        before,
        after,
        attribution=MemoryAttributionOptions(events=True),
    )

    assert comparison.attribution_status.events.available is True
    assert comparison.attribution_status.events.complete is True
    assert comparison.allocator_events
    assert all(item.match == "same_probe" for item in comparison.pool_comparisons)
    assert comparison.to_dict()["kind"] == "snapshot-comparison"
    assert "same probe" in comparison.to_text()

    lifetime_comparison = probe.compare(
        before,
        after,
        attribution=MemoryAttributionOptions(lifetimes=True, events=True),
    )
    lifetimes = lifetime_comparison.allocation_lifetimes
    assert lifetimes is not None
    assert lifetimes.source_kind == "probe"
    assert lifetimes.source_id == before.probe_id
    assert not hasattr(lifetimes, "run")
    assert lifetimes.to_dict()["source"]["kind"] == "probe"


@pytest.mark.parametrize("missing_snapshot_index", [0, 1])
@pytest.mark.parametrize(
    "attribution",
    [
        pytest.param(MemoryAttributionOptions(events=True), id="events"),
        pytest.param(MemoryAttributionOptions(lifetimes=True), id="lifetimes"),
    ],
)
def test_same_probe_compare_reports_unrecorded_boundary(
    missing_snapshot_index: int,
    attribution: MemoryAttributionOptions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markers: list[str] = []

    def provider(marker: str) -> dict[str, object]:
        markers.append(marker)
        return snapshot(
            segment(active=10),
            traces=[[event("snapshot", marker=recorded) for recorded in markers]],
        )

    probe = MemoryProbe._from_snapshot_provider(provider)
    original_capture = probe._collector.capture
    capture_index = 0

    def capture(marker: str, **kwargs: object) -> object:
        nonlocal capture_index
        result = original_capture(marker, **kwargs)
        current_index = capture_index
        capture_index += 1
        return replace(
            result,
            boundary_recorded=current_index != missing_snapshot_index,
        )

    monkeypatch.setattr(probe._collector, "capture", capture)
    snapshots = (probe.snapshot(), probe.snapshot())

    assert snapshots[missing_snapshot_index]._boundary_recorded is False
    with pytest.raises(MemoryHistoryBoundaryError, match="boundar"):
        probe.compare(snapshots[0], snapshots[1], attribution=attribution)


def test_compare_snapshots_supports_independent_probes_without_history() -> None:
    reference = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(segment(active=10))
    ).snapshot()
    candidate = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(
            segment(active=12),
            segment(active=4, pool=(1, 0), address=2000),
        )
    ).snapshot()

    comparison = compare_snapshots(reference, candidate)

    assert comparison.pool_comparisons[0].delta.allocated_bytes == 2
    assert "cross probe" in comparison.to_text()
    assert any("independent probes" in item for item in comparison.warnings)
    with pytest.raises(MemoryDebugError, match="independent probes"):
        compare_snapshots(
            reference,
            candidate,
            attribution=MemoryAttributionOptions(events=True),
        )


def test_probe_compare_validates_ownership_and_order() -> None:
    values = iter(
        [
            snapshot(segment(active=10)),
            snapshot(segment(active=11)),
        ]
    )
    probe = MemoryProbe._from_snapshot_provider(lambda marker: next(values))
    before = probe.snapshot()
    after = probe.snapshot()
    foreign = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(segment(active=12))
    ).snapshot()

    with pytest.raises(MemoryOwnershipError, match="does not belong"):
        probe.compare(before, foreign)
    with pytest.raises(ValueError, match="must follow"):
        probe.compare(after, before)


def test_probe_snapshot_override_reaches_collector() -> None:
    seen: list[str] = []
    probe = MemoryProbe._from_snapshot_provider(
        lambda marker: seen.append(marker) or snapshot()
    )

    captured = probe.snapshot(synchronize=False)

    assert seen == [captured.boundary_marker]


def test_probe_snapshot_rejects_nonboolean_boundary_status() -> None:
    probe = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(segment(active=10))
    )
    captured = probe.snapshot()

    with pytest.raises(TypeError, match="boundary status must be boolean"):
        replace(captured, _boundary_recorded=1)


def test_probe_snapshot_rejects_noncontiguous_observation_order() -> None:
    probe = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(segment(active=10))
    )
    captured = probe.snapshot()
    invalid = replace(captured.observations[0], order=1)

    with pytest.raises(ValueError, match="observations must be contiguous"):
        replace(captured, observations=(invalid,))


def test_probe_snapshot_rejects_duplicate_observation_keys() -> None:
    probe = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(segment(active=10))
    )
    captured = probe.snapshot()
    first = captured.observations[0]
    duplicate = replace(first, order=1)

    with pytest.raises(ValueError, match="keys must be unique"):
        replace(captured, observations=(first, duplicate))


def test_probe_selects_and_separates_multiple_devices() -> None:
    device0 = segment(active=10, address=1000)
    device0["device"] = 0
    device1 = segment(active=20, address=2000)
    device1["device"] = 1
    value = snapshot(device0, device1)

    all_devices = MemoryProbe._from_snapshot_provider(
        lambda marker: value,
        devices="all",
    )
    all_snapshot = all_devices.snapshot()
    assert all_devices.devices == (0, 1)
    assert set(all_snapshot.pool_stats) == {
        MemoryPoolKey(0, (0, 0)),
        MemoryPoolKey(1, (0, 0)),
    }

    device1_only = MemoryProbe._from_snapshot_provider(
        lambda marker: value,
        devices=[1],
    )
    selected = device1_only.snapshot()
    assert device1_only.devices == (1,)
    assert set(selected.pool_stats) == {MemoryPoolKey(1, (0, 0))}
    assert selected.allocator_scope_stats["all"].active_bytes == 20


def test_default_probe_delays_device_binding_after_empty_provider_snapshot() -> None:
    device1 = segment(active=20, address=2000)
    device1["device"] = 1
    values = iter((snapshot(), snapshot(device1)))
    probe = MemoryProbe._from_snapshot_provider(lambda marker: next(values))

    first = probe.snapshot()
    second = probe.snapshot()

    assert first.observations == ()
    assert probe.devices == (1,)
    assert set(second.pool_stats) == {MemoryPoolKey(1, (0, 0))}


def test_filtered_deviceless_segments_are_reported() -> None:
    orphan = segment(active=64, address=1000)
    del orphan["device"]
    kept = segment(active=128, address=2000, device=1)
    probe = MemoryProbe._from_snapshot_provider(
        lambda marker: snapshot(orphan, kept),
        devices=[1],
    )

    captured = probe.snapshot()

    assert set(captured.pool_stats) == {MemoryPoolKey(1, (0, 0))}
    # The dropped segment lacked a device field; its exclusion must be
    # disclosed, not silent.
    assert any("missing 'device'" in warning for warning in captured.warnings)


def test_provider_probe_adopts_devices_that_appear_later() -> None:
    device1 = segment(active=4096, address=2000, device=1)
    values = iter(
        (
            snapshot(segment(active=100, address=1000)),
            snapshot(segment(active=100, address=1000), device1),
        )
    )
    probe = MemoryProbe._from_snapshot_provider(lambda marker: next(values))

    first = probe.snapshot()
    second = probe.snapshot()

    assert set(first.pool_stats) == {MemoryPoolKey(0, (0, 0))}
    # A device first touched after binding must join the selection instead
    # of silently disappearing from every later capture.
    assert MemoryPoolKey(1, (0, 0)) in second.pool_stats
    assert second.pool_stats[MemoryPoolKey(1, (0, 0))].allocated_bytes == 4096
    assert any("device" in warning for warning in second.warnings)


def test_all_devices_probe_delays_binding_after_empty_provider_snapshot() -> None:
    device1 = segment(active=20, address=2000)
    device1["device"] = 1
    values = iter((snapshot(), snapshot(device1)))
    probe = MemoryProbe._from_snapshot_provider(
        lambda marker: next(values), devices="all"
    )

    first = probe.snapshot()
    second = probe.snapshot()

    assert first.observations == ()
    assert probe.devices == (1,)
    assert set(second.pool_stats) == {MemoryPoolKey(1, (0, 0))}


def test_memory_probe_name_must_be_a_non_empty_string() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        MemoryProbe(1)  # type: ignore[arg-type]
