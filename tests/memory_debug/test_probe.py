from __future__ import annotations

from dataclasses import replace

import pytest

from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryDebugError,
    MemoryOwnershipError,
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

    assert captured.index == 0
    assert captured.probe_name == "memory"
    assert len(captured.observations) == 2
    assert captured.pool_stats[(0, 0)].allocated_bytes == 10
    assert captured.pool_stats[(1, 0)].allocated_bytes == 20
    assert captured.raw_snapshot()["segments"]
    assert not hasattr(captured, "run_id")


def test_same_probe_compare_supports_event_attribution() -> None:
    markers: list[str] = []

    def provider(marker: str) -> dict[str, object]:
        markers.append(marker)
        if len(markers) == 1:
            return snapshot(segment(active=10))
        return snapshot(
            segment(active=14),
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
        attribution=MemoryAttributionOptions(events=True, on_missing="error"),
    )

    assert comparison.events_available is True
    assert comparison.events_complete is True
    assert comparison.allocator_events
    assert all(item.match == "same_probe" for item in comparison.pool_comparisons)
    assert comparison.to_dict()["kind"] == "snapshot-comparison"
    assert "same probe" in comparison.to_text()

    lifetime_comparison = probe.compare(
        before,
        after,
        attribution=MemoryAttributionOptions(lifetimes=True),
    )
    lifetimes = lifetime_comparison.allocation_lifetimes
    assert lifetimes is not None
    assert lifetimes.source_kind == "probe"
    assert lifetimes.source_id == before.probe_id
    assert not hasattr(lifetimes, "run")
    assert lifetimes.to_dict()["source"]["kind"] == "probe"


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
