"""Decomposition-tree text rendering: grammar, pruning, depth, and identities."""

from __future__ import annotations

from pathlib import Path

import pytest

from torch_cudagraph_debug.memory_debug import (
    MemoryPoolKey,
    MemoryProbe,
    compare_points,
    compare_snapshots,
)

from ._helpers import make_run, segment, snapshot


def _sampled_run():
    return make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=30))],
        labels=("before", "after"),
        device_memory=[{0: (60, 100)}, {0: (20, 100)}],
    )


def test_full_tree_matches_frozen_grammar() -> None:
    text = _sampled_run().compare("before", "after").to_text()
    assert text == (
        "Memory comparison 'before' -> 'after' (same run)\n"
        "  address lifecycle: exact\n"
        "  CUDA scope: device-wide, includes other processes; "
        "residual = CUDA used - allocator reserved. CUDA and allocator "
        "measurements are consecutive, not atomic.\n"
        "\n"
        "  device[0]\n"
        "    CUDA total: 100 B -> 100 B (0 B)\n"
        "    CUDA used: 40 B -> 80 B (+40 B)\n"
        "      residual: 30 B -> 50 B (+20 B)\n"
        "      allocator:\n"
        "        reserved: 10 B -> 30 B (+20 B)\n"
        "        allocated: 10 B -> 30 B (+20 B), "
        "active: 10 B -> 30 B (+20 B), requested: 10 B -> 30 B (+20 B)\n"
        "        pool[0,0] (default)\n"
        "          reserved: 10 B -> 30 B (+20 B)\n"
        "          allocated: 10 B -> 30 B (+20 B), "
        "active: 10 B -> 30 B (+20 B), requested: 10 B -> 30 B (+20 B)\n"
        "          lifecycle: new segment=30 B, removed segment=10 B, "
        "newly active=30 B, became inactive=10 B\n"
        "          stream[0]\n"
        "            reserved: 10 B -> 30 B (+20 B)\n"
        "            allocated: 10 B -> 30 B (+20 B), "
        "active: 10 B -> 30 B (+20 B), requested: 10 B -> 30 B (+20 B)\n"
        "            lifecycle: new segment=30 B, removed segment=10 B, "
        "newly active=30 B, became inactive=10 B"
    )


def test_decomposition_sums_hold_at_every_level() -> None:
    run = make_run(
        [
            snapshot(
                segment(active=10),
                segment(active=20, pool=(1, 0), address=5000),
                segment(active=5, pool=(1, 0), stream=7, address=9000),
            ),
            snapshot(
                segment(active=10),
                segment(active=40, pool=(1, 0), address=5000),
                segment(active=5, pool=(1, 0), stream=7, address=9000),
            ),
        ],
        labels=("before", "after"),
        device_memory=[{0: (900, 1000)}, {0: (800, 1000)}],
    )
    result = run.compare("before", "after")
    (device_row,) = result.device_comparisons
    assert device_row.candidate is not None
    assert device_row.candidate.used_bytes == (
        device_row.candidate_cuda_allocator_residual_bytes
        + device_row.candidate_allocator.reserved_bytes
    )
    assert device_row.candidate_allocator.reserved_bytes == sum(
        item.candidate.reserved_bytes for item in result.pool_comparisons
    )
    for pool_row in result.pool_comparisons:
        streams = [
            item
            for item in result.observation_comparisons
            if (item.candidate_key or item.reference_key).pool_key
            == (pool_row.candidate_key or pool_row.reference_key)
        ]
        assert pool_row.candidate.reserved_bytes == sum(
            item.candidate.reserved_bytes for item in streams
        )


def test_pruning_collapses_static_subtrees() -> None:
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=10))],
        labels=("before", "after"),
        device_memory=[{0: (60, 100)}, {0: (30, 100)}],
    )
    text = run.compare("before", "after").to_text(include_unchanged=False)
    assert text == (
        "Memory comparison 'before' -> 'after' (same run)\n"
        "  address lifecycle: exact\n"
        "  CUDA scope: device-wide, includes other processes; "
        "residual = CUDA used - allocator reserved. CUDA and allocator "
        "measurements are consecutive, not atomic.\n"
        "\n"
        "  device[0]\n"
        "    CUDA used: 40 B -> 70 B (+30 B)\n"
        "      residual: 30 B -> 60 B (+30 B)"
    )


def test_pruning_keeps_parent_context_for_offsetting_churn() -> None:
    run = make_run(
        [
            snapshot(
                segment(active=10),
                segment(active=20, pool=(1, 0), address=5000),
            ),
            snapshot(
                segment(active=20),
                segment(active=10, pool=(1, 0), address=5000),
            ),
        ],
        labels=("before", "after"),
    )
    text = run.compare("before", "after").to_text(include_unchanged=False)
    # The two pools swap 10 B, so the device rollup nets to zero yet must
    # remain as path context above the changed pools.
    assert "reserved: 30 B -> 30 B (0 B)" in text
    assert "pool[0,0] (default)" in text
    assert "pool[1,0] (private)" in text


def test_pruning_reports_no_changed_devices() -> None:
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=10))],
        labels=("before", "after"),
    )
    text = run.compare("before", "after").to_text(include_unchanged=False)
    assert text.endswith("no changed devices")


def test_independent_comparison_renders_both_one_sided_streams() -> None:
    reference = make_run([snapshot(segment(active=512, stream=7))], labels=("p",))
    candidate = make_run([snapshot(segment(active=2048, stream=8))], labels=("p",))
    result = compare_points(reference.point("p"), candidate.point("p"))
    rows = result.observation_comparison_rows(include_unchanged=True)
    assert len(rows) == len(result.observation_comparisons) == 2
    text = result.to_text(include_unchanged=True)
    assert "stream[7] [reference_only]" in text
    assert "stream[8] [candidate_only]" in text
    (pool_row,) = result.pool_comparison_rows(include_unchanged=True)
    assert pool_row["delta_reserved_bytes"] == sum(
        row["delta_reserved_bytes"] for row in rows
    )


def test_depth_pruning_keeps_devices_with_changes_below_the_cutoff() -> None:
    reference = make_run([snapshot(segment(active=512, stream=7))], labels=("p",))
    candidate = make_run([snapshot(segment(active=512, stream=8))], labels=("p",))
    result = compare_points(reference.point("p"), candidate.point("p"))
    for depth in ("pool", "device"):
        text = result.to_text(include_unchanged=False, depth=depth)
        assert "no changed devices" not in text
        assert "device[0]" in text
        assert "pool[0,0] (default)" in text or depth == "device"


def test_depth_truncates_containers() -> None:
    result = _sampled_run().compare("before", "after")
    device_text = result.to_text(depth="device")
    assert "allocator:" in device_text
    assert "pool[0,0]" not in device_text
    pool_text = result.to_text(depth="pool")
    assert "pool[0,0] (default)" in pool_text
    assert "stream[0]" not in pool_text
    assert result.to_text(depth="stream") == result.to_text()
    with pytest.raises(ValueError, match="depth must be one of"):
        result.to_text(depth="bogus")


def test_invalid_depth_raises_before_directory_creation(tmp_path: Path) -> None:
    result = _sampled_run().compare("before", "after")
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="depth must be one of"):
        result.write(output, depth="bogus")
    assert not output.exists()


def test_same_identity_comparison_renders_no_match_tags() -> None:
    text = _sampled_run().compare("before", "after").to_text()
    assert "[same_run]" not in text
    assert "[same_probe]" not in text
    assert "[default]" not in text


def test_cross_device_mapping_renders_device_pair_node() -> None:
    reference_values = iter([snapshot(segment(active=10, device=0))])
    reference_probe = MemoryProbe._from_snapshot_provider(
        lambda marker: next(reference_values)
    )
    candidate_values = iter([snapshot(segment(active=30, device=1))])
    candidate_probe = MemoryProbe._from_snapshot_provider(
        lambda marker: next(candidate_values)
    )
    result = compare_snapshots(
        reference_probe.snapshot(),
        candidate_probe.snapshot(),
        pool_mapping={MemoryPoolKey(0, (0, 0)): MemoryPoolKey(1, (0, 0))},
    )
    text = result.to_text()
    assert "device[0] -> device[1] [pool_mapping]" in text
    assert text.count("  device[") == 1
    assert "pool[0,0] (default)" in text


def test_timeline_tree_prunes_and_degrades() -> None:
    run = make_run(
        [
            snapshot(segment(active=10)),
            snapshot(segment(active=10)),
        ],
        labels=("first", "steady"),
    )
    full = run.timeline().to_text()
    assert "CUDA scope:" not in full
    assert "CUDA used:" not in full
    assert "device[0]" in full
    assert "allocator:" in full
    # First-point deltas are None and count as unchanged, matching the
    # pre-tree filter semantics: a fully steady run filters to the header.
    filtered = run.timeline().to_text(include_unchanged=False)
    assert filtered == "CUDA allocator memory timeline 'run'"


def test_no_delta_word_remains_in_rendered_text() -> None:
    run = make_run(
        [
            snapshot(
                segment(active=10),
                segment(active=20, pool=(1, 0), address=5000),
            ),
            snapshot(
                segment(active=25),
                segment(active=40, pool=(1, 0), address=5000),
            ),
        ],
        labels=("before", "after"),
        device_memory=[{0: (900, 1000)}, {0: (700, 1000)}],
    )
    assert "(delta " not in run.compare("before", "after").to_text()
    assert "(delta " not in run.timeline().to_text()


def test_bottom_up_pruning_matches_text_html_and_csv(tmp_path: Path) -> None:
    run = make_run(
        [
            snapshot(
                segment(active=10),
                segment(active=20, pool=(1, 0), address=5000),
            ),
            snapshot(
                segment(active=20),
                segment(active=10, pool=(1, 0), address=5000),
            ),
        ],
        labels=("before", "after"),
    )
    result = run.compare("before", "after")
    output = tmp_path / "report"
    result.write(output, include_unchanged=False)

    html = (output / "report.html").read_text(encoding="utf-8")
    assert "allocator:" in html
    assert "pool[0,0] (default)" in html
    assert "pool[1,0] (private)" in html
    assert "No changed devices" not in html
    assert len((output / "devices.csv").read_text(encoding="utf-8").splitlines()) == 2
    assert len((output / "pools.csv").read_text(encoding="utf-8").splitlines()) == 3


def test_html_depth_uses_the_same_tree() -> None:
    result = _sampled_run().compare("before", "after")
    device_html = result.to_html(depth="device")
    assert "allocator:" in device_html
    assert "pool[0,0]" not in device_html
    pool_html = result.to_html(depth="pool")
    assert "pool[0,0] (default)" in pool_html
    assert "stream[0]" not in pool_html
    assert "stream[0]" in result.to_html(depth="stream")


def test_allocator_only_html_does_not_claim_device_wide_scope() -> None:
    result = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        labels=("before", "after"),
    ).compare("before", "after")
    html = result.to_html()
    assert "<h2>Devices</h2>" in html
    assert "CUDA scope: device-wide" not in html


def test_cross_device_html_has_one_device_root() -> None:
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
    assert result.to_html().count('data-kind="device"') == 1
