from __future__ import annotations

import csv
import json
from pathlib import Path

from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    compare_phases,
)

from ._helpers import make_run, segment, snapshot


def test_comparison_owns_all_report_formats(tmp_path: Path) -> None:
    run = make_run(
        [
            snapshot(
                segment(active=10),
                segment(
                    active=20,
                    pool=(0, 2),
                    stream=7,
                    address=4000,
                ),
            ),
            snapshot(
                segment(active=15),
                segment(
                    active=20,
                    pool=(0, 2),
                    stream=7,
                    address=4000,
                ),
            ),
        ],
        labels=("before", "after"),
    )
    result = run.compare(
        "before",
        "after",
        attribution=MemoryAttributionOptions(stacks=True),
    )

    output = tmp_path / "comparison"
    paths = result.write(output)
    assert set(paths) == {
        "text",
        "json",
        "html",
        "allocator_scopes",
        "pools",
        "observations",
        "allocation_stack_comparisons",
    }
    for path in paths.values():
        assert path.is_file()

    payload = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "torch-cudagraph-debug/memory-report"
    assert payload["kind"] == "point-comparison"
    default = next(
        item
        for item in payload["pool_comparisons"]
        if item["candidate_pool_id"] == [0, 0]
    )
    assert default["reference"]["allocated_bytes"] == 10
    assert default["candidate"]["allocated_bytes"] == 15
    assert default["delta"]["allocated_bytes"] == 5
    assert "allocated: 10 B -> 15 B (delta +5 B)" in (output / "report.txt").read_text(
        encoding="utf-8"
    )

    with (output / "pools.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "reference_allocated_bytes" in rows[0]
    assert "candidate_allocated_bytes" in rows[0]
    assert "delta_allocated_bytes" in rows[0]


def test_only_changed_filters_pool_and_stream_csv(tmp_path: Path) -> None:
    run = make_run(
        [
            snapshot(
                segment(active=10),
                segment(
                    active=20,
                    pool=(0, 2),
                    stream=7,
                    address=4000,
                ),
            ),
            snapshot(
                segment(active=15),
                segment(
                    active=20,
                    pool=(0, 2),
                    stream=7,
                    address=4000,
                ),
            ),
        ],
        labels=("before", "after"),
    )
    result = run.compare("before", "after")
    output = tmp_path / "changed"
    result.write(output, include_unchanged=False)

    with (output / "pools.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["candidate_pool_id"] == "pool[0,0]"


def test_timeline_html_contains_charts_and_zero_delta_rows(
    tmp_path: Path,
) -> None:
    run = make_run(
        [
            snapshot(segment(active=10)),
            snapshot(segment(active=10)),
        ],
        labels=("start", "steady"),
    )
    timeline = run.timeline()
    output = tmp_path / "timeline"
    paths = timeline.write(output)

    assert set(paths) == {
        "text",
        "json",
        "html",
        "allocator_scopes",
        "pools",
        "observations",
    }
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "Allocated Memory" in html
    assert "<svg" in html
    with (output / "pools.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    steady = next(item for item in rows if item["point_label"] == "steady")
    assert steady["delta_allocated_bytes"] == "0"


def test_phase_report_writes_decomposition_and_component_tables(
    tmp_path: Path,
) -> None:
    baseline = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=30))],
        name="baseline",
        labels=("start", "end"),
    )
    candidate = make_run(
        [snapshot(segment(active=20)), snapshot(segment(active=35))],
        name="candidate",
        labels=("start", "end"),
    )
    phase = compare_phases(
        baseline.between("start", "end"),
        candidate.between("start", "end"),
    )

    paths = phase.write(tmp_path / "phase")
    assert "pool_decomposition" in paths
    assert paths["pool_decomposition"].name == "pool_decomposition.csv"
    assert paths["pools"].name == "pools.csv"
    assert phase.to_dict()["kind"] == "phase-comparison"
    content = paths["pool_decomposition"].read_text(encoding="utf-8")
    assert "identity_holds" in content
    assert "True" in content


def test_timeline_and_phase_propagate_component_warnings() -> None:
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        labels=("before", "after"),
    )
    timeline = run.timeline(attribution=MemoryAttributionOptions(events=True))
    assert "allocator event history" in timeline.to_text()
    assert any(
        "allocator event history" in item for item in timeline.to_dict()["warnings"]
    )

    baseline = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        name="baseline",
        labels=("start", "end"),
    )
    candidate = make_run(
        [snapshot(segment(active=12)), snapshot(segment(active=25))],
        name="candidate",
        labels=("start", "end"),
    )
    phase = compare_phases(
        baseline.between("start", "end"),
        candidate.between("start", "end"),
        attribution=MemoryAttributionOptions(events=True),
    )
    assert "allocator event history" in phase.to_text()
    assert any(
        "allocator event history" in item for item in phase.to_dict()["warnings"]
    )


def test_timeline_only_changed_filters_csv_and_html(tmp_path: Path) -> None:
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=10))],
        labels=("start", "steady"),
    )
    output = tmp_path / "timeline-changed"
    run.timeline().write(output, include_unchanged=False)

    with (output / "pools.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["point_label"] for row in rows} == {"start"}
    assert ">steady<" not in (output / "report.html").read_text(encoding="utf-8")
