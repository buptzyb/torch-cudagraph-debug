from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from torch_cudagraph_debug.memory_debug import (
    MemoryAttributionOptions,
    MemoryDisplayOptions,
    compare_phases,
)

from ._helpers import event, make_history_run, make_run, segment, snapshot


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
        if item["candidate_key"]
        == {
            "device_index": 0,
            "pool_id": [0, 0],
        }
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


def test_stream_only_stack_deltas_are_written_to_html_and_csv(
    tmp_path: Path,
) -> None:
    run = make_run(
        [
            snapshot(
                segment(active=10, stream=1, address=1000, frame="same.py"),
                segment(active=20, stream=2, address=2000, frame="same.py"),
            ),
            snapshot(
                segment(active=20, stream=1, address=3000, frame="same.py"),
                segment(active=10, stream=2, address=4000, frame="same.py"),
            ),
        ],
        labels=("before", "after"),
    )
    result = run.compare(
        "before",
        "after",
        attribution=MemoryAttributionOptions(stacks=True),
    )

    assert result.allocation_stack_comparisons == ()
    assert result.allocation_stack_observation_comparisons

    output = tmp_path / "stream-only-stacks"
    paths = result.write(output)

    assert "allocation_stack_comparisons" in paths
    html = paths["html"].read_text(encoding="utf-8")
    assert "Allocation Stacks" in html
    with paths["allocation_stack_comparisons"].open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["scope"] for row in rows} == {"observation"}


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
    assert rows[0]["candidate_key"] == "device[0]/pool[0,0]"


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
    assert "Active Memory" in html
    assert "Requested Memory" in html
    assert "<svg" in html
    with (output / "pools.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    steady = next(item for item in rows if item["point_label"] == "steady")
    assert steady["delta_allocated_bytes"] == "0"


def test_timeline_charts_ignore_include_unchanged_filter(tmp_path: Path) -> None:
    run = make_run(
        [
            snapshot(segment(active=1024)),
            snapshot(segment(active=4096)),
            snapshot(segment(active=4096)),
            snapshot(segment(active=8192)),
        ],
        labels=("a", "b", "c", "d"),
    )
    timeline = run.timeline()
    full = tmp_path / "full"
    filtered = tmp_path / "filtered"
    timeline.write(full)
    timeline.write(filtered, include_unchanged=False)

    def polylines(output: Path) -> list[str]:
        html = (output / "report.html").read_text(encoding="utf-8")
        return re.findall(r'<polyline[^>]*points="([^"]+)"', html)

    # Charts must always draw the true per-point series; the row filter is
    # a table concern and must not bend or drop chart lines.
    assert polylines(filtered) == polylines(full)
    assert all(line.count(" ") == 3 for line in polylines(filtered))


def test_timeline_svg_draws_one_labeled_line_per_pool(tmp_path: Path) -> None:
    run = make_run(
        [
            snapshot(
                segment(active=10),
                segment(active=20, pool=(0, 7), address=4000),
            ),
            snapshot(
                segment(active=30),
                segment(active=20, pool=(0, 7), address=4000),
            ),
        ],
        labels=("start", "end"),
    )
    timeline = run.timeline()
    output = tmp_path / "timeline"
    timeline.write(output)

    html = (output / "report.html").read_text(encoding="utf-8")
    # Four metric charts, each with one polyline per pool.
    assert html.count("<polyline") == 8
    assert "<title>device[0]/pool[0,0]</title>" in html
    assert "<title>device[0]/pool[0,7]</title>" in html


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
    def history_steps():
        return [
            ([segment(active=10)], []),
            (
                [segment(active=10)],
                [event("future_action", address=4000, size=1)],
            ),
        ]

    run = make_history_run(history_steps(), labels=("before", "after"))
    timeline = run.timeline(
        attribution=MemoryAttributionOptions(lifetimes=True, events=True)
    )
    assert "unknown allocator actions" in timeline.to_text()
    assert any(
        "unknown allocator actions" in item for item in timeline.to_dict()["warnings"]
    )

    baseline = make_history_run(
        history_steps(), name="baseline", labels=("start", "end")
    )
    candidate = make_history_run(
        history_steps(), name="candidate", labels=("start", "end")
    )
    phase = compare_phases(
        baseline.between("start", "end"),
        candidate.between("start", "end"),
        attribution=MemoryAttributionOptions(lifetimes=True, events=True),
    )
    assert "unknown allocator actions" in phase.to_text()
    assert any(
        "unknown allocator actions" in item for item in phase.to_dict()["warnings"]
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
    assert rows == []
    assert ">steady<" not in (output / "report.html").read_text(encoding="utf-8")


def test_text_reports_show_core_metrics_at_pool_stream_scope() -> None:
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=12, requested=11))],
        labels=("before", "after"),
    )

    comparison_text = run.compare("before", "after").to_text()
    timeline_text = run.timeline().to_text()

    assert "device/pool/stream observations:" in comparison_text
    assert "active: 10 B -> 12 B" in comparison_text
    assert "requested: 10 B -> 11 B" in comparison_text
    assert "active=12 B (delta +2 B)" in timeline_text
    assert "requested=11 B (delta +1 B)" in timeline_text


def test_structural_only_changes_remain_explainable_when_filtered() -> None:
    before = segment(active=10)
    after = segment(active=10)
    after["blocks"].append(
        {
            "address": 2000,
            "size": 0,
            "requested_size": 0,
            "state": "inactive",
            "frames": [],
        }
    )
    run = make_run(
        [snapshot(before), snapshot(after)],
        labels=("before", "structural"),
    )

    text = run.timeline().to_text(include_unchanged=False)

    assert "[1] structural" in text
    assert "blocks=2 (delta +1)" in text


def test_write_requires_explicit_overwrite_for_nonempty_directory(
    tmp_path: Path,
) -> None:
    run = make_run(
        [snapshot(segment(active=10)), snapshot(segment(active=20))],
        labels=("before", "after"),
    )
    output = tmp_path / "report"
    output.mkdir()
    stale = output / "events.csv"
    stale.write_text("stale", encoding="utf-8")
    unrelated = output / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    comparison = run.compare("before", "after")

    with pytest.raises(FileExistsError, match="not empty"):
        comparison.write(output)

    paths = comparison.write(output, overwrite=True)
    assert all(path.is_absolute() for path in paths.values())
    assert not stale.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_attribution_display_options_do_not_change_structured_results(
    tmp_path: Path,
) -> None:
    shared = {"filename": "shared.py", "line": 1, "name": "allocate"}
    left = {"filename": "left.py", "line": 2, "name": "forward"}
    right = {"filename": "right.py", "line": 3, "name": "forward"}

    def state(left_size: int, right_size: int):
        left_segment = segment(active=left_size, address=1000)
        right_segment = segment(active=right_size, address=2000)
        left_segment["blocks"][0]["frames"] = [shared, left]
        right_segment["blocks"][0]["frames"] = [shared, right]
        return snapshot(left_segment, right_segment)

    run = make_run(
        [state(10, 20), state(15, 30)],
        labels=("before", "after"),
    )
    result = run.compare(
        "before",
        "after",
        attribution=MemoryAttributionOptions(
            stacks=True,
            display=MemoryDisplayOptions(stack_depth=1, limit=1),
        ),
    )

    assert len(result.allocation_stack_comparisons) == 2
    structured_rows = result.to_dict()["allocation_stack_comparisons"]
    assert len(structured_rows) == 2
    assert all(row["stack_frames"] for row in structured_rows)
    assert all("stack_frames_json" not in row for row in structured_rows)
    compact = result.to_text()
    assert "showing 1 of 2" in compact
    assert "shared.py:1:allocate" in compact
    assert "left.py" not in compact
    assert "right.py" not in compact

    expanded = result.to_text(limit=2, stack_depth=2)
    assert "left.py:2:forward" in expanded
    assert "right.py:3:forward" in expanded
    assert len(result.to_dict()["allocation_stack_comparisons"]) == 2

    paths = result.write(tmp_path / "full-structured")
    with paths["allocation_stack_comparisons"].open(
        newline="", encoding="utf-8"
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len([row for row in rows if row["scope"] == "pool"]) == 2
    assert all(json.loads(row["stack_frames_json"]) for row in rows)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert len(payload["allocation_stack_comparisons"]) == 2
    html = paths["html"].read_text(encoding="utf-8")
    assert "Showing 2 of 4 rows." in html
    assert "shared.py:1:allocate" in html
    assert "left.py:2:forward" not in html
    assert "right.py:3:forward" not in html


def test_timeline_and_phase_propagate_attribution_display_options(
    tmp_path: Path,
) -> None:
    shared = {"filename": "shared.py", "line": 1, "name": "allocate"}
    left = {"filename": "left.py", "line": 2, "name": "forward"}
    right = {"filename": "right.py", "line": 3, "name": "forward"}

    def state(left_size: int, right_size: int):
        left_segment = segment(active=left_size, address=1000)
        right_segment = segment(active=right_size, address=2000)
        left_segment["blocks"][0]["frames"] = [shared, left]
        right_segment["blocks"][0]["frames"] = [shared, right]
        return snapshot(left_segment, right_segment)

    options = MemoryAttributionOptions(
        stacks=True, display=MemoryDisplayOptions(stack_depth=1, limit=1)
    )
    baseline = make_run(
        [state(10, 20), state(15, 30)],
        name="baseline",
        labels=("start", "end"),
    )
    candidate = make_run(
        [state(20, 40), state(30, 60)],
        name="candidate",
        labels=("start", "end"),
    )

    timeline = baseline.timeline(attribution=options)
    assert len(timeline.point_comparisons[0].allocation_stack_comparisons) == 2
    assert "showing 1 of 2" in timeline.to_text()
    assert "left.py" not in timeline.to_text()
    assert "left.py:2:forward" in timeline.to_text(limit=2, stack_depth=2)
    assert "Showing 2 of 4 rows." in timeline.to_html()

    phase = compare_phases(
        baseline.between("start", "end"),
        candidate.between("start", "end"),
        attribution=options,
    )
    assert len(phase.baseline_change.allocation_stack_comparisons) == 2
    assert "showing 1 of 2" in phase.to_text()
    assert "left.py" not in phase.to_text()
    assert "left.py:2:forward" in phase.to_text(limit=2, stack_depth=2)
    assert "Showing 2 of 4 rows." in phase.to_html()

    paths = phase.write(tmp_path / "phase", limit=2, stack_depth=2)
    assert "left.py:2:forward" in paths["text"].read_text(encoding="utf-8")
    assert "left.py:2:forward" in paths["html"].read_text(encoding="utf-8")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert len(payload["baseline_change"]["allocation_stack_comparisons"]) == 2


def test_reports_preserve_fx_frames_and_render_them_readably(
    tmp_path: Path,
) -> None:
    def fx_segment(address: int, active: int, node: str):
        value = segment(active=active, address=address)
        value["blocks"][0]["frames"] = [
            {
                "filename": "model.py",
                "line": 10,
                "name": "forward",
                "fx_node_op": "call_module",
                "fx_node_name": node,
                "fx_original_trace": f"model.{node}",
            }
        ]
        return value

    run = make_run(
        [
            snapshot(fx_segment(1000, 10, "left"), fx_segment(2000, 20, "right")),
            snapshot(fx_segment(1000, 15, "left"), fx_segment(2000, 30, "right")),
        ],
        labels=("before", "after"),
    )
    result = run.compare(
        "before",
        "after",
        attribution=MemoryAttributionOptions(
            stacks=True, display=MemoryDisplayOptions(limit=2)
        ),
    )

    text = result.to_text(stack_depth=1)
    assert 'fx_node_name="left"' in text
    assert 'fx_node_name="right"' in text
    payload = result.to_dict()
    assert {
        row["stack_frames"][0]["fx_node_name"]
        for row in payload["allocation_stack_comparisons"]
    } == {"left", "right"}

    paths = result.write(tmp_path / "fx-frames", stack_depth=1)
    html = paths["html"].read_text(encoding="utf-8")
    assert "fx_node_name=&quot;left&quot;" in html
    assert "fx_node_name=&quot;right&quot;" in html
    with paths["allocation_stack_comparisons"].open(
        newline="", encoding="utf-8"
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert {
        json.loads(row["stack_frames_json"])[0]["fx_node_name"] for row in rows
    } == {"left", "right"}


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("limit", 0, "limit must be an integer >= 1"),
        ("stack_depth", 0, "stack_depth must be an integer >= 1"),
    ),
)
def test_display_options_are_validated_before_render_or_write_side_effects(
    tmp_path: Path,
    keyword: str,
    value: int,
    message: str,
) -> None:
    run = make_run([snapshot(), snapshot()], labels=("before", "after"))
    result = run.compare("before", "after")
    options = {keyword: value}

    with pytest.raises(ValueError, match=message):
        result.to_text(**options)
    with pytest.raises(ValueError, match=message):
        result.to_html(**options)

    output = tmp_path / keyword
    with pytest.raises(ValueError, match=message):
        result.write(output, **options)
    assert not output.exists()


def test_timeline_cohort_chart_respects_display_limit() -> None:
    run = make_history_run(
        [
            (
                [
                    segment(active=0, total=64, address=1000),
                    segment(active=0, total=64, address=2000),
                ],
                [],
            ),
            (
                [
                    segment(active=64, address=1000, frame="first.py"),
                    segment(active=64, address=2000, frame="second.py"),
                ],
                [
                    event("alloc", address=1000, size=64, frame="first.py"),
                    event("alloc", address=2000, size=64, frame="second.py"),
                ],
            ),
        ],
        labels=("before", "after"),
    )
    timeline = run.timeline(
        attribution=MemoryAttributionOptions(
            lifetimes=True,
            display=MemoryDisplayOptions(limit=1),
        )
    )

    analysis = timeline.allocation_lifetimes
    assert analysis is not None
    assert len(analysis.cohorts) == 2
    first, second = (cohort.cohort_id for cohort in analysis.cohorts)
    html = timeline.to_html()
    assert first in html
    assert second not in html
    structured = timeline.to_dict()["allocation_lifetimes"]
    assert first in json.dumps(structured)
    assert second in json.dumps(structured)
