"""Atomic report-output helpers shared by tensor and memory analysis."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

_REPORT_ARTIFACT_FILENAMES = frozenset(
    {
        "report.txt",
        "report.json",
        "report.html",
        "observations.csv",
        "allocator_scopes.csv",
        "pools.csv",
        "devices.csv",
        "device_decomposition.csv",
        "rank_devices.csv",
        "rank_device_decomposition.csv",
        "allocation_stack_comparisons.csv",
        "events.csv",
        "cohorts.csv",
        "cohort_points.csv",
        "size_histograms.csv",
        "size_outcomes.csv",
        "birth_stacks.csv",
        "free_request_stacks.csv",
        "free_completion_stacks.csv",
        "pool_decomposition.csv",
        "allocator_scope_decomposition.csv",
        "rank_points.csv",
        "point_aggregates.csv",
        "rank_comparisons.csv",
        "rank_decomposition.csv",
        "rank_pool_decomposition.csv",
        "phase_aggregates.csv",
    }
)


def prepare_output_dir(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    root = Path(output_dir).resolve()
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f"report output is not a directory: {root}")
        nonempty = any(root.iterdir())
        if nonempty and not overwrite:
            raise FileExistsError(f"report directory is not empty: {root}")
        if nonempty:
            for name in _REPORT_ARTIFACT_FILENAMES:
                artifact = root / name
                if artifact.is_file() or artifact.is_symlink():
                    artifact.unlink()
    else:
        root.mkdir(parents=True)
    return root


def atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
    )


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    empty_fieldnames: Sequence[str] = (),
    extrasaction: str = "raise",
) -> Path:
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    if not fieldnames:
        fieldnames.extend(empty_fieldnames)

    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            if fieldnames:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                    extrasaction=extrasaction,
                )
                writer.writeheader()
                writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path
