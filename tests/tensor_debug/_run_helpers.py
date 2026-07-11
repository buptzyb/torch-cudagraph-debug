from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from torch_cudagraph_debug.tensor_debug import (
    TensorProbeSnapshot,
    TensorRecorder,
    TensorRun,
)
from torch_cudagraph_debug.tensor_debug.recording import (
    PayloadKind,
    _PendingObservation,
)

ObservationInput = tuple[str, torch.Tensor, PayloadKind]


def make_tensor_run(
    points: Sequence[tuple[str, Sequence[ObservationInput]]],
    *,
    name: str = "run",
    bundle_dir: Path | None = None,
    default_payload: PayloadKind = "full",
    rank: int = 0,
    group_id: str | None = None,
    world_size: int = 1,
) -> TensorRun:
    recorder = TensorRecorder(
        execution="eager",
        name=name,
        bundle_dir=bundle_dir,
        payload=default_payload,
        rank=rank,
        group_id=group_id,
        world_size=world_size,
        run_metadata={"test": name},
    )
    for label, values in points:
        counts: dict[str, int] = {}
        pending = []
        for order, (observation_name, tensor, payload) in enumerate(values):
            invocation = counts.get(observation_name, 0)
            counts[observation_name] = invocation + 1
            pending.append(
                _PendingObservation(
                    order=order,
                    name=observation_name,
                    invocation_index=invocation,
                    replay_index=None,
                    shape=tuple(tensor.shape),
                    stride=tuple(tensor.stride()),
                    dtype=tensor.dtype,
                    source_device="cuda:0",
                    payload=payload,
                    tensor=tensor.detach().contiguous().cpu(),
                )
            )
        recorder._commit_point(label, {}, pending)
    return recorder.finish()


def make_probe_snapshot(
    tensor: torch.Tensor,
    *,
    probe_name: str = "mid",
    replay_index: int = 1,
    snapshot_index: int | None = None,
    probe_id: str = "test-probe",
) -> TensorProbeSnapshot:
    run = make_tensor_run(
        [("snapshot", [(probe_name, tensor, "full")])],
        name="probe-snapshot",
    )
    return TensorProbeSnapshot(
        probe_id=probe_id,
        probe_name=probe_name,
        snapshot_index=(replay_index if snapshot_index is None else snapshot_index),
        replay_index=replay_index,
        timestamp=0.0,
        observations=run.points[0].observations,
    )
