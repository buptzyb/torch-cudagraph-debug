from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from torch_cudagraph_debug.tensor_debug import TensorRecorder, TensorRun
from torch_cudagraph_debug.tensor_debug.runs import (
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
) -> TensorRun:
    recorder = TensorRecorder(
        execution="eager",
        name=name,
        bundle_dir=bundle_dir,
        payload=default_payload,
        rank=0,
        world_size=1,
        run_metadata={"test": name},
    )
    for label, values in points:
        counts: dict[str, int] = {}
        pending = []
        for order, (probe_name, tensor, payload) in enumerate(values):
            invocation = counts.get(probe_name, 0)
            counts[probe_name] = invocation + 1
            pending.append(
                _PendingObservation(
                    order=order,
                    probe_name=probe_name,
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
