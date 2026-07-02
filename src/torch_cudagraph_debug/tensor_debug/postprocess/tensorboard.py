"""Python-side exporters for recorded tensor snapshots."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

import torch

from ..snapshots import TensorProbeSnapshot

StepSelector = int | Callable[[TensorProbeSnapshot], int]


class TensorBoardWriter(Protocol):
    """Minimal TensorBoard writer protocol used by the exporter helper."""

    def add_scalar(
        self, tag: str, scalar_value: object, global_step: int
    ) -> object: ...

    def add_histogram(
        self, tag: str, values: torch.Tensor, global_step: int
    ) -> object: ...


def export_snapshots_to_tensorboard(
    writer: TensorBoardWriter,
    snapshots: Iterable[TensorProbeSnapshot],
    *,
    tag_prefix: str = "",
    step: StepSelector | None = None,
    write_scalars: bool = True,
    write_histograms: bool = False,
) -> None:
    """Export recorded tensor snapshots to TensorBoard summaries.

    The caller owns synchronization, writer lifecycle, snapshot clearing, and any raw tensor
    persistence. This helper intentionally does not import TensorBoard; it accepts any object
    with ``add_scalar`` and ``add_histogram`` methods.
    """

    for snapshot in snapshots:
        global_step = _resolve_step(snapshot, step)
        multiple = len(snapshot.observations) > 1
        for observation in snapshot.observations:
            base_tag = f"{tag_prefix}{observation.probe_name}"
            if multiple:
                base_tag += f"/invocation_{observation.invocation_index}"
            tensor = observation.tensor().detach()
            numel = int(tensor.numel())

            if write_scalars:
                writer.add_scalar(f"{base_tag}/numel", numel, global_step)

            if numel == 0:
                continue

            values = tensor.to(dtype=torch.float32)
            flat = values.reshape(-1)

            if write_scalars:
                writer.add_scalar(f"{base_tag}/mean", flat.mean().item(), global_step)
                writer.add_scalar(
                    f"{base_tag}/std",
                    flat.std(unbiased=False).item(),
                    global_step,
                )
                writer.add_scalar(f"{base_tag}/min", flat.min().item(), global_step)
                writer.add_scalar(f"{base_tag}/max", flat.max().item(), global_step)
                writer.add_scalar(
                    f"{base_tag}/l2_norm",
                    torch.linalg.vector_norm(flat).item(),
                    global_step,
                )

            if write_histograms:
                writer.add_histogram(f"{base_tag}/hist", values, global_step)


def _resolve_step(snapshot: TensorProbeSnapshot, step: StepSelector | None) -> int:
    if step is None:
        return int(snapshot.replay_index)
    if callable(step):
        return int(step(snapshot))
    return int(step)
