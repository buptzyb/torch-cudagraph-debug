"""Python-side exporters for recorded tensor snapshots."""

from __future__ import annotations

from collections import Counter
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

    The caller owns synchronization, writer lifecycle, snapshot retention, and
    any raw tensor persistence. This helper intentionally does not import
    TensorBoard; it accepts any object with ``add_scalar`` and
    ``add_histogram`` methods.

    Scalars come from each snapshot observation's float64 ``summary``; undefined
    summary statistics are skipped. Histograms require a full payload and pass
    the writer a detached copy, so summary-only observations export scalars and
    skip the histogram.

    ``step=None`` uses each snapshot's ``replay_index`` as the global step;
    an ``int`` or a per-snapshot callable overrides it. Observations with
    ``numel == 0`` export only the ``numel`` scalar.
    """

    if type(write_scalars) is not bool or type(write_histograms) is not bool:
        raise TypeError("write_scalars and write_histograms must be booleans")
    if not write_scalars and not write_histograms:
        return

    for snapshot in snapshots:
        global_step = _resolve_step(snapshot, step)
        name_counts = Counter(item.name for item in snapshot.observations)
        for observation in snapshot.observations:
            base_tag = f"{tag_prefix}{observation.name}"
            if name_counts[observation.name] > 1:
                base_tag += f"/invocation_{observation.invocation_index}"
            numel = observation.summary.numel

            if write_scalars:
                writer.add_scalar(f"{base_tag}/numel", numel, global_step)

            if numel == 0:
                continue

            if write_scalars:
                summary = observation.summary
                scalars = (
                    ("mean", summary.mean),
                    ("std", summary.std),
                    ("min", summary.minimum),
                    ("max", summary.maximum),
                    ("l2_norm", summary.l2_norm),
                )
                for suffix, value in scalars:
                    if value is not None:
                        writer.add_scalar(f"{base_tag}/{suffix}", value, global_step)

            if write_histograms and observation.has_payload:
                values = observation._materialize_tensor().detach().clone()
                writer.add_histogram(
                    f"{base_tag}/hist",
                    values.to(dtype=torch.float32),
                    global_step,
                )


def _resolve_step(snapshot: TensorProbeSnapshot, step: StepSelector | None) -> int:
    if step is None:
        return snapshot.replay_index
    if callable(step):
        value = step(snapshot)
    else:
        value = step
    if type(value) is not int:
        raise TypeError("TensorBoard step must be an integer")
    return value
