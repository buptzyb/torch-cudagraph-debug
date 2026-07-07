from __future__ import annotations

import pytest
import torch

from torch_cudagraph_debug.tensor_debug import TensorProbeSnapshot
from torch_cudagraph_debug.tensor_debug.postprocess import (
    export_snapshots_to_tensorboard,
)

from ._run_helpers import make_probe_snapshot, make_tensor_run


class FakeWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, object, int]] = []
        self.histograms: list[tuple[str, torch.Tensor, int]] = []

    def add_scalar(self, tag: str, scalar_value: object, global_step: int) -> None:
        self.scalars.append((tag, scalar_value, global_step))

    def add_histogram(self, tag: str, values: torch.Tensor, global_step: int) -> None:
        self.histograms.append((tag, values, global_step))


def scalar_value(writer: FakeWriter, tag: str) -> object:
    matches = [value for actual_tag, value, _ in writer.scalars if actual_tag == tag]
    assert len(matches) == 1
    return matches[0]


def test_export_snapshots_writes_default_scalars_with_replay_step() -> None:
    writer = FakeWriter()
    snapshot = make_probe_snapshot(torch.tensor([1.0, 2.0, 3.0]), replay_index=7)

    export_snapshots_to_tensorboard(writer, [snapshot], tag_prefix="debug/")

    assert {step for _, _, step in writer.scalars} == {7}
    assert scalar_value(writer, "debug/mid/numel") == 3
    assert scalar_value(writer, "debug/mid/mean") == pytest.approx(2.0)
    assert scalar_value(writer, "debug/mid/std") == pytest.approx(0.81649658)
    assert scalar_value(writer, "debug/mid/min") == pytest.approx(1.0)
    assert scalar_value(writer, "debug/mid/max") == pytest.approx(3.0)
    assert scalar_value(writer, "debug/mid/l2_norm") == pytest.approx(3.74165738)
    assert writer.histograms == []


def test_export_snapshots_supports_fixed_and_callable_steps() -> None:
    fixed_writer = FakeWriter()
    callable_writer = FakeWriter()
    snapshot = make_probe_snapshot(
        torch.tensor([1, 2, 3], dtype=torch.int32), replay_index=7
    )

    export_snapshots_to_tensorboard(fixed_writer, [snapshot], step=123)
    export_snapshots_to_tensorboard(
        callable_writer,
        [snapshot],
        step=lambda item: item.replay_index + 1000,
    )

    assert {step for _, _, step in fixed_writer.scalars} == {123}
    assert {step for _, _, step in callable_writer.scalars} == {1007}


def test_export_snapshots_writes_histograms_only_when_enabled() -> None:
    writer = FakeWriter()
    snapshot = make_probe_snapshot(
        torch.tensor([[True, False], [True, True]]), replay_index=2
    )

    export_snapshots_to_tensorboard(writer, [snapshot], write_histograms=True)

    assert len(writer.histograms) == 1
    tag, values, step = writer.histograms[0]
    assert tag == "mid/hist"
    assert step == 2
    assert values.dtype == torch.float32
    assert torch.equal(values, torch.tensor([[1.0, 0.0], [1.0, 1.0]]))


def test_export_snapshots_can_disable_scalars() -> None:
    writer = FakeWriter()
    snapshot = make_probe_snapshot(torch.tensor([1.0, 2.0]), replay_index=2)

    export_snapshots_to_tensorboard(
        writer,
        [snapshot],
        write_scalars=False,
        write_histograms=True,
    )

    assert writer.scalars == []
    assert len(writer.histograms) == 1


def test_export_snapshots_empty_tensor_writes_only_numel() -> None:
    writer = FakeWriter()
    snapshot = make_probe_snapshot(torch.empty(0), probe_name="empty", replay_index=3)

    export_snapshots_to_tensorboard(writer, [snapshot], write_histograms=True)

    assert writer.scalars == [("empty/numel", 0, 3)]
    assert writer.histograms == []


def test_distinct_observation_names_do_not_get_invocation_zero_suffix() -> None:
    run = make_tensor_run(
        [
            (
                "point",
                [
                    ("hidden", torch.tensor([1.0]), "full"),
                    ("logits", torch.tensor([2.0]), "full"),
                ],
            )
        ]
    )
    snapshot = TensorProbeSnapshot(
        probe_id="probe",
        probe_name="model",
        snapshot_index=0,
        replay_index=1,
        timestamp=0.0,
        observations=run["point"].observations,
    )
    writer = FakeWriter()

    export_snapshots_to_tensorboard(writer, [snapshot])

    tags = {tag for tag, _, _ in writer.scalars}
    assert "hidden/numel" in tags
    assert "logits/numel" in tags
    assert not any("invocation_0" in tag for tag in tags)


def test_tensorboard_step_must_be_an_integer() -> None:
    writer = FakeWriter()
    snapshot = make_probe_snapshot(torch.ones(1))

    with pytest.raises(TypeError, match="step must be an integer"):
        export_snapshots_to_tensorboard(writer, [snapshot], step=1.5)  # type: ignore[arg-type]
