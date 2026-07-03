from __future__ import annotations

import pytest
from dataclasses import replace
import torch

from torch_cudagraph_debug.tensor_debug import (
    PrintAction,
    TensorObservationKey,
    TensorOwnershipError,
    TensorProbe,
    TensorProbeSnapshot,
    compare_snapshots,
)

from ._run_helpers import make_probe_snapshot, make_tensor_run


def test_probe_snapshot_aggregates_all_invocations() -> None:
    run = make_tensor_run(
        [
            (
                "snapshot",
                [
                    ("hidden", torch.tensor([1.0]), "full"),
                    ("hidden", torch.tensor([2.0]), "full"),
                ],
            )
        ]
    )
    captured = TensorProbeSnapshot(
        probe_id="probe",
        probe_name="hidden",
        replay_index=3,
        timestamp=0.0,
        observations=run.points[0].observations,
    )

    assert captured.replay_index == 3
    assert [item.invocation_index for item in captured.observations] == [0, 1]
    assert torch.equal(captured.tensor(invocation_index=0), torch.tensor([1.0]))
    assert torch.equal(captured.tensor(invocation_index=1), torch.tensor([2.0]))
    assert not hasattr(captured.observation(), "run_id")
    assert not hasattr(captured.observation(), "point_index")


def test_probe_snapshot_supports_named_observations() -> None:
    run = make_tensor_run(
        [
            (
                "snapshot",
                [
                    ("hidden", torch.tensor([1.0]), "full"),
                    ("logits", torch.tensor([2.0]), "full"),
                    ("hidden", torch.tensor([3.0]), "full"),
                ],
            )
        ]
    )
    snapshot = TensorProbeSnapshot(
        probe_id="probe",
        probe_name="collector",
        replay_index=1,
        timestamp=0.0,
        observations=run.points[0].observations,
    )

    assert list(snapshot.by_key) == [
        TensorObservationKey("hidden", 0),
        TensorObservationKey("logits", 0),
        TensorObservationKey("hidden", 1),
    ]
    assert snapshot.observation("logits").order == 1
    assert torch.equal(
        snapshot.tensor("hidden", invocation_index=1),
        torch.tensor([3.0]),
    )
    with pytest.raises(KeyError, match="collector"):
        snapshot.tensor()


def test_compare_snapshots_reuses_observation_comparison() -> None:
    reference = make_probe_snapshot(
        torch.tensor([1.0, 2.0]), replay_index=1, probe_id="probe"
    )
    candidate = make_probe_snapshot(
        torch.tensor([1.0, 3.0]), replay_index=2, probe_id="probe"
    )

    comparison = compare_snapshots(reference, candidate)

    assert comparison.status == "mismatch"
    assert comparison.observation_comparisons[0].mismatch_count == 1
    assert comparison.to_dict()["kind"] == "snapshot-comparison"
    assert "replay=1" in comparison.to_text()


def test_compare_snapshots_aligns_observations_across_probe_owners() -> None:
    reference = replace(
        make_probe_snapshot(torch.tensor([1.0]), probe_name="hidden"),
        probe_name="eager-probe",
    )
    candidate = replace(
        make_probe_snapshot(torch.tensor([1.0]), probe_name="hidden"),
        probe_id="graph",
        probe_name="graph-probe",
    )

    comparison = compare_snapshots(reference, candidate)

    assert comparison.ok
    assert comparison.observation_comparisons[0].key == TensorObservationKey(
        "hidden", 0
    )


def test_probe_compare_validates_ownership_and_replay_order() -> None:
    probe = TensorProbe("hidden", [PrintAction(enabled=False)])
    reference = make_probe_snapshot(
        torch.tensor([1.0]),
        replay_index=1,
        probe_id=probe._probe_id,
    )
    candidate = make_probe_snapshot(
        torch.tensor([1.0]),
        replay_index=2,
        probe_id=probe._probe_id,
    )
    foreign = make_probe_snapshot(
        torch.tensor([1.0]),
        replay_index=3,
        probe_id="foreign",
    )

    assert probe.compare(reference, candidate).ok
    with pytest.raises(ValueError, match="must follow"):
        probe.compare(candidate, reference)
    with pytest.raises(ValueError, match="must follow"):
        compare_snapshots(candidate, reference)
    with pytest.raises(TensorOwnershipError, match="does not belong"):
        probe.compare(reference, foreign)


def test_snapshot_rejects_noncontiguous_invocation_indices() -> None:
    captured = make_probe_snapshot(torch.tensor([1.0]))
    observation = captured.observations[0]
    invalid = type(observation)(
        **{
            **observation.__dict__,
            "invocation_index": 2,
        }
    )
    with pytest.raises(ValueError, match="contiguous"):
        TensorProbeSnapshot(
            probe_id="probe",
            probe_name="mid",
            replay_index=1,
            timestamp=0.0,
            observations=(invalid,),
        )


def test_snapshot_rejects_noncontiguous_observation_order() -> None:
    captured = make_probe_snapshot(torch.tensor([1.0]))
    observation = captured.observations[0]
    invalid = type(observation)(
        **{
            **observation.__dict__,
            "order": 1,
        }
    )
    with pytest.raises(ValueError, match="observations must be contiguous"):
        TensorProbeSnapshot(
            probe_id="probe",
            probe_name="mid",
            replay_index=1,
            timestamp=0.0,
            observations=(invalid,),
        )
