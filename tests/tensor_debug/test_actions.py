from __future__ import annotations

import numpy as np
import pytest
import torch

from torch_cudagraph_debug.tensor_debug import (
    CheckAction,
    PrintAction,
    RecordAction,
    TensorObservationKey,
)


def test_print_action_spec() -> None:
    spec = PrintAction(max_items=4, every=2, summary=False)._to_native()

    assert spec == {
        "kind": "print",
        "max_items": 4,
        "every": 2,
        "summary": False,
        "enabled": True,
    }


def test_record_action_spec() -> None:
    assert RecordAction()._to_native() == {"kind": "record", "enabled": True}


def test_check_accepts_expected_tensor_sequence() -> None:
    expected = torch.tensor([1.0, 2.0])

    spec = CheckAction([expected], rtol=0.1, atol=0.2, equal_nan=True)._to_native()

    assert spec["kind"] == "check"
    assert spec["rtol"] == 0.1
    assert spec["atol"] == 0.2
    assert spec["equal_nan"] is True
    assert len(spec["expected"]) == 1
    assert torch.equal(spec["expected"][0], expected)
    assert spec["expected"][0].device.type == "cpu"
    assert spec["expected"][0].is_contiguous()


def test_check_accepts_numpy_array_sequence() -> None:
    expected = np.array([1, 2, 3], dtype=np.int32)

    spec = CheckAction([expected])._to_native()

    assert torch.equal(spec["expected"][0], torch.tensor([1, 2, 3], dtype=torch.int32))


def test_check_accepts_single_tensor_or_numpy_array() -> None:
    tensor_spec = CheckAction(torch.tensor([1.0]))._to_native()
    array_spec = CheckAction(np.array([2.0], dtype=np.float32))._to_native()

    assert len(tensor_spec["expected"]) == 1
    assert torch.equal(tensor_spec["expected"][0], torch.tensor([1.0]))
    assert len(array_spec["expected"]) == 1
    assert torch.equal(array_spec["expected"][0], torch.tensor([2.0]))


def test_check_accepts_semantic_observation_mapping() -> None:
    hidden_key = TensorObservationKey("hidden", 0)
    output_key = TensorObservationKey("output", 1)

    spec = CheckAction(
        {
            hidden_key: torch.tensor([1.0]),
            output_key: np.array([2.0], dtype=np.float32),
        }
    )._to_native()

    assert [(item["name"], item["invocation_index"]) for item in spec["expected"]] == [
        ("hidden", 0),
        ("output", 1),
    ]
    assert all(isinstance(item["tensor"], torch.Tensor) for item in spec["expected"])

    with pytest.raises(TypeError, match="TensorObservationKey"):
        CheckAction({"hidden": torch.tensor([1.0])})._to_native()  # type: ignore[dict-item]


def test_check_rejects_empty_expected_sequence() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CheckAction([])._to_native()


@pytest.mark.gpu
def test_check_rejects_non_cpu_expected() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    with pytest.raises(ValueError, match="CPU"):
        CheckAction([torch.ones(1, device="cuda")])._to_native()


def test_actions_reject_lossy_scalar_types() -> None:
    with pytest.raises(TypeError, match="max_items"):
        PrintAction(max_items=True)
    with pytest.raises(TypeError, match="every"):
        PrintAction(every=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="enabled"):
        RecordAction(enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rtol must be finite"):
        CheckAction(torch.ones(1), rtol=float("nan"))
    with pytest.raises(TypeError, match="equal_nan"):
        CheckAction(torch.ones(1), equal_nan=1)  # type: ignore[arg-type]
