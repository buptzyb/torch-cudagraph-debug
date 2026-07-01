from __future__ import annotations

import numpy as np
import pytest
import torch

from torch_cudagraph_debug.tensor_debug import (
    CompareTensor,
    PrintTensor,
    RecordTensor,
)


def test_print_action_spec() -> None:
    spec = PrintTensor(max_items=4, every=2, summary=False)._to_native()

    assert spec == {
        "kind": "print",
        "max_items": 4,
        "every": 2,
        "summary": False,
        "enabled": True,
    }


def test_record_action_spec() -> None:
    assert RecordTensor()._to_native() == {"kind": "record", "enabled": True}


def test_compare_accepts_expected_tensor_sequence() -> None:
    expected = torch.tensor([1.0, 2.0])

    spec = CompareTensor([expected], rtol=0.1, atol=0.2, equal_nan=True)._to_native()

    assert spec["kind"] == "compare"
    assert spec["rtol"] == 0.1
    assert spec["atol"] == 0.2
    assert spec["equal_nan"] is True
    assert len(spec["expected"]) == 1
    assert torch.equal(spec["expected"][0], expected)
    assert spec["expected"][0].device.type == "cpu"
    assert spec["expected"][0].is_contiguous()


def test_compare_accepts_numpy_array_sequence() -> None:
    expected = np.array([1, 2, 3], dtype=np.int32)

    spec = CompareTensor([expected])._to_native()

    assert torch.equal(spec["expected"][0], torch.tensor([1, 2, 3], dtype=torch.int32))


def test_compare_accepts_single_tensor_or_numpy_array() -> None:
    tensor_spec = CompareTensor(torch.tensor([1.0]))._to_native()
    array_spec = CompareTensor(np.array([2.0], dtype=np.float32))._to_native()

    assert len(tensor_spec["expected"]) == 1
    assert torch.equal(tensor_spec["expected"][0], torch.tensor([1.0]))
    assert len(array_spec["expected"]) == 1
    assert torch.equal(array_spec["expected"][0], torch.tensor([2.0]))


def test_compare_rejects_empty_expected_sequence() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CompareTensor([])._to_native()


def test_compare_rejects_non_cpu_expected() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    with pytest.raises(ValueError, match="CPU"):
        CompareTensor([torch.ones(1, device="cuda")])._to_native()
