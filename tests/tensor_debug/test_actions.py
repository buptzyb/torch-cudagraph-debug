from __future__ import annotations

import numpy as np
import pytest
import torch

import torch_cudagraph_debug.tensor_debug as tensor_debug
from torch_cudagraph_debug.tensor_debug import (
    TensorCompare,
    TensorPrint,
    TensorRecord,
)


def test_print_action_spec() -> None:
    spec = TensorPrint(max_items=4, every=2, summary=False)._to_native()

    assert spec == {
        "kind": "print",
        "max_items": 4,
        "every": 2,
        "summary": False,
        "enabled": True,
    }


def test_record_action_spec() -> None:
    assert TensorRecord()._to_native() == {"kind": "record", "enabled": True}


def test_compare_accepts_expected_tensor_sequence() -> None:
    expected = torch.tensor([1.0, 2.0])

    spec = TensorCompare([expected], rtol=0.1, atol=0.2, equal_nan=True)._to_native()

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

    spec = TensorCompare([expected])._to_native()

    assert torch.equal(spec["expected"][0], torch.tensor([1, 2, 3], dtype=torch.int32))


def test_compare_rejects_bare_tensor_or_numpy_array() -> None:
    with pytest.raises(TypeError, match="wrap a single expected tensor"):
        TensorCompare(torch.tensor([1.0]))._to_native()

    with pytest.raises(TypeError, match="wrap a single expected tensor"):
        TensorCompare(np.array([1.0], dtype=np.float32))._to_native()


def test_compare_rejects_empty_expected_sequence() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TensorCompare([])._to_native()


def test_compare_rejects_non_cpu_expected() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    with pytest.raises(ValueError, match="CPU"):
        TensorCompare([torch.ones(1, device="cuda")])._to_native()


def test_multi_expected_compare_action_is_not_exported() -> None:
    legacy_name = "Tensor" + "Compare" + "Each"

    assert legacy_name not in tensor_debug.__all__
    assert not hasattr(tensor_debug, legacy_name)
