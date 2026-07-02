from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from torch_cudagraph_debug.tensor_debug import (
    TensorBundleError,
    TensorDebugError,
    TensorOwnershipError,
    TensorPayloadUnavailableError,
    TensorRecorder,
    TensorRun,
)

from ._run_helpers import make_tensor_run


def _value(dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.tensor([[True, False], [False, True]], dtype=dtype)
    return torch.arange(4, dtype=dtype).reshape(2, 2)


def test_bundle_round_trips_every_supported_dtype_and_loads_lazily(
    tmp_path: Path,
) -> None:
    dtypes = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    )
    bundle = tmp_path / "all-dtypes.tcgd-tensor"
    make_tensor_run(
        [
            (
                "forward",
                [(str(dtype), _value(dtype), "full") for dtype in dtypes],
            )
        ],
        bundle_dir=bundle,
    )

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "torch-cudagraph-debug/tensor-run"
    assert manifest["complete"] is True
    assert manifest["execution"] == "eager"
    assert manifest["points"][0]["observations"][0]["blob_file"].startswith("blobs/")

    loaded = TensorRun.load(bundle)
    for dtype in dtypes:
        observation = loaded["forward"].observation(str(dtype))
        assert observation._tensor_cache == {}
        assert torch.equal(observation.tensor(), _value(dtype))
        assert observation._tensor_cache == {}

    cached = TensorRun.load(bundle, cache_tensors=True)
    observation = cached["forward"].observation(str(torch.bfloat16))
    observation.tensor()
    assert "tensor" in observation._tensor_cache


def test_content_addressing_deduplicates_and_summary_has_no_blob(
    tmp_path: Path,
) -> None:
    tensor = torch.arange(8, dtype=torch.float32)
    bundle = tmp_path / "deduplicated.tcgd-tensor"
    make_tensor_run(
        [
            ("first", [("full", tensor, "full"), ("summary", tensor + 1, "summary")]),
            ("second", [("full", tensor.clone(), "full")]),
        ],
        bundle_dir=bundle,
    )

    assert len(list((bundle / "blobs").glob("*.bin"))) == 1
    loaded = TensorRun.load(bundle)
    assert torch.equal(loaded["second"].observation("full").tensor(), tensor)
    summary = loaded["first"].observation("summary")
    assert summary.payload == "summary"
    assert summary.summary.mean == pytest.approx(4.5)
    with pytest.raises(TensorPayloadUnavailableError, match="summary data only"):
        summary.tensor()


def test_load_rejects_corrupt_blob_and_invalid_manifest(tmp_path: Path) -> None:
    bundle = tmp_path / "corrupt.tcgd-tensor"
    make_tensor_run(
        [("point", [("x", torch.arange(4), "full")])],
        bundle_dir=bundle,
    )
    run = TensorRun.load(bundle)
    observation = run["point"].observation("x")
    assert observation._blob_path is not None
    observation._blob_path.write_bytes(b"bad!")

    with pytest.raises(TensorBundleError, match="unexpected size"):
        TensorRun.load(bundle)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TensorBundleError, match="unexpected"):
        TensorRun.load(bundle)


def test_point_lookup_rejects_foreign_point() -> None:
    first = make_tensor_run([("point", [("x", torch.ones(1), "full")])], name="a")
    second = make_tensor_run([("point", [("x", torch.ones(1), "full")])], name="b")

    with pytest.raises(TensorOwnershipError, match="belongs to run"):
        first.point(second["point"])
    with pytest.raises(KeyError, match="does not exist"):
        first["point"].observation("missing")


def test_scalar_and_empty_tensor_round_trip(tmp_path: Path) -> None:
    bundle = tmp_path / "edge-shapes.tcgd-tensor"
    make_tensor_run(
        [
            (
                "point",
                [
                    ("scalar", torch.tensor(3.5), "full"),
                    ("empty", torch.empty((0, 3)), "full"),
                ],
            )
        ],
        bundle_dir=bundle,
    )
    loaded = TensorRun.load(bundle)
    assert loaded["point"].observation("scalar").tensor().shape == ()
    assert loaded["point"].observation("scalar").tensor().item() == 3.5
    assert loaded["point"].observation("empty").tensor().shape == (0, 3)


def test_extreme_finite_values_keep_manifest_json_valid(tmp_path: Path) -> None:
    bundle = tmp_path / "extreme.tcgd-tensor"
    value = torch.tensor([torch.finfo(torch.float64).max], dtype=torch.float64)
    make_tensor_run(
        [("point", [("x", value, "summary")])],
        bundle_dir=bundle,
    )

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    summary = manifest["points"][0]["observations"][0]["summary"]
    assert summary["mean"] == torch.finfo(torch.float64).max
    assert summary["std"] is None
    assert summary["l2_norm"] is None


def test_payload_digest_is_verified_when_materialized(tmp_path: Path) -> None:
    bundle = tmp_path / "digest.tcgd-tensor"
    make_tensor_run(
        [("point", [("x", torch.arange(4, dtype=torch.float32), "full")])],
        bundle_dir=bundle,
    )
    loaded = TensorRun.load(bundle)
    observation = loaded["point"].observation("x")
    assert observation._blob_path is not None
    observation._blob_path.write_bytes(bytes(observation.nbytes))

    reloaded = TensorRun.load(bundle)
    with pytest.raises(TensorBundleError, match="does not match its SHA-256"):
        reloaded["point"].observation("x").tensor()


def test_load_normalizes_invalid_modes_to_bundle_errors(tmp_path: Path) -> None:
    bundle = tmp_path / "invalid-mode.tcgd-tensor"
    make_tensor_run(
        [("point", [("x", torch.ones(1), "full")])],
        bundle_dir=bundle,
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution"] = "hybrid"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TensorBundleError, match="invalid tensor bundle mode"):
        TensorRun.load(bundle)


def test_recorder_exception_exit_persists_incomplete_terminal_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "aborted.tcgd-tensor"
    recorder: TensorRecorder | None = None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    with pytest.raises(RuntimeError, match="boom"):
        with TensorRecorder(
            execution="eager",
            name="aborted",
            bundle_dir=bundle,
        ) as recorder:
            with recorder.record_point("kept"):
                pass
            with recorder.record_point("discarded"):
                raise RuntimeError("boom")

    assert recorder is not None
    assert recorder.result.complete is False
    assert recorder.result.finished_at is not None
    loaded = TensorRun.load(bundle)
    assert loaded.complete is False
    assert [point.label for point in loaded.points] == ["kept"]
    assert recorder.snapshot_run() is recorder.result
    assert recorder.finish() is recorder.result
    with pytest.raises(TensorDebugError, match="closed"):
        with recorder.record_point("late"):
            pass
