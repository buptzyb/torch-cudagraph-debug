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
from torch_cudagraph_debug.tensor_debug.recording import _SUMMARY_CHUNK_ELEMENTS

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
    observation = manifest["points"][0]["observations"][0]
    assert observation["name"] == str(dtypes[0])
    assert "probe_name" not in observation
    assert observation["blob_file"].startswith("blobs/")

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


def test_load_rejects_removed_probe_name_field(tmp_path: Path) -> None:
    bundle = tmp_path / "old-field.tcgd-tensor"
    make_tensor_run(
        [("point", [("x", torch.ones(1), "full")])],
        bundle_dir=bundle,
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observation = manifest["points"][0]["observations"][0]
    observation["probe_name"] = observation.pop("name")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TensorBundleError, match="invalid fields"):
        TensorRun.load(bundle)


def test_load_rejects_noncontiguous_per_name_invocations(tmp_path: Path) -> None:
    bundle = tmp_path / "invalid-invocation.tcgd-tensor"
    make_tensor_run(
        [("point", [("x", torch.ones(1), "full")])],
        bundle_dir=bundle,
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["points"][0]["observations"][0]["invocation_index"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TensorBundleError, match="independently per name"):
        TensorRun.load(bundle)


def test_recorder_watch_grad_validates_observation_name_at_registration() -> None:
    recorder = TensorRecorder(execution="eager")
    tensor = torch.ones(1, requires_grad=True)

    with pytest.raises(ValueError, match="observation name must be a non-empty string"):
        recorder.watch_grad(tensor, name="")

    recorder.close()


def test_summary_std_is_stable_for_large_mean_offsets(tmp_path: Path) -> None:
    bundle = tmp_path / "large-offset.tcgd-tensor"
    value = torch.full((100_000,), 1e12, dtype=torch.float64)
    value += torch.arange(100_000, dtype=torch.float64) % 2
    make_tensor_run(
        [("point", [("x", value, "summary")])],
        bundle_dir=bundle,
    )

    summary = TensorRun.load(bundle)["point"].observation("x").summary
    assert summary.mean == pytest.approx(value.mean().item())
    assert summary.std == pytest.approx(value.std().item(), rel=1e-3)


def test_summary_std_is_stable_across_chunk_boundaries() -> None:
    numel = _SUMMARY_CHUNK_ELEMENTS + 3
    value = torch.full((numel,), 1e12, dtype=torch.float64)
    value += torch.arange(numel, dtype=torch.float64) % 2
    run = make_tensor_run([("point", [("x", value, "summary")])])

    summary = run["point"].observation("x").summary
    assert summary.mean == pytest.approx(value.mean().item())
    assert summary.std == pytest.approx(value.std().item(), rel=1e-3)


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
    assert summary["std"] == 0.0
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


def test_equal_digest_shortcut_verifies_both_full_blobs(tmp_path: Path) -> None:
    reference_bundle = tmp_path / "reference.tcgd-tensor"
    candidate_bundle = tmp_path / "candidate.tcgd-tensor"
    value = torch.arange(4, dtype=torch.float32)
    make_tensor_run(
        [("point", [("x", value, "full")])],
        bundle_dir=reference_bundle,
    )
    make_tensor_run(
        [("point", [("x", value, "full")])],
        bundle_dir=candidate_bundle,
    )
    reference = TensorRun.load(reference_bundle)
    candidate = TensorRun.load(candidate_bundle)
    blob = candidate["point"].observation("x")._blob_path
    assert blob is not None
    blob.write_bytes(bytes(value.numel() * value.element_size()))

    from torch_cudagraph_debug.tensor_debug import compare_points

    with pytest.raises(TensorBundleError, match="does not match its SHA-256"):
        compare_points(reference["point"], candidate["point"])


def test_recorder_manifest_failures_do_not_commit_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = TensorRecorder(
        execution="eager",
        bundle_dir=tmp_path / "transaction.tcgd-tensor",
    )
    original = recorder._write_manifest
    failed_point = False
    failed_finish = False

    def flaky(**kwargs: object) -> None:
        nonlocal failed_point, failed_finish
        complete = kwargs["complete"]
        if not complete and not failed_point:
            failed_point = True
            raise TensorBundleError("injected manifest failure")
        if complete and not failed_finish:
            failed_finish = True
            raise TensorBundleError("injected manifest failure")
        original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(recorder, "_write_manifest", flaky)
    with pytest.raises(TensorBundleError, match="injected"):
        with recorder.record_point("point"):
            pass
    assert recorder.preview().points == ()

    with recorder.record_point("point"):
        pass
    with pytest.raises(TensorBundleError, match="injected"):
        recorder.finish()
    assert recorder._result is None
    run = recorder.finish()
    assert [point.label for point in run.points] == ["point"]


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


def test_tensor_bundle_load_rejects_lossy_scalar_coercions(tmp_path: Path) -> None:
    bundle = tmp_path / "strict.tcgd-tensor"
    make_tensor_run(
        [("point", [("x", torch.ones(1), "full")])],
        bundle_dir=bundle,
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["complete"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TensorBundleError, match="boolean"):
        TensorRun.load(bundle)

    manifest["complete"] = True
    manifest["points"][0]["index"] = 0.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TensorBundleError, match="integer"):
        TensorRun.load(bundle)

    manifest["points"][0]["index"] = 0
    manifest["created_at"] = float("nan")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TensorBundleError, match="non-finite"):
        TensorRun.load(bundle)


def test_tensor_recorder_rejects_bad_environment_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "not-an-integer")

    with pytest.raises(ValueError, match="environment variable RANK"):
        TensorRecorder(execution="eager")


def test_tensor_recorder_context_rejects_reentry_and_explicit_finish() -> None:
    recorder = TensorRecorder(execution="eager")

    with recorder:
        with pytest.raises(TensorDebugError, match="re-entered"):
            recorder.__enter__()
        with pytest.raises(TensorDebugError, match="inside its context"):
            recorder.finish()


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
    assert recorder.preview() is recorder.result
    assert recorder.finish() is recorder.result
    with pytest.raises(TensorDebugError, match="closed"):
        with recorder.record_point("late"):
            pass


def test_tensor_run_and_point_metadata_are_deeply_immutable() -> None:
    source = {"nested": {"values": [1, 2]}}
    recorder = TensorRecorder(
        execution="eager",
        run_metadata=source,
    )
    with recorder.record_point("point", metadata=source):
        pass
    run = recorder.finish()

    source["nested"]["values"].append(3)
    assert run.run_metadata["nested"]["values"] == (1, 2)
    assert run["point"].metadata["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        run.run_metadata["nested"]["new"] = 1


def test_eager_recorder_strict_scope_rejects_out_of_point_observation() -> None:
    tensor = torch.ones(1)
    permissive = TensorRecorder(execution="eager")
    assert permissive.observe(tensor, name="ignored") is tensor

    strict = TensorRecorder(execution="eager", strict_scope=True)
    with pytest.raises(TensorDebugError, match="record_point"):
        strict.observe(tensor, name="outside")


def test_recorder_close_inherits_configured_synchronization() -> None:
    recorder = TensorRecorder(execution="eager", synchronize=False)
    calls: list[object] = []

    class FakeCollector:
        def close(self, *, synchronize: object) -> None:
            calls.append(synchronize)

    recorder._collector = FakeCollector()  # type: ignore[assignment]
    recorder.close()

    assert calls == [False]
