from __future__ import annotations

from pathlib import Path

import torch

from torch_cudagraph_debug.tensor_debug.cli import main

from ._run_helpers import make_tensor_run


def test_tensor_cli_summary_compare_runs_and_series(
    tmp_path: Path,
    capsys,
) -> None:
    reference_bundle = tmp_path / "reference.tcgd-tensor"
    candidate_bundle = tmp_path / "candidate.tcgd-tensor"
    make_tensor_run(
        [("forward", [("x", torch.tensor([1.0]), "full")])],
        name="reference",
        bundle_dir=reference_bundle,
    )
    make_tensor_run(
        [
            ("replay-1", [("x", torch.tensor([1.0]), "full")]),
            ("replay-2", [("x", torch.tensor([2.0]), "full")]),
        ],
        name="candidate",
        bundle_dir=candidate_bundle,
    )

    assert main(["summary", str(reference_bundle)]) == 0
    assert "Tensor run 'reference'" in capsys.readouterr().out

    assert (
        main(
            [
                "compare-points",
                str(reference_bundle),
                "--reference-point",
                "forward",
                "--candidate-point",
                "forward",
            ]
        )
        == 0
    )

    output = tmp_path / "point-report"
    assert (
        main(
            [
                "compare-points",
                str(reference_bundle),
                str(candidate_bundle),
                "--reference-point",
                "forward",
                "--candidate-point",
                "replay-1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "report.json").is_file()

    assert (
        main(
            [
                "compare-point-series",
                str(reference_bundle),
                str(candidate_bundle),
                "--reference-point",
                "forward",
                "--only-changed",
            ]
        )
        == 1
    )
    assert "replay-2: mismatch" in capsys.readouterr().out

    assert (
        main(
            [
                "compare-runs",
                str(reference_bundle),
                str(candidate_bundle),
            ]
        )
        == 1
    )
