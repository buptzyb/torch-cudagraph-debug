"""Compare independent graph pools and decompose two phase ranges.

Run with:
  python examples/memory_debug/compare_runs_and_phases.py --output-dir /tmp/tcgd-runs
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from torch_cudagraph_debug.memory_debug import (
    MemoryRecorder,
    MemoryRun,
    compare_phases,
    compare_points,
)

MIB = 1024 * 1024
DEFAULT_POOL = (0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-only", action="store_true")
    return parser.parse_args()


def _format_pool_id(pool_id: tuple[object, ...]) -> str:
    return ",".join(str(item) for item in pool_id)


def _record_scenario(
    name: str,
    bundle_dir: Path,
    *,
    private_bytes: int,
) -> tuple[object, ...]:
    recorder = MemoryRecorder(
        name=name,
        rank=0,
        bundle_dir=bundle_dir,
        run_metadata={"scenario": name},
    )
    static_state = torch.empty(4 * MIB, dtype=torch.uint8, device="cuda")
    recorder.record_point("before_pool")

    pool = torch.cuda.graph_pool_handle()
    seed_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(seed_graph, pool=pool):
        seed_state = torch.empty(MIB, dtype=torch.uint8, device="cuda")
        seed_state.fill_(1)
    recorder.record_point("phase_start")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        graph_state = torch.empty(private_bytes, dtype=torch.uint8, device="cuda")
        graph_state.fill_(1)
        recorder.record_point("during_capture")

    recorder.record_point("phase_end")
    graph.replay()
    recorder.record_point("after_replay")
    run = recorder.finish()

    created_pools = set(run["phase_start"].pool_stats) - set(
        run["before_pool"].pool_stats
    )
    private_pools = [pool_id for pool_id in created_pools if pool_id != DEFAULT_POOL]
    if len(private_pools) != 1:
        raise RuntimeError(f"expected one seeded private pool, found {private_pools}")
    assert static_state.numel() == 4 * MIB
    assert seed_state.numel() == MIB
    assert graph_state.numel() == private_bytes
    return private_pools[0]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.empty_cache()
    baseline_bundle = output_dir / "baseline.tcgd-memory"
    candidate_bundle = output_dir / "candidate.tcgd-memory"
    baseline_pool = _record_scenario(
        "baseline",
        baseline_bundle,
        private_bytes=8 * MIB,
    )

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    candidate_pool = _record_scenario(
        "candidate",
        candidate_bundle,
        private_bytes=12 * MIB,
    )

    pool_mapping = {baseline_pool: candidate_pool}
    pool_map_text = (
        f"{_format_pool_id(baseline_pool)}={_format_pool_id(candidate_pool)}"
    )
    (output_dir / "pool-map.txt").write_text(pool_map_text + "\n", encoding="utf-8")

    if args.record_only:
        print(f"baseline bundle: {baseline_bundle}")
        print(f"candidate bundle: {candidate_bundle}")
        print(f"pool mapping: {pool_map_text}")
        return

    baseline = MemoryRun.load(baseline_bundle, cache_snapshots=False)
    candidate = MemoryRun.load(candidate_bundle, cache_snapshots=False)
    endpoint = compare_points(
        baseline["phase_end"],
        candidate["phase_end"],
        pool_mapping=pool_mapping,
    )
    phase = compare_phases(
        baseline.between("phase_start", "phase_end"),
        candidate.between("phase_start", "phase_end"),
        pool_mapping=pool_mapping,
    )
    endpoint_paths = endpoint.write(output_dir / "endpoint-comparison")
    phase_paths = phase.write(output_dir / "phase-comparison")

    assert any(item.match == "mapped" for item in endpoint.pool_comparisons)
    assert all(row["identity_holds"] for row in phase.allocator_scope_decomposition)
    assert endpoint_paths["json"].is_file()
    assert phase_paths["allocator_scope_decomposition"].is_file()

    print(f"pool mapping: {pool_map_text}")
    print(endpoint.to_text(include_unchanged=False))
    print()
    print(phase.to_text(include_unchanged=False))


if __name__ == "__main__":
    main()
