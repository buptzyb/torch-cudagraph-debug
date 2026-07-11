#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECORDER_DIR="$(cd -- "${SCRIPT_DIR}/../recorder" && pwd)"
MODE="${1:-}"
OUTPUT_ROOT="${2:-}"
PYTHON_BIN="${PYTHON:-python}"
TCGD_MEMORY_BIN="${TCGD_MEMORY_BIN:-tcgd-memory}"

if [[ "${MODE}" != "single" && "${MODE}" != "distributed" ]]; then
    echo "usage: $0 {single|distributed} OUTPUT_ROOT" >&2
    exit 2
fi
if [[ -z "${OUTPUT_ROOT}" ]]; then
    echo "OUTPUT_ROOT is required" >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "refusing to reuse output directory: ${OUTPUT_ROOT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}"

if [[ "${MODE}" == "single" ]]; then
    SCENARIOS="${OUTPUT_ROOT}/scenarios"
    LIFETIMES="${OUTPUT_ROOT}/lifetimes"
    REPORTS="${OUTPUT_ROOT}/reports"
    "${PYTHON_BIN}" "${RECORDER_DIR}/compare_runs_and_phases.py" \
        --output-dir "${SCENARIOS}" --record-only
    "${PYTHON_BIN}" "${RECORDER_DIR}/allocation_lifetimes.py" \
        --output-dir "${LIFETIMES}" --record-only
    POOL_MAP="$(cat "${SCENARIOS}/pool-map.txt")"

    "${TCGD_MEMORY_BIN}" timeline "${SCENARIOS}/candidate.tcgd-memory" \
        --only-changed --output "${REPORTS}/timeline"
    "${TCGD_MEMORY_BIN}" summary "${SCENARIOS}/candidate.tcgd-memory" > "${REPORTS}/summary.txt"
    "${TCGD_MEMORY_BIN}" compare-points "${SCENARIOS}/candidate.tcgd-memory" \
        --reference-point phase_start --candidate-point phase_end --only-changed \
        --output "${REPORTS}/compare"
    "${TCGD_MEMORY_BIN}" allocation-lifetimes "${LIFETIMES}/lifetimes.tcgd-memory" \
        --active-at anchor --through after_cleanup \
        --output "${REPORTS}/lifetimes-active"
    "${TCGD_MEMORY_BIN}" allocation-lifetimes "${LIFETIMES}/lifetimes.tcgd-memory" \
        --born-between before_transient after_transient --through after_cleanup \
        --output "${REPORTS}/lifetimes-born"
    "${TCGD_MEMORY_BIN}" compare-points \
        "${SCENARIOS}/baseline.tcgd-memory" \
        "${SCENARIOS}/candidate.tcgd-memory" \
        --reference-point phase_end --candidate-point phase_end --pool-map "${POOL_MAP}" \
        --only-changed --output "${REPORTS}/compare-points"
    "${TCGD_MEMORY_BIN}" compare-phases \
        "${SCENARIOS}/baseline.tcgd-memory" \
        "${SCENARIOS}/candidate.tcgd-memory" \
        --baseline-start phase_start --baseline-end phase_end \
        --candidate-start phase_start --candidate-end phase_end \
        --pool-map "${POOL_MAP}" --only-changed \
        --output "${REPORTS}/compare-phases"

    test "$(find "${REPORTS}" -name report.json | wc -l)" -eq 6
else
    GROUP_ROOT="${OUTPUT_ROOT}/groups"
    REPORTS="${OUTPUT_ROOT}/reports"
    NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone --nproc-per-node="${NPROC_PER_NODE}" \
        "${RECORDER_DIR}/distributed_run_groups.py" \
        --output-dir "${GROUP_ROOT}" --record-only

    "${TCGD_MEMORY_BIN}" group-summary "${GROUP_ROOT}/baseline" \
        --output "${REPORTS}/baseline-summary"
    "${TCGD_MEMORY_BIN}" group-summary "${GROUP_ROOT}/candidate" \
        --output "${REPORTS}/candidate-summary"
    "${TCGD_MEMORY_BIN}" compare-run-group-phases \
        "${GROUP_ROOT}/baseline" "${GROUP_ROOT}/candidate" \
        --baseline-start phase_start --baseline-end phase_end \
        --candidate-start phase_start --candidate-end phase_end \
        --only-changed --output "${REPORTS}/group-phase"

    test "$(find "${REPORTS}" -name report.json | wc -l)" -eq 3
fi

printf 'reports: %s\n' "$(cd -- "${OUTPUT_ROOT}/reports" && pwd)"
