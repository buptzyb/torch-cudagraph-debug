#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECORDER_DIR="$(cd -- "${SCRIPT_DIR}/../recorder" && pwd)"
OUTPUT_ROOT="${1:-}"
PYTHON_BIN="${PYTHON:-python}"
TCGD_TENSOR_BIN="${TCGD_TENSOR_BIN:-tcgd-tensor}"

if [[ -z "${OUTPUT_ROOT}" ]]; then
    echo "usage: $0 OUTPUT_ROOT" >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "refusing to reuse output directory: ${OUTPUT_ROOT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}"

MATCHING="${OUTPUT_ROOT}/matching"
SERIES="${OUTPUT_ROOT}/series"
REPORTS="${OUTPUT_ROOT}/reports"

"${PYTHON_BIN}" "${RECORDER_DIR}/eager_vs_cuda_graph.py" \
    --output-dir "${MATCHING}" --record-only
"${PYTHON_BIN}" "${RECORDER_DIR}/replay_series.py" \
    --output-dir "${SERIES}" --record-only
mkdir -p "${REPORTS}"

"${TCGD_TENSOR_BIN}" summary \
    "${MATCHING}/eager.tcgd-tensor" > "${REPORTS}/summary.txt"
"${TCGD_TENSOR_BIN}" compare-points \
    "${MATCHING}/eager.tcgd-tensor" \
    "${MATCHING}/cuda-graph.tcgd-tensor" \
    --reference-point forward --candidate-point forward \
    --output "${REPORTS}/compare-points"
"${TCGD_TENSOR_BIN}" compare-runs \
    "${MATCHING}/eager.tcgd-tensor" \
    "${MATCHING}/cuda-graph.tcgd-tensor" \
    --output "${REPORTS}/compare-runs"

if "${TCGD_TENSOR_BIN}" compare-point-series \
    "${SERIES}/eager-summary.tcgd-tensor" \
    "${SERIES}/graph-series.tcgd-tensor" \
    --reference-point forward --only-changed \
    --output "${REPORTS}/compare-point-series"; then
    echo "compare-point-series should report the intentional replay mismatch" >&2
    exit 1
else
    status=$?
    if [[ "${status}" -ne 1 ]]; then
        echo "compare-point-series returned unexpected status ${status}" >&2
        exit "${status}"
    fi
fi

test "$(find "${REPORTS}" -name report.json | wc -l)" -eq 3
printf 'reports: %s\n' "$(cd -- "${REPORTS}" && pwd)"

