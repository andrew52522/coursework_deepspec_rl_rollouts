#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${COURSEWORK_WORKDIR:-$(dirname "${REPO_ROOT}")}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/environment/versions.env"

VERL_DIR="${VERL_DIR:-${WORK_ROOT}/third_party/verl}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models/Qwen3-0.6B}"
DATA_DIR="${DATA_DIR:-${WORK_ROOT}/datasets/gsm8k}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

IMAGE="${COURSEWORK_IMAGE:-coursework-verl-cu130:latest}"
GPU_CAPACITY_GIB="${GPU_CAPACITY_GIB:-7.62}"

for path in \
    "${VERL_DIR}/.git" \
    "${MODEL_DIR}/config.json" \
    "${DATA_DIR}/train.parquet" \
    "${DATA_DIR}/test.parquet"; do

    if [[ ! -e "${path}" ]]; then
        echo "ERROR: required setup artifact missing:"
        echo "  ${path}"
        echo
        echo "Run first:"
        echo "  ./scripts/setup.sh"
        exit 1
    fi
done

ACTUAL_VERL_COMMIT="$(git -C "${VERL_DIR}" rev-parse HEAD)"

if [[ "${ACTUAL_VERL_COMMIT}" != "${VERL_COMMIT}" ]]; then
    echo "ERROR: verl checkout is not at the pinned commit."
    echo "expected: ${VERL_COMMIT}"
    echo "actual:   ${ACTUAL_VERL_COMMIT}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE="${LOG_DIR}/grpo_1x4070_${TIMESTAMP}.log"

echo "============================================================"
echo "1x RTX 4070 GRPO BASELINE"
echo "============================================================"
echo "verl:  ${VERL_DIR}"
echo "model: ${MODEL_DIR}"
echo "data:  ${DATA_DIR}"
echo "log:   ${LOG_FILE}"
echo

set +e

docker run --rm \
    --gpus all \
    --ipc=host \
    -v "${VERL_DIR}:/workspace/verl" \
    -v "${MODEL_DIR}:/model:ro" \
    -v "${DATA_DIR}:/data/gsm8k:ro" \
    -v "${REPO_ROOT}:/workspace/coursework:ro" \
    "${IMAGE}" \
    bash -lc '
        cd /workspace/verl
        /workspace/coursework/scripts/grpo/run_1x4070_qwen3_0.6b.sh
    ' \
    2>&1 | tee "${LOG_FILE}"

RUN_EXIT_CODE="${PIPESTATUS[0]}"

set -e

{
    echo
    echo "=== DOCKER/GRPO EXIT CODE ==="
    echo "${RUN_EXIT_CODE}"
} | tee -a "${LOG_FILE}"

echo
echo "============================================================"
echo "GRPO SUMMARY"
echo "============================================================"

python3 \
    "${REPO_ROOT}/scripts/analysis/summarize_grpo_log.py" \
    "${LOG_FILE}" \
    --gpu-capacity-gib "${GPU_CAPACITY_GIB}"

echo
echo "raw log: ${LOG_FILE}"

exit "${RUN_EXIT_CODE}"
