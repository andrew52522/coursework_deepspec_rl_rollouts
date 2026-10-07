#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${COURSEWORK_WORKDIR:-$(dirname "${REPO_ROOT}")}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/environment/versions.env"

VERL_DIR="${VERL_DIR:-${WORK_ROOT}/third_party/verl}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models/Qwen3-0.6B}"
DATA_DIR="${DATA_DIR:-${WORK_ROOT}/datasets/gsm8k}"
IMAGE="${COURSEWORK_IMAGE:-coursework-verl-cu130:latest}"

echo "============================================================"
echo "COURSEWORK SETUP"
echo "============================================================"
echo "repo:    ${REPO_ROOT}"
echo "work:    ${WORK_ROOT}"
echo "verl:    ${VERL_DIR}"
echo "model:   ${MODEL_DIR}"
echo "dataset: ${DATA_DIR}"
echo "image:   ${IMAGE}"
echo

for cmd in git docker nvidia-smi python3; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "ERROR: required command not found: ${cmd}"
        exit 1
    fi
done

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is unavailable to the current user."
    exit 1
fi


# ---------------------------------------------------------------------
# 1. Pinned verl checkout
# ---------------------------------------------------------------------

echo "=== PINNED VERL ==="

mkdir -p "$(dirname "${VERL_DIR}")"

if [[ ! -d "${VERL_DIR}/.git" ]]; then
    git clone https://github.com/verl-project/verl.git "${VERL_DIR}"
fi

if ! git -C "${VERL_DIR}" cat-file -e "${VERL_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${VERL_DIR}" fetch origin
fi

git -C "${VERL_DIR}" checkout --detach "${VERL_COMMIT}"

ACTUAL_VERL_COMMIT="$(git -C "${VERL_DIR}" rev-parse HEAD)"

if [[ "${ACTUAL_VERL_COMMIT}" != "${VERL_COMMIT}" ]]; then
    echo "ERROR: wrong verl commit"
    echo "expected: ${VERL_COMMIT}"
    echo "actual:   ${ACTUAL_VERL_COMMIT}"
    exit 1
fi

echo "verl commit: ${ACTUAL_VERL_COMMIT}"


# ---------------------------------------------------------------------
# 2. Build validated coursework container
# ---------------------------------------------------------------------

echo
echo "=== BUILD CONTAINER ==="

docker build \
    -f "${REPO_ROOT}/docker/Dockerfile.cu130" \
    -t "${IMAGE}" \
    "${REPO_ROOT}"


# ---------------------------------------------------------------------
# 3. CUDA smoke
# ---------------------------------------------------------------------

echo
echo "=== CUDA SMOKE ==="

docker run --rm --gpus all \
    -v "${REPO_ROOT}:/workspace/coursework:ro" \
    "${IMAGE}" \
    bash /workspace/coursework/scripts/bootstrap/cuda_container_smoke.sh


# ---------------------------------------------------------------------
# 4. Materialize pinned verl environment
# ---------------------------------------------------------------------

echo
echo "=== VERL / UV ENVIRONMENT ==="

docker run --rm \
    -v "${VERL_DIR}:/workspace/verl" \
    "${IMAGE}" \
    bash -lc '
        cd /workspace/verl
        uv run \
          --frozen \
          --all-packages \
          --extra vllm \
          --extra fsdp \
          python3 -c "
import torch
import transformers
import vllm
print(\"torch:\", torch.__version__)
print(\"transformers:\", transformers.__version__)
print(\"vllm:\", vllm.__version__)
"
    '


# ---------------------------------------------------------------------
# 5. Download Qwen3-0.6B
# ---------------------------------------------------------------------

mkdir -p "${MODEL_DIR}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo
    echo "=== DOWNLOAD MODEL: ${MODEL_ID} ==="

    docker run --rm \
        -e MODEL_ID="${MODEL_ID}" \
        -v "${VERL_DIR}:/workspace/verl" \
        -v "${MODEL_DIR}:/model" \
        "${IMAGE}" \
        bash -lc '
            cd /workspace/verl

            uv run \
              --frozen \
              --all-packages \
              --extra vllm \
              --extra fsdp \
              python3 -c "
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ[\"MODEL_ID\"],
    local_dir=\"/model\",
)
"
        '
else
    echo
    echo "=== MODEL ALREADY PRESENT ==="
    echo "${MODEL_DIR}"
fi


# ---------------------------------------------------------------------
# 6. Prepare GSM8K using the pinned verl preprocessing script
# ---------------------------------------------------------------------

mkdir -p "${DATA_DIR}"

if [[ ! -f "${DATA_DIR}/train.parquet" ||
      ! -f "${DATA_DIR}/test.parquet" ]]; then

    echo
    echo "=== PREPARE GSM8K ==="

    docker run --rm \
        -v "${VERL_DIR}:/workspace/verl" \
        -v "${DATA_DIR}:/data/gsm8k" \
        "${IMAGE}" \
        bash -lc '
            cd /workspace/verl

            uv run \
              --frozen \
              --all-packages \
              --extra vllm \
              --extra fsdp \
              python3 examples/data_preprocess/gsm8k.py \
              --local_save_dir /data/gsm8k
        '
else
    echo
    echo "=== GSM8K ALREADY PRESENT ==="
    echo "${DATA_DIR}"
fi

if [[ ! -f "${DATA_DIR}/train.parquet" ||
      ! -f "${DATA_DIR}/test.parquet" ]]; then
    echo "ERROR: GSM8K preprocessing did not create expected parquet files."
    exit 1
fi


# ---------------------------------------------------------------------
# 7. Standalone vLLM smoke using the downloaded model
# ---------------------------------------------------------------------

echo
echo "=== VLLM SMOKE ==="

docker run --rm \
    --gpus all \
    --ipc=host \
    -v "${VERL_DIR}:/workspace/verl" \
    -v "${MODEL_DIR}:/model:ro" \
    -v "${REPO_ROOT}:/workspace/coursework:ro" \
    "${IMAGE}" \
    bash -lc '
        cd /workspace/verl

        uv run \
          --frozen \
          --all-packages \
          --extra vllm \
          --extra fsdp \
          python3 /workspace/coursework/scripts/bootstrap/vllm_smoke.py
    '

echo
echo "============================================================"
echo "SETUP: PASS"
echo "============================================================"
echo "verl:    ${VERL_DIR}"
echo "model:   ${MODEL_DIR}"
echo "dataset: ${DATA_DIR}"
