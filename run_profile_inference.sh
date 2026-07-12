#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${DATA_BASE:?Set DATA_BASE to the dataset root}"
PROFILE_RESULT_DIR=${PROFILE_RESULT_DIR:-"${ROOT_DIR}/results/profile/teal"}
PYTHON=${PYTHON:-python}

BATCH_SIZE=${BATCH_SIZE:-1}
NUM_PATHS_PER_PAIR=${NUM_PATHS_PER_PAIR:-4}
PROFILE_WARMUP_SAMPLES=${PROFILE_WARMUP_SAMPLES:-20}
ADMM_STEPS=${ADMM_STEPS:-3}
LAYERS=${LAYERS:-6}
RHO=${RHO:-1.0}
LR=${LR:-0.001}
OBJ=${OBJ:-total_flow}
DEVID=${DEVID:-0}

TOPOLOGIES=${TOPOLOGIES:-"DynGEANT geant"}

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "Python environment not found: ${PYTHON}" >&2
    exit 1
fi

cd "${ROOT_DIR}/run"

for topo_name in ${TOPOLOGIES}; do
    data_dir="${DATA_BASE}/${topo_name}"
    if [[ ! -d "${data_dir}" ]]; then
        echo "Missing dataset directory: ${data_dir}" >&2
        exit 1
    fi

    "${PYTHON}" teal.py \
        --profile-inference \
        --profile-warmup-samples "${PROFILE_WARMUP_SAMPLES}" \
        --profile-result-dir "${PROFILE_RESULT_DIR}" \
        --topo_name "${topo_name}" \
        --data_dir "${data_dir}" \
        --bsz "${BATCH_SIZE}" \
        --num_paths_per_pair "${NUM_PATHS_PER_PAIR}" \
        --admm-steps "${ADMM_STEPS}" \
        --layers "${LAYERS}" \
        --rho "${RHO}" \
        --lr "${LR}" \
        --obj "${OBJ}" \
        --devid "${DEVID}"
done
