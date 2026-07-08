#!/usr/bin/env bash
set -euo pipefail

DATA_BASE=${DATA_BASE:-/workspace/NetAI/data}
PROFILE_RESULT_DIR=${PROFILE_RESULT_DIR:-/workspace/NetAI/KaeTE/results/profile/teal}

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

cd "$(dirname "$0")/run"

for topo_name in ${TOPOLOGIES}; do
    data_dir="${DATA_BASE}/${topo_name}"
    if [[ ! -d "${data_dir}" ]]; then
        echo "Missing dataset directory: ${data_dir}" >&2
        exit 1
    fi

    python teal.py \
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
