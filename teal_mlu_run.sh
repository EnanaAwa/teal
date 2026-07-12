#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${DATA_BASE:?Set DATA_BASE to the MLU-labelled dataset root}"
RESULT_BASE=${RESULT_BASE:-"${ROOT_DIR}/results/teal-mlu"}
MODEL_BASE=${MODEL_BASE:-${RESULT_BASE}/models}
PYTHON=${PYTHON:-python}

ADMM_STEPS=${ADMM_STEPS:-0}
LEARNING_RATE=${LEARNING_RATE:-0.001}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-10}
SEED=${SEED:-42}
NUM_PATHS_PER_PAIR=${NUM_PATHS_PER_PAIR:-4}
COMA_SAMPLES=${COMA_SAMPLES:-5}
LAYERS=${LAYERS:-6}
RHO=${RHO:-1.0}
DEVICE_ID=${DEVICE_ID:-0}
DATASETS=${DATASETS:-"DynGEANT geant abilene"}
NUM_CLUSTERS=${NUM_CLUSTERS:-50}
NUM_TRAIN_CLUSTERS=${NUM_TRAIN_CLUSTERS:-30}
NUM_VAL_CLUSTERS=${NUM_VAL_CLUSTERS:-5}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-0.75}
MAX_DATASET_SAMPLES=${MAX_DATASET_SAMPLES:-0}
MODEL_SAVE=${MODEL_SAVE:-false}
MODEL_LOAD=${MODEL_LOAD:-false}

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Python environment not found: ${PYTHON}" >&2
  exit 1
fi

if [[ "${DATA_BASE}" == *data_kaete* ]]; then
  echo "Refusing KaeTE throughput labels for an MLU run: ${DATA_BASE}" >&2
  echo "Set DATA_BASE to the MLU-labelled dataset root." >&2
  exit 1
fi

run_teal() {
  local topo_name="$1"
  local lr="$2"
  local batch_size="$3"
  local epochs="$4"
  local admm_steps="${5:-${ADMM_STEPS}}"
  local data_dir="${DATA_BASE}/${topo_name}"

  if [[ ! -d "${data_dir}" ]]; then
    echo "Dataset not found: ${data_dir}" >&2
    return 1
  fi

  "${PYTHON}" teal.py \
         --data_dir "${data_dir}" \
         --topo_name "${topo_name}" \
         --obj mlu \
         --epochs "${epochs}" \
         --lr "${lr}" \
         --bsz "${batch_size}" \
         --admm-steps "${admm_steps}" \
         --seed "${SEED}" \
         --num_paths_per_pair "${NUM_PATHS_PER_PAIR}" \
         --samples "${COMA_SAMPLES}" \
         --layers "${LAYERS}" \
         --rho "${RHO}" \
         --devid "${DEVICE_ID}" \
         --num-clusters "${NUM_CLUSTERS}" \
         --num-train-clusters "${NUM_TRAIN_CLUSTERS}" \
         --num-val-clusters "${NUM_VAL_CLUSTERS}" \
         --train-test-split "${TRAIN_TEST_SPLIT}" \
         --max-dataset-samples "${MAX_DATASET_SAMPLES}" \
         --model-save "${MODEL_SAVE}" \
         --model-load "${MODEL_LOAD}" \
         --result-dir "${RESULT_BASE}" \
         --model-dir "${MODEL_BASE}"

}

cd "${ROOT_DIR}/run"
for dataset in ${DATASETS}; do
  run_teal "${dataset}" "${LEARNING_RATE}" "${BATCH_SIZE}" "${EPOCHS}"
done
