DATA_BASE=/workspace/NetAI/data
ADMM_STEPS=${ADMM_STEPS:-0}

run_teal() {
  local topo_name="$1"
  local lr="$2"
  local batch_size="$3"
  local epochs="$4"
  local admm_steps="${5:-${ADMM_STEPS}}"
  local data_dir="${DATA_BASE}/${topo_name}"

  python teal.py \
         --data_dir ${data_dir} \
         --topo_name ${topo_name} \
         --obj mlu \
         --epochs ${epochs} \
         --lr ${lr} \
         --bsz ${batch_size} \
         --admm-steps ${admm_steps}

}

cd ./run
#run_teal DynGEANT 0.001 16 3 3
run_teal DynGEANT 0.001 16 10
run_teal geant 0.001 16 10
run_teal abilene 0.001 16 10
