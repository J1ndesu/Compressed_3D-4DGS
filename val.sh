#!/bin/bash

set -euo pipefail

device=0
ckpt_name="chkpnt6000.pth"

n3v_root="/root/autodl-tmp/data/N3V"
dnerf_root="/root/autodl-tmp/data/dnerf"

n3v_config="configs/n3v/default.yaml"
dnerf_config="configs/dnerf/default.yaml"

n3v_scenes=(
    "coffee_martini"
    "cook_spinach"
    "cut_roasted_beef"
    "flame_salmon"
    "flame_steak"
    "sear_steak"
)

dnerf_scenes=(
    # "bouncingballs"
    # "hellwarrior"
    # "hook"
    # "jumpingjacks"
    # "lego"
    "mutant"
    "standup"
    # "trex"
)

run_val_group () {
    local dataset_name="$1"
    local data_root="$2"
    local config_path="$3"
    shift 3
    local scenes=("$@")

    for scene in "${scenes[@]}"; do
        local source_path="${data_root}/${scene}"
        local model_path="output/${dataset_name}/${scene}"
        local checkpoint_path="${model_path}/${ckpt_name}"

        if [ ! -d "${source_path}" ]; then
            echo "[Skip] source not found: ${source_path}"
            continue
        fi

        if [ ! -f "${checkpoint_path}" ]; then
            echo "[Skip] checkpoint not found: ${checkpoint_path}"
            continue
        fi

        echo "========================================"
        echo "VAL Dataset       : ${dataset_name}"
        echo "VAL Scene         : ${scene}"
        echo "Config            : ${config_path}"
        echo "Source            : ${source_path}"
        echo "Model path        : ${model_path}"
        echo "Checkpoint        : ${checkpoint_path}"
        echo "GPU               : ${device}"
        echo "use_pruning       : True"
        echo "use_sh_adaptive   : True"
        echo "========================================"

        CUDA_VISIBLE_DEVICES=${device} python main.py \
            --config "${config_path}" \
            --source_path "${source_path}" \
            --model_path "${model_path}" \
            --start_checkpoint "${checkpoint_path}" \
            --val \
            --use_pruning \
            --use_sh_adaptive
    done
}

#run_val_group "N3V"   "${n3v_root}"   "${n3v_config}"   "${n3v_scenes[@]}"
run_val_group "DNeRF" "${dnerf_root}" "${dnerf_config}" "${dnerf_scenes[@]}"