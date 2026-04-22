#!/bin/bash
set -euo pipefail

device=0
python_bin=python

TRAIN_N3V=true
TRAIN_DNERF=true

n3v_root="/root/autodl-tmp/data/N3V"
dnerf_root="/root/autodl-tmp/data/dnerf"

n3v_config="./configs/n3v/default.yaml"
dnerf_config="./configs/dnerf/default.yaml"

n3v_scenes=(
    # "coffee_martini"
    # "cook_spinach"
    # "cut_roasted_beef"
    # "flame_salmon"
    # "flame_steak"
    # "sear_steak"
)

dnerf_scenes=(
    #  "bouncingballs"
    # "hellwarrior"
    # "hook"
    # "jumpingjacks"
    # "lego"
    # "mutant"
     "standup"
    # "trex"
)

method_args=()

train_one_scene () {
    local dataset_name="$1"
    local source_path="$2"
    local model_path="$3"
    local config_path="$4"

    mkdir -p "${model_path}"

    echo "========================================"
    echo "[TRAIN]"
    echo "Dataset : ${dataset_name}"
    echo "Config  : ${config_path}"
    echo "Source  : ${source_path}"
    echo "Output  : ${model_path}"
    echo "GPU     : ${device}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=${device} ${python_bin} main.py \
        --config "${config_path}" \
        --source_path "${source_path}" \
        --model_path "${model_path}" \
        "${method_args[@]}"
}

run_dataset_group () {
    local enable_flag="$1"
    local dataset_name="$2"
    local data_root="$3"
    local config_path="$4"
    shift 4
    local scenes=("$@")

    if [ "${enable_flag}" != true ]; then
        echo "[Skip Dataset] ${dataset_name}"
        return
    fi

    for scene in "${scenes[@]}"; do
        local source_path="${data_root}/${scene}"
        local model_path="output/${dataset_name}/${scene}"

        if [ ! -d "${source_path}" ]; then
            echo "[Skip Scene] ${source_path} not found"
            continue
        fi

        train_one_scene "${dataset_name}" "${source_path}" "${model_path}" "${config_path}"
    done
}

#run_dataset_group "${TRAIN_N3V}"   "N3V"   "${n3v_root}"   "${n3v_config}"   "${n3v_scenes[@]}"
run_dataset_group "${TRAIN_DNERF}" "DNeRF" "${dnerf_root}" "${dnerf_config}" "${dnerf_scenes[@]}"