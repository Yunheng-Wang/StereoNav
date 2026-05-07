#!/bin/bash
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
set -x
umask 000

# Get script directory and change to it
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

# R2R dataset
DATASET=R2R
CONFIG_PATH=../config/gen_data_r2r.yaml
DATA_PATH=../data/task/r2r/train/train.json.gz
OUTPUT_PATH=cache/train/${DATASET}
mkdir -p ${OUTPUT_PATH}
torchrun --nproc_per_node=16 --master_port=$((RANDOM % 101 + 20000)) ./gen_trajectory.py \
    --dataset ${DATASET} --config_path ${CONFIG_PATH} --output_path ${OUTPUT_PATH} \
    --data_path ${DATA_PATH} --render_points > ${OUTPUT_PATH}/log.log 2>&1

# RxR dataset
DATASET=RxR
CONFIG_PATH=../config/gen_data_rxr.yaml
DATA_PATH=../data/task/rxr/train/train_guide.json.gz
OUTPUT_PATH=cache/train/${DATASET}
mkdir -p ${OUTPUT_PATH}
torchrun --nproc_per_node=16 --master_port=$((RANDOM % 101 + 20000)) ./gen_trajectory.py \
    --dataset ${DATASET} --config_path ${CONFIG_PATH} --output_path ${OUTPUT_PATH} \
    --data_path ${DATA_PATH} --render_points > ${OUTPUT_PATH}/log.log 2>&1

# ScaleVLN dataset
DATASET=ScaleVLN
CONFIG_PATH=../config/gen_data_r2r.yaml
DATA_PATH=../data/task/scalevln/scalevln_subset_150k.json.gz
OUTPUT_PATH=cache/train/${DATASET}
mkdir -p ${OUTPUT_PATH}
torchrun --nproc_per_node=16 --master_port=$((RANDOM % 101 + 20000)) ./gen_trajectory.py \
    --dataset ${DATASET} --config_path ${CONFIG_PATH} --output_path ${OUTPUT_PATH} \
    --data_path ${DATA_PATH} --render_points > ${OUTPUT_PATH}/log.log 2>&1
