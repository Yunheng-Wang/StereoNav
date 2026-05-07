#!/bin/bash
# DAgger data generation for R2R and RxR datasets
# Usage: bash data/dagger.sh

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
set -x
umask 000

# Get script directory and change to it
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

# DAgger parameters
DAGGER_UPDATE_SIZE=160000
DAGGER_COMMIT_FREQ=50
DAGGER_P=0.9            # 0 = pure expert; >0 = mix model
DAGGER_DATA_IT=1        # iteration index (used when DAGGER_P > 0)

# Server URLs (must match server.sh)
NUM_SERVERS=8
BASE_PORT=7200
SERVER_URLS=""
for i in $(seq 0 $((NUM_SERVERS-1))); do
    [ -n "$SERVER_URLS" ] && SERVER_URLS="${SERVER_URLS},"
    SERVER_URLS="${SERVER_URLS}http://localhost:$((BASE_PORT+i))"
done

# R2R dataset
DATASET=R2R
HABITAT_CONFIG_PATH=../config/gen_data_r2r.yaml
DATA_PATH=../data/task/r2r/train/train.json.gz
OUTPUT_PATH=cache/train/dagger/${DATASET}
mkdir -p ${OUTPUT_PATH}
MASTER_PORT=$((RANDOM % 101 + 20000))

torchrun --nproc_per_node=8 --master_port=$MASTER_PORT dagger.py \
    --habitat_config_path ${HABITAT_CONFIG_PATH} \
    --dagger_dataset ${DATASET} \
    --dagger_data_path ${DATA_PATH} \
    --dagger_output_path ${OUTPUT_PATH} \
    --dagger_update_size ${DAGGER_UPDATE_SIZE} \
    --dagger_commit_freq ${DAGGER_COMMIT_FREQ} \
    --dagger_p ${DAGGER_P} \
    --dagger_data_it ${DAGGER_DATA_IT} \
    --server_urls ${SERVER_URLS} \
    --history_num 8 \
    --execute_steps 4 \
    --num_future_steps 1

# RxR dataset
DATASET=RxR
HABITAT_CONFIG_PATH=../config/gen_data_rxr.yaml
DATA_PATH=../data/task/rxr/train/train_guide.json.gz
OUTPUT_PATH=cache/train/dagger/${DATASET}
mkdir -p ${OUTPUT_PATH}
MASTER_PORT=$((RANDOM % 101 + 20000))

torchrun --nproc_per_node=8 --master_port=$MASTER_PORT dagger.py \
    --habitat_config_path ${HABITAT_CONFIG_PATH} \
    --dagger_dataset ${DATASET} \
    --dagger_data_path ${DATA_PATH} \
    --dagger_output_path ${OUTPUT_PATH} \
    --dagger_update_size ${DAGGER_UPDATE_SIZE} \
    --dagger_commit_freq ${DAGGER_COMMIT_FREQ} \
    --dagger_p ${DAGGER_P} \
    --dagger_data_it ${DAGGER_DATA_IT} \
    --server_urls ${SERVER_URLS} \
    --history_num 8 \
    --execute_steps 4 \
    --num_future_steps 1
