#!/bin/bash
# Ground-truth trajectory data generation script for VLN evaluation.
# Generates navigation trajectories from Habitat simulator by following reference paths.
#
# Usage: bash evaluation/cache/gen_data.sh
# Note: Must be run from project root directory

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

set -x
umask 000

# Get script directory and project root
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

# Change to project root for relative path resolution
cd "${PROJECT_ROOT}"

# Configuration paths (relative to project root)
CONFIG_PATH="config/gen_data_r2r.yaml"
SCENES_DIR="data/scene"
DATA_PATH="data/task/r2r/val_unseen/val_unseen.json.gz"

# Output paths (relative to project root, saved under evaluation/cache/)
OUTPUT_PATH_SUCCESS="evaluation/cache/result_r2r_success_streamvln_100"
mkdir -p ${OUTPUT_PATH_SUCCESS}

torchrun --nproc_per_node=8 --master_port=$MASTER_PORT evaluation/cache/gen_data.py \
    --config_path ${CONFIG_PATH} \
    --output_path ${OUTPUT_PATH_SUCCESS} \
    --data_path ${DATA_PATH} \
    --scenes_dir ${SCENES_DIR} \
    --filter_json evaluation/cache/result_r2r_success_streamvln_100.json

OUTPUT_PATH_FAILURE="evaluation/cache/result_r2r_failure_streamvln_100"
mkdir -p ${OUTPUT_PATH_FAILURE}

torchrun --nproc_per_node=8 --master_port=$((MASTER_PORT+1)) evaluation/cache/gen_data.py \
    --config_path ${CONFIG_PATH} \
    --output_path ${OUTPUT_PATH_FAILURE} \
    --data_path ${DATA_PATH} \
    --scenes_dir ${SCENES_DIR} \
    --filter_json evaluation/cache/result_r2r_failure_streamvln_100.json
