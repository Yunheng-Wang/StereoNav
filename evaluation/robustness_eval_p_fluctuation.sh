#!/bin/bash
# Distributed robustness evaluation script for StereoVLN with camera height fluctuation.
# Spawns 8 torchrun workers, each connecting to a dedicated inference server.
# Note: this script only supports R2R evaluation.
#
# Prerequisites: start inference servers first via scripts/server.sh
# Usage: bash evaluation/robustness_eval_p_fluctuation.sh

# Suppress verbose simulator logs
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# Random master port to avoid conflicts when running multiple jobs
MASTER_PORT=$((RANDOM % 101 + 20000))

# Number of inference servers (must match NUM_GPUS in scripts/server.sh)
NUM_SERVERS=8
BASE_PORT=7200

# Build comma-separated server URL list: http://localhost:7200,...,http://localhost:7207
SERVER_URLS=""
for i in $(seq 0 $((NUM_SERVERS-1))); do
    if [ -n "$SERVER_URLS" ]; then
        SERVER_URLS="${SERVER_URLS},"
    fi
    SERVER_URLS="${SERVER_URLS}http://localhost:$((BASE_PORT+i))"
done

# Habitat config path for R2R evaluation
HABITAT_CONFIG_PATH="config/eval_r2r.yaml"

# Dataset split to evaluate on (val_seen / val_unseen / test)
EVAL_SPLIT="val_unseen"

# Directory where per-episode results and summary will be saved
OUTPUT_PATH="./results"

# Sampling temperature (0.0 = greedy decoding)
TEMPERATURE=0.0

# Nucleus sampling threshold (1.0 = disabled)
TOP_P=1.0

# Number of actions to execute per inference call
EXECUTE_STEPS=4

# Maximum steps per episode before forced stop (0 = unlimited)
MAX_STEPS=500

# Set to true to save top-down map videos for each episode
SAVE_VIDEO=false

# Number of historical frames fed to the model (must match training config)
HISTORY_NUM=8

# Camera height offset in meters (e.g., 0.6 or -0.6)
HEIGHT_OFFSET=-0.6

echo "Server URLs: ${SERVER_URLS}"
echo "Eval split: ${EVAL_SPLIT}"
echo "Output path: ${OUTPUT_PATH}"
echo "Height offset: ${HEIGHT_OFFSET}m"

CMD="torchrun --nproc_per_node=8 --master_port=$MASTER_PORT evaluation/robustness_eval_p_fluctuation.py \
    --server_urls $SERVER_URLS \
    --habitat_config_path $HABITAT_CONFIG_PATH \
    --output_path $OUTPUT_PATH \
    --eval_split $EVAL_SPLIT \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_steps $MAX_STEPS \
    --execute_steps $EXECUTE_STEPS \
    --history_num $HISTORY_NUM \
    --height_offset $HEIGHT_OFFSET"

if [ "$SAVE_VIDEO" = true ]; then
    CMD="$CMD --save_video"
fi

eval $CMD