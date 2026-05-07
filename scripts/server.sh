#!/bin/bash
# Launch multiple StereoVLN inference server instances, one per GPU.
# Each instance listens on a separate port for parallel evaluation.
#
# Usage: bash scripts/server.sh
#
# Requirements:
#   - CHECKPOINT_PATH must point to a directory containing:
#       config.json, pytorch_model*.bin
#   - NUM_GPUS should match the number of available GPUs and the
#     NUM_SERVERS value in evaluation scripts (e.g., accuracy_eval_p_exact.sh)

# Path to the model checkpoint directory (edit before running)
CHECKPOINT_PATH="/path/to/your/checkpoint"

# Stereo camera baseline distance in meters (default: 0.1m = 10cm)
CAMERA_BASELINE=0.1

# Host address for the Flask servers
HOST=0.0.0.0

# Base port; server i will listen on BASE_PORT+i
BASE_PORT=7200

# Number of GPU workers (must match NUM_SERVERS in the eval script)
NUM_GPUS=8

echo "Launching $NUM_GPUS server instances (ports ${BASE_PORT}-$((BASE_PORT+NUM_GPUS-1)))..."

for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
    PORT=$((BASE_PORT + GPU_ID))
    echo "Starting server on GPU $GPU_ID, port $PORT"
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/server.py \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --camera_baseline "$CAMERA_BASELINE" \
        --port "$PORT" \
        --host "$HOST" \
        --gpu 0 &
done

# Wait for all background server processes to finish
wait