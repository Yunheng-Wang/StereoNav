#!/bin/bash
# WorldModel Server Launcher
#
# NOTE: This script is part of an experimental attempt at automated goal-point
# prediction for vision-and-language navigation. The WorldModel predicts target
# waypoints from panoramic observations and language instructions.
# This is considered promising future work and is not yet integrated into the
# main StereoVLN evaluation pipeline.
#
# Usage: bash scripts/server_world.sh

# Path to the WorldModel checkpoint directory (edit before running)
MODEL_CHECKPOINT="/path/to/your/world_model_checkpoint"

# Base port; server i will listen on BASE_PORT+i
BASE_PORT=7300

# Number of GPU workers to launch
NUM_GPUS=8

# Host address for the Flask servers
HOST=0.0.0.0

echo "Launching $NUM_GPUS WorldModel server instances (ports ${BASE_PORT}-$((BASE_PORT+NUM_GPUS-1)))..."

for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
    PORT=$((BASE_PORT + GPU_ID))
    echo "Starting WorldModel server on GPU $GPU_ID, port $PORT"
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/server_world.py \
        --model_checkpoint "$MODEL_CHECKPOINT" \
        --port "$PORT" \
        --host "$HOST" \
        --gpu 0 &
done

wait
