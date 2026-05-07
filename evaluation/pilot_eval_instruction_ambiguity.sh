#!/bin/bash
# Pilot evaluation script for instruction ambiguity analysis using VLM.
# This script evaluates whether navigation instructions contain directional
# or docking ambiguity by analyzing ground-truth trajectory frames with a VLM.
#
# Prerequisites:
#   - Set your OpenAI API key as an environment variable: export OPENAI_API_KEY="your-key"
#   - Or pass it via --api_key argument
#
# Usage: bash evaluation/pilot_eval_instruction_ambiguity.sh

set -x

EVAL_DIR=$(cd "$(dirname "$0")" && pwd)

# ===== Configuration =====
# Path to summary.json containing episode metadata (instruction, scene_id, episode_id)
SUMMARY_JSON=${EVAL_DIR}/cache/result_r2r_success_streamvln_100/summary.json

# Root directory containing episode image folders (named by episode_id)
IMAGES_DIR=${EVAL_DIR}/cache/result_r2r_success_streamvln_100

# Output JSONL file for ambiguity evaluation results
OUTPUT=${EVAL_DIR}/cache/ambiguity_results_success_gpt5.jsonl

# VLM model to use (e.g., gpt-4o, claude-opus-4-7, gemini-2.5-pro)
MODEL="gpt-5"

# API key (set via environment variable or pass explicitly here)
API_KEY="${OPENAI_API_KEY}"

# Optional: custom API base URL (leave empty for default OpenAI endpoint)
BASE_URL="https://api2.aigcbest.top/v1"

# Number of frames to sample from each episode trajectory
NUM_FRAMES=10
# =======================

python ${EVAL_DIR}/pilot_eval_instruction_ambiguity.py \
    --summary_json ${SUMMARY_JSON} \
    --images_dir ${IMAGES_DIR} \
    --output ${OUTPUT} \
    --model ${MODEL} \
    --api_key ${API_KEY} \
    ${BASE_URL:+--base_url ${BASE_URL}} \
    --num_frames ${NUM_FRAMES}



# ===== Configuration =====
# Path to summary.json containing episode metadata (instruction, scene_id, episode_id)
SUMMARY_JSON=${EVAL_DIR}/cache/result_r2r_success_streamvln_100/summary.json

# Root directory containing episode image folders (named by episode_id)
IMAGES_DIR=${EVAL_DIR}/cache/result_r2r_success_streamvln_100

# Output JSONL file for ambiguity evaluation results
OUTPUT=${EVAL_DIR}/cache/ambiguity_results_success_gemini-2.5-pro.jsonl

# VLM model to use (e.g., gpt-4o, claude-opus-4-7, gemini-2.5-pro)
MODEL="gemini-2.5-pro"

# API key (set via environment variable or pass explicitly here)
API_KEY="${OPENAI_API_KEY}"

# Optional: custom API base URL (leave empty for default OpenAI endpoint)
BASE_URL="https://api2.aigcbest.top/v1"

# Number of frames to sample from each episode trajectory
NUM_FRAMES=10
# =======================

python ${EVAL_DIR}/pilot_eval_instruction_ambiguity.py \
    --summary_json ${SUMMARY_JSON} \
    --images_dir ${IMAGES_DIR} \
    --output ${OUTPUT} \
    --model ${MODEL} \
    --api_key ${API_KEY} \
    ${BASE_URL:+--base_url ${BASE_URL}} \
    --num_frames ${NUM_FRAMES}



# ===== Configuration =====
# Path to summary.json containing episode metadata (instruction, scene_id, episode_id)
SUMMARY_JSON=${EVAL_DIR}/cache/result_r2r_success_streamvln_100/summary.json

# Root directory containing episode image folders (named by episode_id)
IMAGES_DIR=${EVAL_DIR}/cache/result_r2r_success_streamvln_100

# Output JSONL file for ambiguity evaluation results
OUTPUT=${EVAL_DIR}/cache/ambiguity_results_success_claude-opus-4-7.jsonl

# VLM model to use (e.g., gpt-4o, claude-opus-4-7, gemini-2.5-pro)
MODEL="claude-opus-4-7"

# API key (set via environment variable or pass explicitly here)
API_KEY="${OPENAI_API_KEY}"

# Optional: custom API base URL (leave empty for default OpenAI endpoint)
BASE_URL="https://api2.aigcbest.top/v1"

# Number of frames to sample from each episode trajectory
NUM_FRAMES=10
# =======================

python ${EVAL_DIR}/pilot_eval_instruction_ambiguity.py \
    --summary_json ${SUMMARY_JSON} \
    --images_dir ${IMAGES_DIR} \
    --output ${OUTPUT} \
    --model ${MODEL} \
    --api_key ${API_KEY} \
    ${BASE_URL:+--base_url ${BASE_URL}} \
    --num_frames ${NUM_FRAMES}




# ===== Configuration =====
# Path to summary.json containing episode metadata (instruction, scene_id, episode_id)
SUMMARY_JSON=${EVAL_DIR}/cache/result_r2r_failure_streamvln_100/summary.json

# Root directory containing episode image folders (named by episode_id)
IMAGES_DIR=${EVAL_DIR}/cache/result_r2r_failure_streamvln_100

# Output JSONL file for ambiguity evaluation results
OUTPUT=${EVAL_DIR}/cache/ambiguity_results_failure_gpt5.jsonl

# VLM model to use (e.g., gpt-4o, claude-opus-4-7, gemini-2.5-pro)
MODEL="gpt-5"

# API key (set via environment variable or pass explicitly here)
API_KEY="${OPENAI_API_KEY}"

# Optional: custom API base URL (leave empty for default OpenAI endpoint)
BASE_URL="https://api2.aigcbest.top/v1"

# Number of frames to sample from each episode trajectory
NUM_FRAMES=10
# =======================

python ${EVAL_DIR}/pilot_eval_instruction_ambiguity.py \
    --summary_json ${SUMMARY_JSON} \
    --images_dir ${IMAGES_DIR} \
    --output ${OUTPUT} \
    --model ${MODEL} \
    --api_key ${API_KEY} \
    ${BASE_URL:+--base_url ${BASE_URL}} \
    --num_frames ${NUM_FRAMES}



# ===== Configuration =====
# Path to summary.json containing episode metadata (instruction, scene_id, episode_id)
SUMMARY_JSON=${EVAL_DIR}/cache/result_r2r_failure_streamvln_100/summary.json

# Root directory containing episode image folders (named by episode_id)
IMAGES_DIR=${EVAL_DIR}/cache/result_r2r_failure_streamvln_100

# Output JSONL file for ambiguity evaluation results
OUTPUT=${EVAL_DIR}/cache/ambiguity_results_failure_gemini-2.5-pro.jsonl

# VLM model to use (e.g., gpt-4o, claude-opus-4-7, gemini-2.5-pro)
MODEL="gemini-2.5-pro"

# API key (set via environment variable or pass explicitly here)
API_KEY="${OPENAI_API_KEY}"

# Optional: custom API base URL (leave empty for default OpenAI endpoint)
BASE_URL="https://api2.aigcbest.top/v1"

# Number of frames to sample from each episode trajectory
NUM_FRAMES=10
# =======================

python ${EVAL_DIR}/pilot_eval_instruction_ambiguity.py \
    --summary_json ${SUMMARY_JSON} \
    --images_dir ${IMAGES_DIR} \
    --output ${OUTPUT} \
    --model ${MODEL} \
    --api_key ${API_KEY} \
    ${BASE_URL:+--base_url ${BASE_URL}} \
    --num_frames ${NUM_FRAMES}



# ===== Configuration =====
# Path to summary.json containing episode metadata (instruction, scene_id, episode_id)
SUMMARY_JSON=${EVAL_DIR}/cache/result_r2r_failure_streamvln_100/summary.json

# Root directory containing episode image folders (named by episode_id)
IMAGES_DIR=${EVAL_DIR}/cache/result_r2r_failure_streamvln_100

# Output JSONL file for ambiguity evaluation results
OUTPUT=${EVAL_DIR}/cache/ambiguity_results_failure_claude-opus-4-7.jsonl

# VLM model to use (e.g., gpt-4o, claude-opus-4-7, gemini-2.5-pro)
MODEL="claude-opus-4-7"

# API key (set via environment variable or pass explicitly here)
API_KEY="${OPENAI_API_KEY}"

# Optional: custom API base URL (leave empty for default OpenAI endpoint)
BASE_URL="https://api2.aigcbest.top/v1"

# Number of frames to sample from each episode trajectory
NUM_FRAMES=10
# =======================

python ${EVAL_DIR}/pilot_eval_instruction_ambiguity.py \
    --summary_json ${SUMMARY_JSON} \
    --images_dir ${IMAGES_DIR} \
    --output ${OUTPUT} \
    --model ${MODEL} \
    --api_key ${API_KEY} \
    ${BASE_URL:+--base_url ${BASE_URL}} \
    --num_frames ${NUM_FRAMES}


