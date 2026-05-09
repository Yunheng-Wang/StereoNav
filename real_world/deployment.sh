set -e

# ===== Navigation Instruction =====
# Provide the natural language navigation instruction here.
# The instruction will be parsed by an LLM to extract a structured command
# and a target location (TARGET_X, TARGET_Y) in the robot's coordinate frame.
USER_INSTRUCTION="Walk forward through the hallway, turn left at the end, pass the sofa on your right, and stop in front of the wooden door. The target point is located 3 meters ahead of you and 6 meters to your left."

# ===== LLM API Configuration =====
# Used to parse the raw instruction into structured navigation commands.
# Set your API key and endpoint before running.
LLM_API_KEY="your-api-key-here"
LLM_BASE_URL="https://api.openai.com/v1"   # replace with your API endpoint
LLM_MODEL="gpt-4o"                          # replace with your preferred model

# ===== Network Configuration =====
# ZED_IP: IP address of the machine running the ZED stereo camera server.
# Adjust all URLs to match your deployment environment.
ZED_IP="192.168.x.x"                        # replace with your ZED camera server IP
ZED_URL="http://${ZED_IP}:8000"
ACTION_URL="http://127.0.0.1:8001"          # robot action server
SERVER_URL="http://127.0.0.1:7203"          # StereoNav inference server

# ===== Model Configuration =====
IMAGE_SIZE=448      # input image resolution
HISTORY_NUM=8       # number of historical frames to keep
EXECUTE_STEPS=4     # number of steps to execute per inference call
MAX_STEPS=500       # maximum total navigation steps before stopping



echo "Parsing instruction with LLM..."
eval $(python3 "$(dirname "$0")/preprocess_instruction.py" \
    --input "$USER_INSTRUCTION" \
    --api_key "$LLM_API_KEY" \
    --base_url "$LLM_BASE_URL" \
    --model "$LLM_MODEL")
echo "INSTRUCTION: $INSTRUCTION"
echo "TARGET_X: $TARGET_X"
echo "TARGET_Y: $TARGET_Y"




echo "Checking ZED server..."
curl --noproxy '*' -f "${ZED_URL}/status"

echo ""
echo "Checking local action server..."
curl --noproxy '*' -f "${ACTION_URL}/status"

echo ""
echo "Checking inference server..."
curl --noproxy '*' -f "${SERVER_URL}/" || true

echo ""
echo "Starting deployment.py..."




python3 "$(dirname "$0")/deployment.py" \
    --instruction "$INSTRUCTION" \
    --raw_instruction "$USER_INSTRUCTION" \
    --target "$TARGET_X" "$TARGET_Y" \
    --zed_ip "$ZED_IP" \
    --action_url "$ACTION_URL" \
    --server_url "$SERVER_URL" \
    --image_size "$IMAGE_SIZE" \
    --history_num "$HISTORY_NUM" \
    --execute_steps "$EXECUTE_STEPS" \
    --max_steps "$MAX_STEPS"
