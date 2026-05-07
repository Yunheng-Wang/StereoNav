set -e

# ===== 指令输入 =====
USER_INSTRUCTION="Walk forward through the hallway, turn left at the end, pass the sofa on your right, and stop in front of the wooden door. The target point is located 3 meters ahead of you and 6 meters to your left."

# ===== LLM API 配置 =====
LLM_API_KEY="sk-pGctX7L03Yw8TBk6NwyPxDSFaYrq4J0KxTyou5MuXfN5Rok4"
LLM_BASE_URL="https://api2.aigcbest.top/v1"
LLM_MODEL="gpt-5"

# ===== 端口配置 =====
ZED_IP="192.168.123.164"
ZED_URL="http://${ZED_IP}:8000"
ACTION_URL="http://127.0.0.1:8001"
SERVER_URL="http://127.0.0.1:7203"

# ===== 模型配置 =====
IMAGE_SIZE=448
HISTORY_NUM=8
EXECUTE_STEPS=4
MAX_STEPS=500




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