# 修改这里选择数据集: R2R, RXR, EnvDrop
DATASET="R2R"

if [ "$DATASET" = "R2R" ]; then
    CONFIG_PATH="../config/gen_data_r2r.yaml"
    DATA_PATH="task/r2r/val_unseen/val_unseen.json.gz"
elif [ "$DATASET" = "RXR" ]; then
    CONFIG_PATH="../config/gen_data_rxr.yaml"
    DATA_PATH="task/rxr/train/train_guide.json.gz"
else
    CONFIG_PATH="../config/gen_data_rxr.yaml"
    DATA_PATH="task/envdrop/envdrop.json.gz"
fi

SPLIT="val_unseen"
OUTPUT_DIR="cache_world/${SPLIT}/${DATASET}"

python ./gen_world_data.py \
    --habitat_config_path "$CONFIG_PATH" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --data_path "$DATA_PATH" \
    --dataset "$DATASET"

# cd data
# bash gen_world_data.sh
