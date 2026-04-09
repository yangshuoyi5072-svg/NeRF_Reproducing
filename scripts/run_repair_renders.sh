#!/bin/bash
# fix failed renders

# Configuration and paths
PYTHON_BIN="/opt/miniconda3/envs/nerf_env/bin/python"
CHUNK_SIZE=8192

# models to fix
MODELS=(
    "logs/lego_v8_vanilla/config.txt|logs/lego_v8_vanilla/010000.tar|lego_v8_vanilla"
    "logs/lego_v8_mask_ds_view8/config.txt|logs/lego_v8_mask_ds_view8/010000.tar|lego_v8_mask_ds_view8"
)

echo "Repair rendering..."

for MODEL_INFO in "${MODELS[@]}"; do
    CONFIG=$(echo $MODEL_INFO | cut -d'|' -f1)
    CKPT=$(echo $MODEL_INFO | cut -d'|' -f2)
    EXPNAME=$(echo $MODEL_INFO | cut -d'|' -f3)
    
    echo "---------------------------------------------------"
    echo "Rendering model: $EXPNAME"
    
    $PYTHON_BIN render_cinematic.py --config "$CONFIG" --ckpt "$CKPT" --expname "$EXPNAME" --fast --chunk $CHUNK_SIZE
    
    echo "Finished rendering $EXPNAME"
done

echo "Repair finished"
