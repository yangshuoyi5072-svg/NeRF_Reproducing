#!/bin/bash

# Configuration and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="/opt/miniconda3/envs/nerf_env/bin/python"
CHUNK_SIZE=8192

# List of models to render
# Format: config_path|expname
MODELS=(
    "logs/blender_paper_lego/config.txt|blender_paper_lego"
    "logs/lego_v8_vanilla/config.txt|lego_v8_vanilla"
    "logs/lego_v8_mask_ds_view8_ds_view8/config.txt|lego_v8_mask_ds_view8_ds_view8"
)

echo "Starting final cinematic rendering task..."
echo "Total models: ${#MODELS[@]}"
echo "Resolution: 400x400 (optimized for speed)"
echo "Speed mode: --fast (N_importance=64)"

for MODEL_INFO in "${MODELS[@]}"; do
    CONFIG=$(echo $MODEL_INFO | cut -d'|' -f1)
    EXPNAME=$(echo $MODEL_INFO | cut -d'|' -f2)
    
    echo "---------------------------------------------------"
    echo "Rendering model: $EXPNAME"
    echo "Using config: $CONFIG"
    
    $PYTHON_BIN "$ROOT_DIR/render_cinematic.py" --config "$ROOT_DIR/$CONFIG" --expname "$EXPNAME" --fast --chunk $CHUNK_SIZE
    
    echo "Finished rendering $EXPNAME"
done

echo "All rendering tasks completed successfully."
