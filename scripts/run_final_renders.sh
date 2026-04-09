#!/bin/bash
# render all
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="/opt/miniconda3/envs/nerf_env/bin/python"
CHUNK_SIZE=8192

# models
MODELS=(
    "logs/blender_paper_lego/config.txt|blender_paper_lego"
    "logs/lego_v8_vanilla/config.txt|lego_v8_vanilla"
    "logs/lego_v8_mask_ds_view8_ds_view8/config.txt|lego_v8_mask_ds_view8_ds_view8"
)

echo "Rendering all models..."

for MODEL_INFO in "${MODELS[@]}"; do
    CONFIG=$(echo $MODEL_INFO | cut -d'|' -f1)
    EXPNAME=$(echo $MODEL_INFO | cut -d'|' -f2)
    
    echo "---------------------------------------------------"
    echo "Rendering model: $EXPNAME"
    echo "Using config: $CONFIG"
    
    $PYTHON_BIN "$ROOT_DIR/render_cinematic.py" --config "$ROOT_DIR/$CONFIG" --expname "$EXPNAME" --fast --chunk $CHUNK_SIZE
    
    echo "Finished rendering $EXPNAME"
done

echo "Finished"
