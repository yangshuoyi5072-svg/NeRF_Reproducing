#!/bin/bash
# rendering loop

EXP_NAME="hotdog_viz_100k"
LOG_DIR="./logs/${EXP_NAME}"
OUTPUT_DIR="${LOG_DIR}/evolution_images"
ENV_PYTHON="/opt/miniconda3/envs/nerf_env/bin/python"

mkdir -p "$OUTPUT_DIR"

echo "Rendering evolution..."

# Checkpoints
CKPTS=$(ls ${LOG_DIR}/*.tar | sort -V)

for ckpt in $CKPTS; do
    ITER=$(basename "$ckpt" .tar)
    IMAGE_PATH="${OUTPUT_DIR}/${ITER}.png"
    
    # Skip
    if [ -f "$IMAGE_PATH" ]; then
        echo "Skipping ${ITER}"
        continue
    fi
    
    echo "Rendering: ${ITER}..."
    $ENV_PYTHON render_cinematic.py \
        --config "${LOG_DIR}/args.txt" \
        --ckpt "$ckpt" \
        --expname "$EXP_NAME" \
        --single-frame \
        --gt-pose \
        --zoom 1.0 \
        --chunk 32768 > /dev/null 2>&1
        
    # move result
    mv "${LOG_DIR}/cinematic_submission/rgb_frames/000.png" "$IMAGE_PATH"
done

echo "Done"
