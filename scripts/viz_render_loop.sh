#!/bin/bash
# High-frequency Post-hoc Rendering Script
# Loops through all checkpoints and renders a consistent view for evolution visualization.

EXP_NAME="hotdog_viz_100k"
LOG_DIR="./logs/${EXP_NAME}"
OUTPUT_DIR="${LOG_DIR}/evolution_images"
ENV_PYTHON="/opt/miniconda3/envs/nerf_env/bin/python"

mkdir -p "$OUTPUT_DIR"

echo "Starting Post-hoc Evolution Rendering for ${EXP_NAME}..."

# Sort checkpoints numerically
CKPTS=$(ls ${LOG_DIR}/*.tar | sort -V)

for ckpt in $CKPTS; do
    ITER=$(basename "$ckpt" .tar)
    IMAGE_PATH="${OUTPUT_DIR}/${ITER}.png"
    
    # Check if already rendered to allow resume
    if [ -f "$IMAGE_PATH" ]; then
        echo "Skipping ${ITER} (Already exists)"
        continue
    fi
    
    echo "Rendering frame for iteration: ${ITER}..."
    $ENV_PYTHON render_cinematic.py \
        --config "${LOG_DIR}/args.txt" \
        --ckpt "$ckpt" \
        --expname "$EXP_NAME" \
        --single-frame \
        --gt-pose \
        --zoom 1.0 \
        --chunk 32768 > /dev/null 2>&1
        
    # Move the result to our dedicated evolution folder
    # Note: render_cinematic.py saves frame 000.png in cinematic_submission/rgb_frames
    mv "${LOG_DIR}/cinematic_submission/rgb_frames/000.png" "$IMAGE_PATH"
done

echo "Evolution Visualization Finished!"
