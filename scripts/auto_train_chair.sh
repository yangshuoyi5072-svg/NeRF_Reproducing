#!/bin/bash
# Training script for chair few-shot

export PYTHONPATH=$PYTHONPATH:.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Train Chair Vanilla (8-view, 20k)
echo "Starting Chair Vanilla..."
    --config "$ROOT_DIR/configs/chair.txt" \
    --num_views 8 \
    --N_iters 20000 \
    --expname chair_v8_vanilla \
    --no_reload > output_chair_vanilla.log 2>&1

# Train Chair Mask/DS (8-view, 20k)
echo "Starting Chair Mask..."
    --config "$ROOT_DIR/configs/chair.txt" \
    --num_views 8 \
    --N_iters 20000 \
    --use_depth \
    --depth_lambda 0.1 \
    --expname chair_v8_mask \
    --no_reload > output_chair_mask.log 2>&1

echo "Chair Training Matrix Complete!"
