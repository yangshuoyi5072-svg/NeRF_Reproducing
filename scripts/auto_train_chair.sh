#!/bin/bash
# Automatic Training Script for Chair Few-Shot Ablation
# Each task runs for 20,000 iterations (approx. 1.5h per task on MPS)

export PYTHONPATH=$PYTHONPATH:.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# 1. Train Chair Vanilla (8-view, 20k)
echo "Starting Chair Vanilla (8-view, 20k iters)..."
/opt/miniconda3/envs/nerf_env/bin/python "$ROOT_DIR/run_nerf.py" \
    --config "$ROOT_DIR/configs/chair.txt" \
    --num_views 8 \
    --N_iters 20000 \
    --expname chair_v8_vanilla \
    --no_reload > output_chair_vanilla.log 2>&1

# 2. Train Chair Mask/DS (8-view, 20k)
echo "Starting Chair Mask (8-view, 20k iters)..."
/opt/miniconda3/envs/nerf_env/bin/python "$ROOT_DIR/run_nerf.py" \
    --config "$ROOT_DIR/configs/chair.txt" \
    --num_views 8 \
    --N_iters 20000 \
    --use_depth \
    --depth_lambda 0.1 \
    --expname chair_v8_mask \
    --no_reload > output_chair_mask.log 2>&1

echo "Chair Training Matrix Complete!"
