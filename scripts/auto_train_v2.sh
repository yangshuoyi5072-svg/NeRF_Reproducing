#!/bin/bash
# Physics-Corrected Training

export PYTHONPATH=$PYTHONPATH:.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Train Chair Mask
echo "Restarting Mask..."
/opt/miniconda3/envs/nerf_env/bin/python "$ROOT_DIR/run_nerf.py" \
    --config "$ROOT_DIR/configs/chair.txt" \
    --num_views 8 \
    --N_iters 20000 \
    --use_depth \
    --depth_lambda 0.1 \
    --expname chair_v8_mask_physics_fix \
    --no_reload > output_chair_mask_v2.log 2>&1

# Train Chair Vanilla
echo "Restarting Vanilla..."
/opt/miniconda3/envs/nerf_env/bin/python "$ROOT_DIR/run_nerf.py" \
    --config "$ROOT_DIR/configs/chair.txt" \
    --num_views 8 \
    --N_iters 20000 \
    --expname chair_v8_vanilla_physics_fix \
    --no_reload > output_chair_vanilla_v2.log 2>&1

echo "Chair Physics-Fix Training Complete!"
