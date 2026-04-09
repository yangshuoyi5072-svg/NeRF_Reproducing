#!/bin/bash
# Physics-Corrected Training Script (V2)
# Enforces no_ndc = True for accurate synthetic reconstruction

export PYTHONPATH=$PYTHONPATH:.

# 1. Train Chair Mask/DS (8-view, 20k) - PRIORITY FIX
echo "Restarting Chair Mask (8-view, 20k iters) with NO_NDC=True..."
/opt/miniconda3/envs/nerf_env/bin/python run_nerf.py \
    --config configs/chair.txt \
    --num_views 8 \
    --N_iters 20000 \
    --use_depth \
    --depth_lambda 0.1 \
    --expname chair_v8_mask_physics_fix \
    --no_reload > output_chair_mask_v2.log 2>&1

# 2. Train Chair Vanilla (8-view, 20k)
echo "Restarting Chair Vanilla (8-view, 20k iters) with NO_NDC=True..."
/opt/miniconda3/envs/nerf_env/bin/python run_nerf.py \
    --config configs/chair.txt \
    --num_views 8 \
    --N_iters 20000 \
    --expname chair_v8_vanilla_physics_fix \
    --no_reload > output_chair_vanilla_v2.log 2>&1

echo "Chair Physics-Fix Training Complete!"
