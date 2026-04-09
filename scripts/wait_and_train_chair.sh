#!/bin/bash
# wait_and_train_chair.sh
# Sequences Chair 70k Vanilla Training after Hotdog Evolution Rendering finishes.

ENV_PYTHON="/opt/miniconda3/envs/nerf_env/bin/python"
EXP_NAME="chair_vanilla_70k"

echo "Chair 70k Sequencer Online."
echo "Waiting for Parallel Hotdog Evolution Rendering to finish..."

# Wait until ALL instances of parallel_viz_render or render_worker are gone
while pgrep -f "viz_render_loop.sh|parallel_viz_render.sh|render_worker" > /dev/null; do
    sleep 120
done

echo "Hotdog evolution rendering finished! Starting Chair 70k Vanilla Training..."

# 1. Start Training (70k steps, Weights every 10k)
$ENV_PYTHON run_nerf.py \
    --config configs/chair_vanilla_70k.txt \
    --N_iters 70000 \
    --i_weights 10000 \
    --expname $EXP_NAME > output_chair_vanilla_70k_training.log 2>&1

echo "Training complete. Starting final 15s RGB Cinematic Rendering..."

# 2. Start Final Rendering (15s RGB)
# Note: render_cinematic handles RGB and ACC logic internally.
$ENV_PYTHON render_cinematic.py \
    --config configs/chair_vanilla_70k.txt \
    --ckpt logs/${EXP_NAME}/070000.tar \
    --expname $EXP_NAME \
    --zoom 1.0 \
    --chunk 32768 > output_chair_vanilla_70k_render.log 2>&1

echo "All tasks complete! Chair 70k video is ready."
