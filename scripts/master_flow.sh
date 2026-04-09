#!/bin/bash
# Master Flow Script: The High-Frequency Hotdog Learning Experiment
# Orchestrates sequencing, training, and visualization.

export PYTHONPATH=$PYTHONPATH:.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_PYTHON="/opt/miniconda3/envs/nerf_env/bin/python"
PREV_PID=33078

echo "Master Flow Controller Online."
echo "Waiting for Previous Task (PID: $PREV_PID) to finish..."

# 1. Wait for preceding rendering task
while kill -0 $PREV_PID 2>/dev/null; do
    sleep 60
done

echo "Previous task finished! Starting Hotdog 100k Evolution Experiment..."

# 2. Start Training (100k iters, Save/Log every 200)
# We use vanilla settings (Basic code) as requested.
echo "Stating Training Phase (Approx 10-12 hours)..."
$ENV_PYTHON "$ROOT_DIR/run_nerf.py" \
    --config "$ROOT_DIR/configs/hotdog.txt" \
    --N_iters 100000 \
    --i_weights 200 \
    --i_print 200 \
    --expname hotdog_viz_100k > output_hotdog_viz_training.log 2>&1

echo "Training Complete! Starting Visualization Rendering Phase..."

# 3. Trigger Post-hoc Rendering Loop
chmod +x "$SCRIPT_DIR/viz_render_loop.sh"
"$SCRIPT_DIR/viz_render_loop.sh" > output_hotdog_viz_rendering.log 2>&1

echo "Master Flow Job Complete! Your evolution dataset is in logs/hotdog_viz_100k/evolution_images/"
