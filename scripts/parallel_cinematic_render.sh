#!/bin/bash
# parallel_cinematic_render.sh
# Dual-worker parallel rendering for the 15s RGB cinematic video.
# Uses --no-acc to maximize speed and partitions 450 frames among 3 workers.

EXP_NAME="chair_vanilla_70k"
CKPT="logs/${EXP_NAME}/060000.tar"
ENV_PYTHON="/opt/miniconda3/envs/nerf_env/bin/python"
WORKERS=3

# Get the root directory (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Launching ${WORKERS}-Worker Cinematic Renderer (No-ACC Mode)..."

for i in $(seq 0 $((WORKERS-1))); do
    echo "Starting Worker $i..."
    $ENV_PYTHON "$ROOT_DIR/render_cinematic.py" \
        --config "$ROOT_DIR/logs/${EXP_NAME}/args.txt" \
        --ckpt "$ROOT_DIR/$CKPT" \
        --expname $EXP_NAME \
        --zoom 1.0 \
        --no-acc \
        --worker-id $i \
        --total-workers $WORKERS \
        --chunk 32768 > worker_cinematic_${i}.log 2>&1 &
done

echo "All workers launched. Check worker_cinematic_*.log for progress."
