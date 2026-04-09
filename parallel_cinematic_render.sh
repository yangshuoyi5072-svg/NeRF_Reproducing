#!/bin/bash
# parallel_cinematic_render.sh
# Dual-worker parallel rendering for the 15s RGB cinematic video.
# Uses --no-acc to maximize speed and partitions 450 frames among 3 workers.

EXP_NAME="chair_vanilla_70k"
CKPT="logs/${EXP_NAME}/060000.tar"
ENV_PYTHON="/opt/miniconda3/envs/nerf_env/bin/python"
WORKERS=3

echo "Launching ${WORKERS}-Worker Cinematic Renderer (No-ACC Mode)..."

for i in $(seq 0 $((WORKERS-1))); do
    echo "Starting Worker $i..."
    $ENV_PYTHON render_cinematic.py \
        --config logs/${EXP_NAME}/args.txt \
        --ckpt $CKPT \
        --expname $EXP_NAME \
        --zoom 1.0 \
        --no-acc \
        --worker-id $i \
        --total-workers $WORKERS \
        --chunk 32768 > worker_cinematic_${i}.log 2>&1 &
done

echo "All workers launched. Check worker_cinematic_*.log for progress."
