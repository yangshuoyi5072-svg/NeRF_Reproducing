#!/bin/bash
# Sequential Rendering Script for Cinematic Suite production
# Prevents MPS memory contention by running tasks one by one.

export PYTHONPATH=$PYTHONPATH:.
ENV_PYTHON="/opt/miniconda3/envs/nerf_env/bin/python"

echo "Starting Sequential Rendering Pipeline..."

# 1. Lego v8 Vanilla (Zoom 1.0)
echo "[1/2] Rendering Lego v8 Vanilla..."
$ENV_PYTHON render_cinematic.py \
    --config logs/lego_v8_vanilla/args.txt \
    --ckpt logs/lego_v8_vanilla/010000.tar \
    --expname lego_v8_vanilla \
    --zoom 1.0 \
    --chunk 32768 > output_lego_v8_vanilla_render.log 2>&1

# 2. Chair v8 Vanilla (Zoom 1.0)
echo "[2/2] Rendering Chair v8 Vanilla..."
$ENV_PYTHON render_cinematic.py \
    --config logs/chair_v8_vanilla/args.txt \
    --ckpt logs/chair_v8_vanilla/020000.tar \
    --expname chair_v8_vanilla \
    --zoom 1.0 \
    --chunk 32768 > output_chair_v8_vanilla_render.log 2>&1

echo "Sequential Pipeline Complete!"
