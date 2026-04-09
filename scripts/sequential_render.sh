#!/bin/bash
# Sequential Rendering

export PYTHONPATH=$PYTHONPATH:.
ENV_PYTHON="/opt/miniconda3/envs/nerf_env/bin/python"

echo "Sequential rendering..."

# Lego
echo "Lego..."
$ENV_PYTHON render_cinematic.py \
    --config logs/lego_v8_vanilla/args.txt \
    --ckpt logs/lego_v8_vanilla/010000.tar \
    --expname lego_v8_vanilla \
    --zoom 1.0 \
    --chunk 32768 > output_lego_v8_vanilla_render.log 2>&1

# Chair
echo "Chair..."
$ENV_PYTHON render_cinematic.py \
    --config logs/chair_v8_vanilla/args.txt \
    --ckpt logs/chair_v8_vanilla/020000.tar \
    --expname chair_v8_vanilla \
    --zoom 1.0 \
    --chunk 32768 > output_chair_v8_vanilla_render.log 2>&1

echo "Done"
