#!/bin/bash
# Quick hotdog demo (30-60s on Mac with MPS)
# Renders a single high-quality frame from the 100k model.

export PYTHONPATH=$PYTHONPATH:.
PYTHON_BIN="/opt/miniconda3/envs/nerf_env/bin/python"

echo "Running Hotdog 100k Demo..."
$PYTHON_BIN render_cinematic.py \
    --config final_models/hotdog_100k/args.txt \
    --ckpt final_models/hotdog_100k/weights.tar \
    --expname hotdog_100k_demo \
    --single-frame \
    --gt-pose \
    --zoom 1.0 \
    --chunk 32768

echo "------------------------------------------------"
echo "Demo complete! View the result at:"
echo "./logs/hotdog_100k_demo/cinematic_submission/rgb_frames/000.png"
