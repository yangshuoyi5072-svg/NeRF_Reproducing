#!/bin/bash
# parallel render for hotdog

EXP_NAME="hotdog_viz_100k"
LOG_DIR="./logs/${EXP_NAME}"
OUTPUT_DIR="${LOG_DIR}/evolution_images"
ENV_PYTHON="/opt/miniconda3/envs/nerf_env/bin/python"

mkdir -p "$OUTPUT_DIR"

# Worker function
render_worker() {
    local worker_id=$1
    local total_workers=$2
    local ckpt_list=($3)
    
    echo "Worker ${worker_id} started."
    
    for i in "${!ckpt_list[@]}"; do
        # Split work
        if (( i % total_workers == worker_id )); then
            local ckpt="${ckpt_list[$i]}"
            local iter=$(basename "$ckpt" .tar)
            local image_path="${OUTPUT_DIR}/${iter}.png"
            
            # Skip existing
            if [ -f "$image_path" ]; then
                continue
            fi
            
            echo "Worker ${worker_id} -> Rendering ${iter}..."
            $ENV_PYTHON render_cinematic.py \
                --config "${LOG_DIR}/args.txt" \
                --ckpt "$ckpt" \
                --expname "$EXP_NAME" \
                --single-frame \
                --gt-pose \
                --zoom 1.0 \
                --chunk 32768 > /dev/null 2>&1
            
            mv "${LOG_DIR}/cinematic_submission/rgb_frames/000.png" "$image_path"
        fi
    done
    echo "Worker ${worker_id} finished."
}

# All checkpoints
CKPTS=$(ls ${LOG_DIR}/*.tar | sort -V)

echo "Starting Parallel Evolution Rendering (2 Workers)..."

# Launch 2 workers in background
render_worker 0 2 "$CKPTS" > worker_0.log 2>&1 &
render_worker 1 2 "$CKPTS" > worker_1.log 2>&1 &

echo "Parallel workers launched. Monitoring logs: worker_0.log, worker_1.log"
