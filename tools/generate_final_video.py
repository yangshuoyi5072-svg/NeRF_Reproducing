import os
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

def generate_video(frames_dir, output_path, pattern, num_frames, fps=30):
    print(f"Creating video: {output_path} ({num_frames} frames)")
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    
    for i in tqdm(range(num_frames)):
        filename = pattern % i
        filepath = os.path.join(frames_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} missing!")
            continue
        
        img = imageio.imread(filepath)
        writer.append_data(img)
    
    writer.close()
    print(f"Done: {output_path}")

if __name__ == "__main__":
    base_dir = "logs/lego_v8_mask_ds_view8/cinematic_submission"
    frames_dir = os.path.join(base_dir, "rgb_frames")
    
    # 1. Generate 15s RGB (450 frames)
    generate_video(frames_dir, os.path.join(base_dir, "lego_v8_mask_rgb_15s.mp4"), "%03d.png", 450)
    
    # 2. Generate 5s ACC (150 frames)
    generate_video(frames_dir, os.path.join(base_dir, "lego_v8_mask_acc_5s.mp4"), "%03d_acc.png", 150)
