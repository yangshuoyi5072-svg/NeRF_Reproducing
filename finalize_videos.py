import imageio
import os
import glob

def compose(dir_path, video_name):
    frames = sorted(glob.glob(os.path.join(dir_path, "*.png")))
    if not frames: return
    # Filter RGB (no suffix) vs ACC (_acc suffix)
    rgb_files = [f for f in frames if "_acc" not in f]
    acc_files = [f for f in frames if "_acc" in f]
    
    if rgb_files:
        print(f"Creating RGB video: {video_name}.mp4")
        writer = imageio.get_writer(f"{video_name}.mp4", fps=30)
        for f in rgb_files:
            writer.append_data(imageio.imread(f))
        writer.close()
        
    if acc_files:
        # ACC path is usually only 5s (150 frames) in our cinematic script
        print(f"Creating ACC video: {video_name}_acc.mp4")
        writer = imageio.get_writer(f"{video_name}_acc.mp4", fps=30)
        for f in acc_files:
            writer.append_data(imageio.imread(f))
        writer.close()

models = ["lego_v8_vanilla", "chair_v8_vanilla", "hotdog_viz_100k"]
for m in models:
    path = os.path.join("logs", m, "cinematic_submission", "rgb_frames")
    compose(path, os.path.join("logs", m, f"{m}_cinematic"))
