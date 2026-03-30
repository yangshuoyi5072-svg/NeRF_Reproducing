import os
import glob
import json
import torch
import numpy as np
import imageio.v2 as imageio
import lpips
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return -10.0 * np.log10(mse)

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Initialize LPIPS
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    models = [
        "blender_paper_lego",
        "lego_v3_vanilla",
        "lego_v3_mask_ds_view3_ds_view3",
        "lego_v8_vanilla",
        "lego_v8_mask_ds_view8_ds_view8"
    ]
    
    dataset_dir = "./data/nerf_synthetic/lego"
    transforms_path = os.path.join(dataset_dir, "transforms_test.json")
    
    with open(transforms_path, 'r') as f:
        meta = json.load(f)
        
    # Get ground truth frames mapping with testskip=8
    frames = meta['frames'][::8]
    gt_paths = [os.path.join(dataset_dir, f['file_path'] + '.png') for f in frames]
    
    results = []
    
    for expname in models:
        exp_dir = os.path.join("./logs", expname)
        
        # Find the latest renderonly_test_ directory
        render_dirs = glob.glob(os.path.join(exp_dir, "renderonly_test_*"))
        
        if not render_dirs:
            print(f"\n[!] Model '{expname}' has no render directory.")
            print(f"    Missing renders. Please run the following command to generate them:")
            print(f"    python run_nerf.py --config configs/{expname}.txt --render_only --render_test")
            continue
            
        # Sort to get the latest by looking at the step number (e.g. renderonly_test_199999)
        # We can sort by name or modification time. Assuming name sorting gives the largest step count.
        render_dirs.sort()
        latest_render_dir = render_dirs[-1]
        
        # Determine training steps from folder name
        try:
            step_count = latest_render_dir.split('_')[-1]
            # Try to convert to int, if fails, just use the string
            int(step_count)
        except ValueError:
            step_count = "Unknown"
            
        rendered_images = sorted(glob.glob(os.path.join(latest_render_dir, "*.png")))
        # Filter out _depth.png, _disp.png, _acc.png if any
        rendered_images = [img for img in rendered_images if not img.endswith(('_depth.png', '_disp.png', '_acc.png'))]
        
        if not rendered_images:
            print(f"\n[!] Model '{expname}' has empty render directory: {latest_render_dir}")
            print(f"    Missing images. Please run the following command to generate them:")
            print(f"    python run_nerf.py --config configs/{expname}.txt --render_only --render_test")
            continue
            
        print(f"\nEvaluating: {expname} (Steps: {step_count}) from {latest_render_dir}")
        
        if len(rendered_images) < len(gt_paths):
            print(f"Warning: Only {len(rendered_images)} rendered images found, expected {len(gt_paths)}.")
            # We process whatever we have matches for
            num_eval = min(len(rendered_images), len(gt_paths))
        else:
            num_eval = len(gt_paths)
            
        model_psnrs = []
        model_ssims = []
        model_lpipss = []
        
        for i in range(num_eval):
            gt_path = gt_paths[i]
            pred_path = rendered_images[i]
            
            # Read GT and composite over white background if alpha exists
            gt_img = imageio.imread(gt_path)
            if gt_img.shape[-1] == 4:
                alpha = (gt_img[..., 3:] / 255.0).astype(np.float32)
                rgb = (gt_img[..., :3] / 255.0).astype(np.float32)
                gt_img = rgb * alpha + (1.0 - alpha)
            else:
                gt_img = (gt_img / 255.0).astype(np.float32)
            
            # Read Pred
            pred_img = imageio.imread(pred_path)
            if pred_img.shape[-1] == 4:
                pred_img = pred_img[..., :3]
            pred_img = (pred_img / 255.0).astype(np.float32)
            
            # Ensure shape match
            if gt_img.shape != pred_img.shape:
                # Resize pred to gt
                pred_img = np.array(imageio.imread(pred_path))
                import cv2
                H, W = gt_img.shape[:2]
                pred_img = cv2.resize(pred_img, (W, H), interpolation=cv2.INTER_AREA)
                if pred_img.shape[-1] == 4:
                    pred_img = pred_img[..., :3]
                pred_img = (pred_img / 255.0).astype(np.float32)
                
            # PSNR
            psnr = calculate_psnr(gt_img, pred_img)
            model_psnrs.append(psnr)
            
            # SSIM
            ssim_val = ssim(gt_img, pred_img, data_range=1.0, channel_axis=-1)
            model_ssims.append(ssim_val)
            
            # LPIPS
            # LPIPS expects tensor of shape [N, C, H, W] in range [-1, 1]
            gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            
            with torch.no_grad():
                lpips_val = loss_fn_vgg(gt_tensor, pred_tensor).item()
            model_lpipss.append(lpips_val)
            
        avg_psnr = np.mean(model_psnrs)
        avg_ssim = np.mean(model_ssims)
        avg_lpips = np.mean(model_lpipss)
        
        print(f"PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
        
        results.append({
            "name": expname,
            "steps": step_count,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "lpips": avg_lpips
        })
        
    # Output Markdown table
    print("\n\n### Evaluation Results\n")
    print("| Model Name | PSNR | SSIM | LPIPS | Training Steps |")
    print("|---|---|---|---|---|")
    for res in results:
        print(f"| {res['name']} | {res['psnr']:.4f} | {res['ssim']:.4f} | {res['lpips']:.4f} | {res['steps']} |")

if __name__ == '__main__':
    main()
