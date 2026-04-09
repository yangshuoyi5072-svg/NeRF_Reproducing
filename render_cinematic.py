import os
import sys
import time
import torch
import numpy as np
import imageio
import cv2
from tqdm import tqdm

# Import from run_nerf.py
from run_nerf import config_parser, create_nerf, render, to8b, device
from core.helpers import get_rays, ndc_rays

def trans_t(t): 
    return torch.Tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]]).float()

def rot_phi(phi): 
    return torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

def rot_theta(th): 
    return torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def generate_cinematic_path(radius_scale=1.0):
    """15s Cinematic Path (450 frames @ 30fps)"""
    poses = []
    
    # Segment 1: Slow Orbit (5s, 150 frames)
    for theta in np.linspace(-180, 0, 150):
        poses.append(pose_spherical(theta, -43.0, 4.0 * radius_scale))
        
    # Segment 2: Spiral zoom-in (3s, 90 frames)
    # Rotating 180 degrees while zooming in from 4.0 to 2.2
    for i in range(90):
        alpha = i / 90.0
        theta = 0 + alpha * 360.0
        phi = -43.0 + alpha * 20.0  # Slightly flatter angle for closer look
        radius = (4.0 - alpha * 1.8) * radius_scale
        poses.append(pose_spherical(theta, phi, radius))
        
    # Segment 3: Detail scan (5s, 150 frames)
    # Slow panning at close range
    for i in range(150):
        alpha = i / 150.0
        theta = 360.0 + alpha * 45.0
        phi = -10.0 - alpha * 10.0
        radius = 2.2 * radius_scale
        poses.append(pose_spherical(theta, phi, radius))
        
    # Segment 4: Snap Pull-back (2s, 60 frames)
    for i in range(60):
        alpha = i / 60.0
        theta = 405.0 + alpha * 30.0
        phi = -20.0 - alpha * 10.0
        radius = (2.2 + alpha * 2.3) * radius_scale
        poses.append(pose_spherical(theta, phi, radius))
        
    return torch.stack(poses, 0)

def generate_acc_path(radius_scale=1.0):
    """5s ACC Path (150 frames @ 30fps)"""
    poses = []
    # Static elevation quick 360 orbit
    for theta in np.linspace(-180, 180, 150):
        poses.append(pose_spherical(theta, -30.0, 4.0 * radius_scale))
    return torch.stack(poses, 0)

def render_custom_path(args, render_poses, hwf, K, savedir, N_importance_override=None, ckpt_path=None, chunk=1024*32, render_indices=None):
    H, W, focal = hwf
    
    # [FAILS-SAFE] Load checkpoint first to detect architecture
    if ckpt_path is not None:
        ckpt_dir = os.path.dirname(ckpt_path)
        args_path = os.path.join(ckpt_dir, "args.txt")
        
        print(f"Pre-loading checkpoint to detect architecture: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        
        # [NEW] Synchronize from args.txt if it exists
        if os.path.exists(args_path):
            print(f"Syncing parameters from: {args_path}")
            with open(args_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, val = [x.strip() for x in line.split('=')]
                        if key == 'white_bkgd':
                            args.white_bkgd = (val == 'True')
                            print(f"  Setting white_bkgd = {args.white_bkgd}")
                        elif key == 'use_viewdirs':
                            args.use_viewdirs = (val == 'True')
                            print(f"  Setting use_viewdirs = {args.use_viewdirs}")
                        elif key == 'zoom':
                            args.zoom = float(val)
                            print(f"  Setting zoom = {args.zoom}")
        
        # Detect from state dict (fallback/double check)
        if 'network_fn_state_dict' in ckpt:
            keys = ckpt['network_fn_state_dict'].keys()
            if 'alpha_linear.weight' in keys:
                args.use_viewdirs = True
                
            if 'network_fine_state_dict' in ckpt:
                if args.N_importance <= 0:
                    args.N_importance = 128
            else:
                args.N_importance = 0

    # Reload model with updated architecture flags
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    
    # Manually load weights
    if ckpt_path is not None:
        render_kwargs_test['network_fn'].load_state_dict(ckpt['network_fn_state_dict'])
        if render_kwargs_test['network_fine'] is not None and 'network_fine_state_dict' in ckpt:
            render_kwargs_test['network_fine'].load_state_dict(ckpt['network_fine_state_dict'])

    bds_dict = {'near': 0.1, 'far': 10.0} # Wider bounds for flexibility
    render_kwargs_test.update(bds_dict)
    
    # [IMPORTANT] Override N_importance if requested
    if N_importance_override is not None:
        render_kwargs_test['N_importance'] = N_importance_override
        print(f"Overriding N_importance to {N_importance_override} for rendering.")
    
    rgbs = []
    accs = []
    
    os.makedirs(savedir, exist_ok=True)
    
    # [Apple Silicon] Ensure poses are on the correct device
    render_poses = render_poses.to(device)
    
    print(f"Rendering {len(render_poses)} frames to {savedir}...")
    with torch.no_grad():
        for idx, c2w in enumerate(tqdm(render_poses)):
            # Use the actual frame index if provided (for worker logic)
            i = render_indices[idx] if render_indices is not None else idx
            
            filename = f"{i:03d}.png"
            acc_filename = f"{i:03d}_acc.png"
            
            # [NEW] Resume Logic
            # Check flags to see if we need to check for acc existence
            need_acc = not getattr(args, 'no_acc', False)
            
            if getattr(args, 'resume', False) and os.path.exists(os.path.join(savedir, filename)):
                if not need_acc or os.path.exists(os.path.join(savedir, acc_filename)):
                    continue
            
            # [NEW] Skip ACC computation if requested to save time
            rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs_test)
            
            rgb_np = rgb.cpu().numpy()
            rgbs.append(rgb_np)
            imageio.imwrite(os.path.join(savedir, filename), to8b(rgb_np))

            if need_acc and acc is not None:
                acc_np = acc.cpu().numpy()
                accs.append(acc_np)
                imageio.imwrite(os.path.join(savedir, acc_filename), to8b(acc_np))
            
            if i % 10 == 0:
                sys.stdout.write(f"  Frame {i}/{len(render_poses)} rendered\n")
                sys.stdout.flush()
            
    return np.array(rgbs), np.array(accs)

def main():
    # Use the official config_parser from run_nerf
    parser = config_parser()
    
    # Add cinematic-specific flags that aren't in run_nerf
    parser.add_argument("--test", action='store_true', help='render only 10 frames for testing')
    parser.add_argument("--fast", action='store_true', help='render with fewer importance samples')
    parser.add_argument("--single-frame", action='store_true', help='render only the first frame for verification')
    parser.add_argument("--gt-pose", action='store_true', help='calibration mode: use dataset pose 0')
    parser.add_argument("--ckpt", type=str, default=None, help='specific checkpoint to load')
    parser.add_argument("--zoom", type=float, default=None, help='override zoom factor')
    parser.add_argument("--resume", action='store_true', help='skip existing frames')
    parser.add_argument("--no-acc", action='store_true', help='skip depth (ACC) rendering for speed')
    parser.add_argument("--worker-id", type=int, default=0, help='ID of this worker')
    parser.add_argument("--total-workers", type=int, default=1, help='Total number of workers')
    
    args = parser.parse_args()
    
    # [IMPORTANT] SYNC PARAMETERS FROM THE EXPERIMENT'S args.txt
    # This prevents 'fern' (default) from being used when we target a Lego experiment folder.
    ckpt_dir = os.path.dirname(args.ckpt) if args.ckpt else None
    if ckpt_dir:
        args_path = os.path.join(ckpt_dir, "args.txt")
        if os.path.exists(args_path):
            print(f"Syncing parameters from: {args_path}")
            with open(args_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, val = line.split('=')[0].strip(), line.split('=')[1].strip()
                        if key == 'datadir':
                            args.datadir = val
                            print(f"  Setting datadir = {args.datadir}")
                        elif key == 'white_bkgd':
                            args.white_bkgd = (val == 'True')
                            print(f"  Setting white_bkgd = {args.white_bkgd}")
                        elif key == 'use_viewdirs':
                            args.use_viewdirs = (val == 'True')
                            print(f"  Setting use_viewdirs = {args.use_viewdirs}")
                        elif key == 'zoom':
                            # Only set if not already overridden by command line
                            if args.zoom is None:
                                args.zoom = float(val)
                                print(f"  Setting zoom = {args.zoom} (from args.txt)")
                            else:
                                print(f"  Ignoring zoom from args.txt, using override: {args.zoom}")
    
    # [FIX] Focal length correction (Lego = 0.6911 rad)
    H, W = 400, 400
    camera_angle_x = 0.6911112070083618 # Default for Lego
    
    # Detect from config/args if possible
    zoom_factor = args.zoom if args.zoom is not None else 1.0
    
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    focal = focal * zoom_factor
    hwf = [H, W, focal]
    K = np.array([[focal, 0, 0.5*W], [0, focal, 0.5*H], [0, 0, 1]])

    # Path generation (standard radius 4.0)
    # We use radius_scale=1.0 because zoom is handled by focal scaling
    rgb_poses = generate_cinematic_path(radius_scale=1.0)
    acc_poses = generate_acc_path(radius_scale=1.0)
    
    # [NEW] Calibration Mode: Use Ground Truth Pose 0 from dataset
    if getattr(args, 'gt_pose', False):
        print("CALIBRATION MODE: Overriding path with Test Dataset Pose 0")
        from load_blender import load_blender_data
        _, poses, _, _, _, i_split = load_blender_data(args.datadir, half_res=True)
        gt_pose = torch.Tensor(poses[i_split[2][0]]) # First test pose
        rgb_poses = gt_pose[None, ...]
        acc_poses = gt_pose[None, ...]
    
    # Create output dir with a unique name for this run
    output_dir = os.path.join(args.basedir, args.expname, "cinematic_submission")
    os.makedirs(output_dir, exist_ok=True)
    
    # [FIX] Default N_importance to 128 (Quality Mode)
    N_importance = args.N_importance
    if args.fast:
        N_importance = 64
        print(f"FAST MODE: Reducing N_importance to {N_importance}")
    elif N_importance <= 0:
        N_importance = 128
        print(f"QUALITY MODE: Defaulting N_importance to {N_importance} (Fine Model)")

    # Test mode: only 5 frames for quick sanity check
    if args.test:
        rgb_poses = rgb_poses[:5]
        acc_poses = acc_poses[:5]
        print("TEST MODE: Rendering only 5 frames each.")

    # Single-frame mode for proof of concept
    if args.single_frame:
        rgb_poses = rgb_poses[:1]
        acc_poses = acc_poses[:1]
        print("SINGLE FRAME MODE: Rendering ONLY Frame 000 for verification.")
    
    # [NEW] Worker Partitioning
    if args.total_workers > 1:
        print(f"PARALLEL MODE: Worker {args.worker_id}/{args.total_workers}")
        # We partition based on frame index (Modulus)
        # This allows resume logic to work correctly for individual frames
        filtered_indices = [i for i in range(len(rgb_poses)) if i % args.total_workers == args.worker_id]
        rgb_poses = rgb_poses[filtered_indices]
        # We need to pass the indices to render_custom_path so it knows the filenames
        render_indices = filtered_indices
    else:
        render_indices = range(len(rgb_poses))

    # 1. Render RGB Cinematic (450 frames, 15s)
    # This pass now saves BOTH RGB and ACC for every frame to match baseline efficiency.
    rgbs, accs = render_custom_path(args, rgb_poses, hwf, K, os.path.join(output_dir, "rgb_frames"), 
                                  N_importance_override=N_importance, ckpt_path=args.ckpt, chunk=32768,
                                  render_indices=render_indices)
    
    # Save the 15s RGB video
    imageio.mimwrite(os.path.join(output_dir, f"{args.expname}_cinematic_rgb.mp4"), to8b(rgbs), fps=30, quality=9)
    
    # [OPTIMIZATION] Skip ACC video if requested
    if not getattr(args, 'no_acc', False):
        # The 5s ACC video can follow the same path or a different one.
        # If the user specifically wants the orbit-only ACC video, we take the first 150 frames.
        # Otherwise, we provide the full 15s ACC video as well.
        accs_5s = accs[:150]
        imageio.mimwrite(os.path.join(output_dir, f"{args.expname}_cinematic_acc_5s.mp4"), to8b(accs_5s), fps=30, quality=9)
        # Also save full ACC
        imageio.mimwrite(os.path.join(output_dir, f"{args.expname}_cinematic_acc_15s.mp4"), to8b(accs), fps=30, quality=9)
    else:
        print("Skipping ACC video synthesis per --no-acc flag.")
    
    print(f"Final videos saved in {output_dir}")

if __name__ == "__main__":
    main()
