import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
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


def load_blender_data(basedir, half_res=False, testskip=1, num_views=-1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_depths = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        depths_list = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            img = imageio.imread(fname)
            imgs.append(img)
            
            # Extract depth: check for SpatialVID explicit .npy mapping, fallback to _depth.png, then alpha occupancy
            if 'depth_path' in frame and os.path.exists(os.path.join(basedir, frame['depth_path'])):
                depth_npy_path = os.path.join(basedir, frame['depth_path'])
                depth_arr = np.load(depth_npy_path).astype(np.float32)
                depths_list.append(depth_arr)
            else:
                depth_fname = os.path.join(basedir, frame['file_path'] + '_depth.png')
                if os.path.exists(depth_fname):
                    depth_img = imageio.imread(depth_fname)
                    if len(depth_img.shape) >= 3:
                        depth_img = depth_img[..., 0]
                    depths_list.append((depth_img / 255.).astype(np.float32))
                else:
                    # use alpha channel as proxy depth (Occupancy Mask)
                    if img.shape[-1] == 4:
                        depths_list.append((img[..., 3] / 255.).astype(np.float32))
                    else:
                        depths_list.append(np.ones_like(img[..., 0]).astype(np.float32))
            
            poses.append(np.array(frame['transform_matrix']))
        
        # Few-shot selection logic for training set
        if s == 'train' and num_views > 0:
            idx = np.round(np.linspace(0, len(imgs) - 1, num_views)).astype(int)
            print(f"[Few-Shot] Selecting {num_views} train images at indices: {idx}")
            imgs = [imgs[i] for i in idx]
            depths_list = [depths_list[i] for i in idx]
            poses = [poses[i] for i in idx]

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        depths_arr = np.array(depths_list).astype(np.float32)
        
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_depths.append(depths_arr)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    depths = np.concatenate(all_depths, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        channels = imgs.shape[-1]
        imgs_half_res = np.zeros((imgs.shape[0], H, W, channels))
        depths_half_res = np.zeros((depths.shape[0], H, W))
        for i, (img, depth) in enumerate(zip(imgs, depths)):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            depths_half_res[i] = cv2.resize(depth, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        depths = depths_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, depths, render_poses, [H, W, focal], i_split