#!/usr/bin/env python3
"""
Use CDM (Camera Depth Model) to batch process depth images in real robot data.
Save the processed depth maps to the depth_after_cdm folder.

Usage:
    python process_depth_with_cdm.py --data-dir /path/to/data --model-path /path/to/cdm_model.pth
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple

# Add CDM module path
CDM_PATH = Path(__file__).parent.parent / "manip-as-in-sim-suite" / "cdm"
sys.path.insert(0, str(CDM_PATH))

from rgbddepth.dpt import RGBDDepth


def load_cdm_model(model_path: str, encoder: str = 'vitl', device: str = 'cuda') -> RGBDDepth:
    """Load CDM model"""
    print(f"Loading CDM model from: {model_path}")
    print(f"Encoder: {encoder}, Device: {device}")
    
    # Set features based on encoder type
    encoder_features = {
        'vits': 64,
        'vitb': 128,
        'vitl': 256,
        'vitg': 384
    }
    features = encoder_features.get(encoder, 256)
    
    model = RGBDDepth(encoder=encoder, features=features)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        # Handle checkpoints that wrap state dict in 'model' key
        states = {k[7:]: v for k, v in checkpoint['model'].items()}
        print(f"✓ Loaded checkpoint with 'model' key")
    elif 'state_dict' in checkpoint:
        # PyTorch Lightning checkpoint format
        states = checkpoint['state_dict']
        # Remove 'pipeline.' prefix (9 characters)
        states = {k[9:]: v for k, v in states.items() if k.startswith('pipeline.')}
        print(f"✓ Loaded PyTorch Lightning checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # Standard PyTorch state_dict
        states = checkpoint
        print(f"✓ Loaded standard PyTorch checkpoint")
    
    model.load_state_dict(states, strict=False)
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model


def process_depth_image(
    model: RGBDDepth,
    rgb_path: str,
    depth_path: str,
    input_size: int = 518,
    depth_scale: float = 1000.0,
    max_depth: float = 6.0,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Process a single depth image using CDM
    
    Args:
        model: CDM model
        rgb_path: Path to RGB image
        depth_path: Path to original depth image
        input_size: Input size
        depth_scale: Depth scale factor (usually 1000.0 means unit is mm)
        max_depth: Maximum valid depth (meters)
        device: Computing device
    
    Returns:
        Processed depth map (unit: mm)
    """
    # Load RGB image (keep BGR format, infer_image will handle it internally)
    rgb_bgr = cv2.imread(rgb_path)
    if rgb_bgr is None:
        raise ValueError(f"Could not load RGB image: {rgb_path}")
    
    # Load depth image
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Could not load depth image: {depth_path}")
    
    # Convert to meters and clip out of range values
    depth = depth.astype(np.float32) / depth_scale
    depth[depth > max_depth] = 0.0
    
    # Create inverse depth
    simi_depth = np.zeros_like(depth)
    valid_mask = depth > 0
    simi_depth[valid_mask] = 1.0 / depth[valid_mask]
    
    # CDM inference (input BGR RGB and inverse depth)
    with torch.no_grad():
        pred_inverse_depth = model.infer_image(rgb_bgr, simi_depth, input_size=input_size)
    
    # CDM outputs inverse depth, need to convert back to real depth
    # Avoid division by zero and invalid values
    valid_pred_mask = pred_inverse_depth > 1e-6
    pred_depth = np.zeros_like(pred_inverse_depth)
    pred_depth[valid_pred_mask] = 1.0 / pred_inverse_depth[valid_pred_mask]
    
    # Clip to reasonable range
    pred_depth = np.clip(pred_depth, 0.1, max_depth)
    
    # Convert back to mm and save as uint16
    pred_depth_mm = (pred_depth * depth_scale).astype(np.uint16)
    
    return pred_depth_mm


def find_all_episodes(data_dir: str) -> List[Tuple[str, int, str]]:
    """
    Find all episode directories
    Supports two structures:
    1. Single episode: data_dir/images/observation.../episode_000000/
    2. Multiple episodes: data_dir/01/images/.../episode_000000/, data_dir/02/images/.../
    
    Returns:
        List of (episode_root_dir, episode_number, episode_name)
    """
    data_path = Path(data_dir)
    episodes = []
    
    # Check if it's a single episode directory (contains images/ and meta/)
    if (data_path / "images").exists() and (data_path / "meta").exists():
        # Single episode mode
        camera_dirs = [
            "images/observation.images.cam_arm",
            "images/observation.images.cam_left", 
            "images/observation.images.cam_right"
        ]
        
        for cam_dir in camera_dirs:
            cam_path = data_path / cam_dir
            if cam_path.exists():
                # Find episode_XXXXXX directory
                for episode_dir in sorted(cam_path.iterdir()):
                    if episode_dir.is_dir() and episode_dir.name.startswith("episode_"):
                        try:
                            episode_num = int(episode_dir.name.split("_")[1])
                            episodes.append((str(data_path), episode_num, data_path.name))
                        except (IndexError, ValueError):
                            continue
                break  # Finding one camera is enough
    else:
        # Multi episode mode: scan subdirectories
        for subdir in sorted(data_path.iterdir()):
            if not subdir.is_dir():
                continue
            
            # Check if subdirectory contains images/ and meta/
            if (subdir / "images").exists() and (subdir / "meta").exists():
                # This is an episode directory, check episode_XXXXXX inside
                camera_dirs = [
                    "images/observation.images.cam_arm",
                    "images/observation.images.cam_left", 
                    "images/observation.images.cam_right"
                ]
                
                for cam_dir in camera_dirs:
                    cam_path = subdir / cam_dir
                    if cam_path.exists():
                        for episode_dir in sorted(cam_path.iterdir()):
                            if episode_dir.is_dir() and episode_dir.name.startswith("episode_"):
                                try:
                                    episode_num = int(episode_dir.name.split("_")[1])
                                    episodes.append((str(subdir), episode_num, subdir.name))
                                except (IndexError, ValueError):
                                    continue
                        break  # Finding one camera is enough
    
    # Deduplicate and sort
    episodes = sorted(list(set(episodes)), key=lambda x: (x[0], x[1]))
    
    return episodes


def process_episode(
    model: RGBDDepth,
    data_dir: str,
    episode_num: int,
    camera_name: str,
    input_size: int = 518,
    depth_scale: float = 1000.0,
    max_depth: float = 6.0,
    device: str = 'cuda'
) -> int:
    """
    Process a specific camera's depth images for a single episode
    
    Returns:
        Number of processed frames
    """
    data_path = Path(data_dir)
    episode_str = f"episode_{episode_num:06d}"
    
    # Build paths
    rgb_dir = data_path / "images" / f"observation.images.{camera_name}" / episode_str
    depth_input_dir = data_path / "images" / f"observation.depths.{camera_name}" / episode_str
    depth_output_dir = data_path / "images" / f"observation.depths_after_cdm.{camera_name}" / episode_str
    
    # Check if input directory exists
    if not rgb_dir.exists() or not depth_input_dir.exists():
        return 0
    
    # Create output directory
    depth_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all depth image files
    depth_files = sorted(depth_input_dir.glob("*.png"))
    
    if not depth_files:
        return 0
    
    # Process each frame
    processed_count = 0
    for depth_file in tqdm(depth_files, desc=f"  {camera_name}", leave=False):
        frame_name = depth_file.name
        rgb_file = rgb_dir / frame_name
        output_file = depth_output_dir / frame_name
        
        # Check if RGB file exists
        if not rgb_file.exists():
            print(f"  ⚠ Warning: RGB file does not exist: {rgb_file}")
            continue
        
        try:
            # Process depth image
            processed_depth = process_depth_image(
                model=model,
                rgb_path=str(rgb_file),
                depth_path=str(depth_file),
                input_size=input_size,
                depth_scale=depth_scale,
                max_depth=max_depth,
                device=device
            )
            
            # Save processed depth image
            cv2.imwrite(str(output_file), processed_depth)
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Processing failed {frame_name}: {e}")
            continue
    
    return processed_count


def main():
    parser = argparse.ArgumentParser(description="Batch process real robot depth data using CDM")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data root directory (contains images/ etc.)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="CDM model weights path (.pth file)")
    parser.add_argument("--encoder", type=str, default="vitl",
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help="Encoder type (default: vitl)")
    parser.add_argument("--input-size", type=int, default=518,
                        help="Input image size (default: 518)")
    parser.add_argument("--depth-scale", type=float, default=1000.0,
                        help="Depth scale factor (default: 1000.0 for mm)")
    parser.add_argument("--max-depth", type=float, default=6.0,
                        help="Maximum valid depth in meters (default: 6.0)")
    parser.add_argument("--cameras", type=str, nargs='+',
                        default=['cam_arm', 'cam_left', 'cam_right'],
                        help="List of cameras to process (default: cam_arm cam_left cam_right)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computing device (default: cuda)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CDM Depth Image Batch Processing Tool")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Cameras: {args.cameras}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Check paths
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"✗ Error: Data directory does not exist: {args.data_dir}")
        return 1
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"✗ Error: Model file does not exist: {args.model_path}")
        return 1
    
    # Load model
    try:
        model = load_cdm_model(
            model_path=str(model_path),
            encoder=args.encoder,
            device=args.device
        )
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return 1
    
    # Find all episodes
    print("\nScanning episodes...")
    episodes = find_all_episodes(args.data_dir)
    
    if not episodes:
        print("✗ No episodes found")
        return 1
    
    print(f"✓ Found {len(episodes)} episodes")
    
    # Process each episode
    total_frames = 0
    for episode_root, episode_num, episode_name in episodes:
        print(f"\nProcessing {episode_name} (Episode {episode_num:06d}):")
        
        episode_frames = 0
        for camera in args.cameras:
            frames = process_episode(
                model=model,
                data_dir=episode_root,
                episode_num=episode_num,
                camera_name=camera,
                input_size=args.input_size,
                depth_scale=args.depth_scale,
                max_depth=args.max_depth,
                device=args.device
            )
            episode_frames += frames
            if frames > 0:
                print(f"  ✓ {camera}: {frames} frames processed")
        
        total_frames += episode_frames
        print(f"  {episode_name} Total: {episode_frames} frames")
    
    print("\n" + "=" * 60)
    print(f"✓ Processing complete!")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output directory: {args.data_dir}/images/observation.depths_after_cdm.*/")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
