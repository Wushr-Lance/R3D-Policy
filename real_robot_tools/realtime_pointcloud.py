#!/usr/bin/env python3
"""
Optimized real-time point cloud generation for robot deployment.

Performance optimizations:
1. Pre-compute pixel grids
2. Cache transformation matrices
3. Use GPU for coordinate transformations
4. Vectorized operations
5. Fast random sampling instead of FPS for real-time
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
import time
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R


class RealtimePointCloudGenerator:
    """
    Real-time point cloud generator optimized for robot deployment.
    
    Usage:
        generator = RealtimePointCloudGenerator(calib, intrinsics, device='cuda:0')
        
        # In control loop (30Hz):
        rgb_dict = {'cam_left': rgb_left, 'cam_right': rgb_right, 'cam_arm': rgb_arm}
        depth_dict = {'cam_left': depth_left, 'cam_right': depth_right, 'cam_arm': depth_arm}
        ee_pos = [x, y, z, roll, pitch, yaw]  # Current end-effector pose
        
        pointcloud = generator.generate(rgb_dict, depth_dict, ee_pos, num_points=1024)
    """
    
    def __init__(
        self,
        calib: Dict,
        intrinsics: Dict,
        device: str = 'cuda:0',
        crop_x_range: Tuple[float, float] = (-0.5, 0.82),
        crop_y_range: Tuple[float, float] = (-0.7, 0.8),
        crop_z_range: Tuple[float, float] = (-np.inf, np.inf),
        depth_scale: float = 1000.0,
        depth_min: float = 0.1,
        depth_max: float = 5.0
    ):
        """
        Initialize the generator with calibration data.
        
        Args:
            calib: Camera calibration dict with 'T_c2b' and 'T_c2h'
            intrinsics: Camera intrinsics dict
            device: PyTorch device ('cuda:0' or 'cpu')
            crop_x_range: X cropping range (meters), default (-0.5, 0.82) aligned with convert script
            crop_y_range: Y cropping range (meters), default (-0.7, 0.8) aligned with convert script
            crop_z_range: Z cropping range (meters)
            depth_scale: Depth unit conversion (1000.0 for mm to m)
            depth_min: Minimum valid depth (meters)
            depth_max: Maximum valid depth (meters)
        """
        self.device = torch.device(device)
        self.depth_scale = depth_scale
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.crop_ranges = (crop_x_range, crop_y_range, crop_z_range)
        
        # Camera names
        self.camera_names = ['cam_left', 'cam_right', 'cam_arm']
        
        # Pre-compute and cache data
        self._precompute_pixel_grids(intrinsics)
        self._precompute_transforms(calib)
        
        print(f"✅ RealtimePointCloudGenerator initialized on {device}")
        print(f"   Crop ranges: X{crop_x_range}, Y{crop_y_range}, Z{crop_z_range}")
    
    def _precompute_pixel_grids(self, intrinsics: Dict):
        """Pre-compute pixel coordinate grids for each camera."""
        self.pixel_grids = {}
        self.intrinsics_torch = {}
        
        for cam_name in self.camera_names:
            if cam_name not in intrinsics:
                continue
            
            K = np.array(intrinsics[cam_name]['depth'], dtype=np.float32)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # Assume 640x480 depth image
            h, w = 480, 640
            
            # Pre-compute pixel grids
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            
            # Pre-compute normalized coordinates (ensure float32)
            u_norm = ((u - cx) / fx).astype(np.float32)  # (H, W)
            v_norm = ((v - cy) / fy).astype(np.float32)  # (H, W)
            
            # Store as torch tensors on GPU
            self.pixel_grids[cam_name] = {
                'u_norm': torch.from_numpy(u_norm).to(self.device),
                'v_norm': torch.from_numpy(v_norm).to(self.device),
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }
            
            print(f"   Pre-computed pixel grid for {cam_name}: {h}x{w}")
    
    def _precompute_transforms(self, calib):
        """Pre-compute static camera-to-base transforms."""
        # Handle both calib formats: direct dict or nested T_c2b dict
        if 'T_c2b' in calib:
            calib_transforms = calib['T_c2b']
        else:
            calib_transforms = calib
        
        print(f"   Pre-computing static transforms for {len(calib_transforms)} cameras")
        self.transforms = {}
        
        # Left and right cameras (static transforms)
        for cam_name in ['cam_left', 'cam_right']:
            if cam_name in calib_transforms:
                # The calib stores base-to-camera, we need camera-to-base
                T_b2c = np.array(calib_transforms[cam_name], dtype=np.float32)
                T_c2b = np.linalg.inv(T_b2c).astype(np.float32)  # Invert to get camera-to-base
                
                # Store as torch tensor (ensure float32)
                self.transforms[cam_name] = torch.from_numpy(T_c2b).float().to(self.device)
        
        # Head camera (dynamic, need T_c2h)
        if 'T_c2h' in calib:
            T_c2h = np.array(calib['T_c2h'], dtype=np.float32)
            self.T_c2h = torch.from_numpy(T_c2h).to(self.device)
        else:
            self.T_c2h = None
        
        print(f"   Pre-computed static transforms for {len(self.transforms)} cameras")
    
    def _build_hand2base_transform(self, ee_pos: np.ndarray) -> torch.Tensor:
        """
        Build hand2base transform from end-effector pose.
        
        Args:
            ee_pos: [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
            
        Returns:
            T_h2b: (4, 4) torch tensor on GPU
        """
        # Translation (mm to m)
        translation = ee_pos[:3] / 1000.0
        
        # Rotation (degrees to radians)
        euler_rad = np.deg2rad(ee_pos[3:6])
        rotation_matrix = R.from_euler('xyz', euler_rad).as_matrix()
        
        # Build 4x4 matrix
        T_h2b = np.eye(4, dtype=np.float32)
        T_h2b[:3, :3] = rotation_matrix
        T_h2b[:3, 3] = translation
        
        return torch.from_numpy(T_h2b).to(self.device)
    
    def _rgbd_to_pointcloud_gpu(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        cam_name: str,
        transform: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate point cloud from RGB-D on GPU.
        
        Args:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) uint16
            cam_name: Camera name
            transform: (4, 4) camera2base transform
            
        Returns:
            Point cloud (N, 6) torch tensor with xyz + rgb
        """
        # Convert to torch tensors and move to GPU
        depth_torch = torch.from_numpy(depth).to(self.device).float()
        rgb_torch = torch.from_numpy(rgb).to(self.device).float() / 255.0
        
        # Convert depth to meters
        depth_m = depth_torch / self.depth_scale
        
        # Create valid mask
        valid_mask = (depth_m > self.depth_min) & (depth_m < self.depth_max)
        
        # Get pre-computed pixel grids
        grid_data = self.pixel_grids[cam_name]
        u_norm = grid_data['u_norm']
        v_norm = grid_data['v_norm']
        
        # Back-project to camera coordinates (vectorized)
        z = depth_m  # (H, W)
        x = u_norm * z  # (H, W)
        y = v_norm * z  # (H, W)
        
        # Stack into point cloud (H, W, 3)
        points_cam = torch.stack([x, y, z], dim=-1)
        
        # Transform to base coordinates
        # Reshape to (H*W, 3)
        h, w = points_cam.shape[:2]
        points_flat = points_cam.reshape(-1, 3)
        
        # Add homogeneous coordinate
        ones = torch.ones(points_flat.shape[0], 1, device=self.device, dtype=torch.float32)
        points_homo = torch.cat([points_flat, ones], dim=1)  # (H*W, 4)
        
        # Apply transform: (4, 4) @ (H*W, 4).T -> (4, H*W) -> (H*W, 4)
        points_base_homo = (transform @ points_homo.T).T
        points_base = points_base_homo[:, :3]  # (H*W, 3)
        
        # Reshape back to (H, W, 3)
        points_base = points_base.reshape(h, w, 3)
        
        # Apply valid mask and flatten
        valid_mask_flat = valid_mask.reshape(-1)
        points_valid = points_base.reshape(-1, 3)[valid_mask_flat]
        rgb_valid = rgb_torch.reshape(-1, 3)[valid_mask_flat]
        
        # Combine xyz + rgb
        pointcloud = torch.cat([points_valid, rgb_valid], dim=1)  # (N, 6)
        
        return pointcloud
    
    def _crop_pointcloud_gpu(self, points: torch.Tensor) -> torch.Tensor:
        """Crop point cloud on GPU."""
        x_range, y_range, z_range = self.crop_ranges
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        mask = (
            (x >= x_range[0]) & (x <= x_range[1]) &
            (y >= y_range[0]) & (y <= y_range[1]) &
            (z >= z_range[0]) & (z <= z_range[1])
        )
        
        return points[mask]
    
    def _fast_random_sampling_gpu(self, points: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Fast random sampling on GPU (for real-time use).
        Much faster than FPS, suitable for 30Hz control loop.
        """
        n_points = points.shape[0]
        
        if n_points <= num_points:
            # Pad with zeros
            padding = torch.zeros(num_points - n_points, 6, device=self.device)
            return torch.cat([points, padding], dim=0)
        
        # Random sampling (very fast)
        indices = torch.randperm(n_points, device=self.device)[:num_points]
        return points[indices]
    
    def generate(
        self,
        rgb_dict: Dict[str, np.ndarray],
        depth_dict: Dict[str, np.ndarray],
        ee_pos: Optional[np.ndarray] = None,
        num_points: int = 1024
    ) -> np.ndarray:
        """
        Generate merged point cloud from multiple cameras (REAL-TIME).
        
        Args:
            rgb_dict: Dict of RGB images {cam_name: rgb_array (H, W, 3) uint8}
            depth_dict: Dict of depth images {cam_name: depth_array (H, W) uint16}
            ee_pos: End-effector pose [x, y, z, roll, pitch, yaw] for head camera (optional)
            num_points: Number of points to sample (default 1024 for real-time)
            
        Returns:
            Point cloud (num_points, 6) numpy array with xyz + rgb
        """
        pointclouds = []
        
        # Process each camera
        for cam_name in self.camera_names:
            if cam_name not in rgb_dict or cam_name not in depth_dict:
                continue
            
            rgb = rgb_dict[cam_name]
            depth = depth_dict[cam_name]
            
            # Get transform
            if cam_name == 'cam_arm':
                # Head camera: dynamic transform
                if ee_pos is None or self.T_c2h is None:
                    continue
                T_h2b = self._build_hand2base_transform(ee_pos)
                transform = T_h2b @ self.T_c2h
            else:
                # Static cameras
                if cam_name not in self.transforms:
                    continue
                transform = self.transforms[cam_name]
            
            # Generate point cloud
            pc = self._rgbd_to_pointcloud_gpu(rgb, depth, cam_name, transform)
            pointclouds.append(pc)
        
        # Merge all cameras
        if len(pointclouds) == 0:
            # Return zeros if no valid cameras
            return np.zeros((num_points, 6), dtype=np.float32)
        
        merged_pc = torch.cat(pointclouds, dim=0)
        
        # Crop
        cropped_pc = self._crop_pointcloud_gpu(merged_pc)
        
        # Fast random sampling (for real-time)
        sampled_pc = self._fast_random_sampling_gpu(cropped_pc, num_points)
        
        # Convert to numpy
        return sampled_pc.cpu().numpy().astype(np.float32)
    
    def benchmark(self, rgb_dict: Dict, depth_dict: Dict, ee_pos: np.ndarray, iterations: int = 100):
        """
        Benchmark the generation speed.
        
        Args:
            rgb_dict: Sample RGB images
            depth_dict: Sample depth images
            ee_pos: Sample end-effector pose
            iterations: Number of iterations to average
        """
        import time
        
        # Warmup
        for _ in range(10):
            _ = self.generate(rgb_dict, depth_dict, ee_pos, num_points=1024)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(iterations):
            _ = self.generate(rgb_dict, depth_dict, ee_pos, num_points=1024)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        avg_time = elapsed / iterations * 1000  # ms
        fps = iterations / elapsed
        
        print(f"\n📊 Benchmark Results ({iterations} iterations):")
        print(f"   Average time: {avg_time:.2f} ms/frame")
        print(f"   Throughput: {fps:.1f} FPS")
        print(f"   Suitable for: {'✅ Real-time (>30Hz)' if fps > 30 else '⚠️  May be too slow'}")


# Convenience function for quick testing
def create_realtime_generator(
    calib_path: str,
    inst_path: str,
    device: str = 'cuda:0',
    **kwargs
) -> RealtimePointCloudGenerator:
    """
    Create a real-time point cloud generator from calibration files.
    
    Args:
        calib_path: Path to calib.json
        inst_path: Path to inst.json
        device: Device to use
        **kwargs: Additional arguments for RealtimePointCloudGenerator
        
    Returns:
        Initialized generator
    """
    import json
    
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    with open(inst_path, 'r') as f:
        inst = json.load(f)
    
    return RealtimePointCloudGenerator(calib, inst, device=device, **kwargs)


def load_real_data_sample(data_dir: str, frame_idx: int = 0, use_cdm_depth: bool = False):
    """
    Load real camera data from dataset.
    
    Args:
        data_dir: Path to episode directory (e.g., 'data/test/01/')
        frame_idx: Frame index to load
        use_cdm_depth: Use CDM-processed depth
        
    Returns:
        rgb_dict, depth_dict, ee_pos
    """
    import imageio.v3 as iio
    from pathlib import Path
    import pandas as pd
    
    data_dir = Path(data_dir)
    
    # Load metadata
    parquet_path = data_dir / 'data' / 'chunk-000' / 'episode_000000.parquet'
    df = pd.read_parquet(parquet_path)
    
    if frame_idx >= len(df):
        raise ValueError(f"Frame {frame_idx} out of range (total: {len(df)})")
    
    row = df.iloc[frame_idx]
    
    # Load images
    rgb_dict = {}
    depth_dict = {}
    
    for cam_name in ['cam_left', 'cam_right', 'cam_arm']:
        # RGB
        rgb_path = data_dir / 'images' / f'observation.images.{cam_name}' / 'episode_000000' / f'frame_{frame_idx:06d}.png'
        rgb_dict[cam_name] = iio.imread(rgb_path)
        
        # Depth
        if use_cdm_depth:
            depth_folder = f'observation.depths_after_cdm.{cam_name}'
        else:
            depth_folder = f'observation.depths.{cam_name}'
        depth_path = data_dir / 'images' / depth_folder / 'episode_000000' / f'frame_{frame_idx:06d}.png'
        depth_dict[cam_name] = iio.imread(depth_path)
    
    # End-effector pose
    ee_pos = np.array(row['observation.ee_pos'])
    
    return rgb_dict, depth_dict, ee_pos


def visualize_pointcloud_plotly(pointcloud: np.ndarray, output_path: str = 'realtime_pointcloud.html', title: str = 'Real-time Point Cloud'):
    """
    Generate interactive HTML visualization with Plotly.
    
    Args:
        pointcloud: (N, 6) array with xyz + rgb
        output_path: Output HTML file path
        title: Plot title
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("⚠️  Plotly not installed. Install with: pip install plotly")
        return
    
    xyz = pointcloud[:, :3]
    rgb = pointcloud[:, 3:6]
    
    # Convert RGB to hex colors
    rgb_255 = (rgb * 255).astype(np.uint8)
    colors = [f'rgb({r},{g},{b})' for r, g, b in rgb_255]
    
    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
        ),
        text=[f'XYZ: ({x:.3f}, {y:.3f}, {z:.3f})' for x, y, z in xyz],
        hoverinfo='text'
    )])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1200,
        height=800
    )
    
    # Save to HTML
    fig.write_html(output_path)
    print(f"✅ Saved visualization to: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Point Cloud Generator')
    parser.add_argument('--mode', type=str, choices=['random', 'real', 'benchmark'], default='random',
                        help='Mode: random (synthetic data), real (load from dataset), benchmark (performance test)')
    parser.add_argument('--data-dir', type=str, default='data/test/01/',
                        help='Episode directory for real mode')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to load in real mode')
    parser.add_argument('--use-cdm-depth', action='store_true',
                        help='Use CDM-processed depth in real mode')
    parser.add_argument('--num-points', type=int, default=8192,
                        help='Number of points to sample (default 8192, aligned with convert script)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate Plotly HTML visualization')
    parser.add_argument('--output-html', type=str, default='realtime_pointcloud.html',
                        help='Output HTML file path for visualization')
    parser.add_argument('--calib', type=str, default='calib.json',
                        help='Path to calibration file')
    parser.add_argument('--inst', type=str, default='inst.json',
                        help='Path to intrinsics file')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0 or cpu)')
    
    args = parser.parse_args()
    
    # Example usage and benchmark
    print("=" * 60)
    print("Real-time Point Cloud Generator")
    print("=" * 60)
    
    # Create generator (with aligned default crop ranges)
    generator = create_realtime_generator(
        calib_path=args.calib,
        inst_path=args.inst,
        device=args.device
    )
    
    if args.mode == 'real':
        # Load real data
        print(f"\n📷 Loading real data from: {args.data_dir}")
        print(f"   Frame: {args.frame}")
        print(f"   CDM depth: {args.use_cdm_depth}")
        
        rgb_dict, depth_dict, ee_pos = load_real_data_sample(
            args.data_dir, 
            args.frame, 
            args.use_cdm_depth
        )
        
        print(f"   Loaded RGB shapes: {[rgb.shape for rgb in rgb_dict.values()]}")
        print(f"   Loaded depth shapes: {[depth.shape for depth in depth_dict.values()]}")
        print(f"   End-effector pose: {ee_pos}")
        
    elif args.mode == 'random':
        # Simulate camera inputs
        print("\n📷 Generating synthetic camera inputs...")
        rgb_left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        rgb_right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        rgb_arm = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        depth_left = np.random.randint(300, 2000, (480, 640), dtype=np.uint16)
        depth_right = np.random.randint(300, 2000, (480, 640), dtype=np.uint16)
        depth_arm = np.random.randint(300, 2000, (480, 640), dtype=np.uint16)
        
        rgb_dict = {
            'cam_left': rgb_left,
            'cam_right': rgb_right,
            'cam_arm': rgb_arm
        }
        depth_dict = {
            'cam_left': depth_left,
            'cam_right': depth_right,
            'cam_arm': depth_arm
        }
        ee_pos = np.array([500.0, 200.0, 300.0, 0.0, 0.0, 90.0])  # [x, y, z, roll, pitch, yaw]
    
    else:  # benchmark
        # Use random data for benchmark
        print("\n📷 Generating synthetic data for benchmark...")
        rgb_left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        rgb_right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        rgb_arm = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        depth_left = np.random.randint(300, 2000, (480, 640), dtype=np.uint16)
        depth_right = np.random.randint(300, 2000, (480, 640), dtype=np.uint16)
        depth_arm = np.random.randint(300, 2000, (480, 640), dtype=np.uint16)
        
        rgb_dict = {
            'cam_left': rgb_left,
            'cam_right': rgb_right,
            'cam_arm': rgb_arm
        }
        depth_dict = {
            'cam_left': depth_left,
            'cam_right': depth_right,
            'cam_arm': depth_arm
        }
        ee_pos = np.array([500.0, 200.0, 300.0, 0.0, 0.0, 90.0])
    
    # Generate point cloud
    print(f"\n🔄 Generating point cloud (num_points={args.num_points})...")
    pointcloud = generator.generate(rgb_dict, depth_dict, ee_pos, num_points=args.num_points)
    print(f"   Generated: {pointcloud.shape} point cloud")
    print(f"   XYZ range: X[{pointcloud[:, 0].min():.3f}, {pointcloud[:, 0].max():.3f}], "
          f"Y[{pointcloud[:, 1].min():.3f}, {pointcloud[:, 1].max():.3f}], "
          f"Z[{pointcloud[:, 2].min():.3f}, {pointcloud[:, 2].max():.3f}]")
    
    # Visualize if requested
    if args.visualize:
        print(f"\n🎨 Generating visualization...")
        visualize_pointcloud_plotly(pointcloud, args.output_html, 
                                   title=f'Real-time Point Cloud ({args.mode} mode, {args.num_points} points)')
    
    # Benchmark
    if args.mode == 'benchmark':
        print("\n⏱️  Running benchmark...")
        generator.benchmark(rgb_dict, depth_dict, ee_pos, iterations=100)
    
    print("\n✅ Done!")
