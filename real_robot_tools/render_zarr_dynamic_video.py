#!/usr/bin/env python3
"""
Render a single Zarr point cloud sequence to a dynamic video (Matplotlib offscreen rendering).

Notes:
- This script focuses on "single zarr -> dynamic video", no longer includes 2x3 overview and multi-source scanning logic
- For multi-data-source batch processing, please use the bash script in the same directory:
  `batch_render_zarr_dynamic_videos.sh`
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from tqdm import tqdm

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

# In the current environment, Open3D offscreen rendering is slow and produces poor results, forcing the use of matplotlib
USE_OPEN3D = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

def build_output_stem(zarr_path: str, episode_idx: int, output_name: Optional[str]) -> str:
    """Build output filename (without extension)."""
    if output_name is not None and output_name.strip():
        name = output_name.strip()
        if name.lower().endswith('.mp4'):
            name = name[:-4]
        return name
    return f"{Path(zarr_path).stem}_ep{episode_idx}"


def load_zarr_data(zarr_path: str):
    """Load zarr data handle (lazy loading for point clouds) and episode boundaries."""
    root = zarr.open(zarr_path, 'r')
    point_clouds = root['data/point_cloud']
    episode_ends = root['meta/episode_ends'][:]
    return point_clouds, episode_ends


def get_episode_frames(episode_ends: np.ndarray, episode_idx: int) -> Tuple[int, int]:
    """
    Get the frame range for a specified episode.

    Args:
        episode_ends: Array of episode end indices
        episode_idx: Episode index (0-based)

    Returns:
        (start_frame, end_frame)
    """
    if episode_idx == 0:
        start_frame = 0
    else:
        start_frame = episode_ends[episode_idx - 1] + 1
    
    end_frame = episode_ends[episode_idx]
    
    return start_frame, end_frame


def compute_axis_limits_for_frames(
    point_clouds: np.ndarray,
    start_frame: int,
    end_frame: int,
    padding_ratio: float = 0.08,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Compute the global axis range for a specified frame range (fixed coordinate system, reduces video jitter).

    Uses percentiles to remove outliers, avoiding individual anomalous points from expanding the view.
    """
    xyz = point_clouds[start_frame:end_frame + 1, :, :3].reshape(-1, 3)
    valid_mask = ~np.all(xyz == 0, axis=1)
    xyz = xyz[valid_mask]

    if len(xyz) == 0:
        return None

    if 0.0 <= lower_percentile < upper_percentile <= 100.0:
        lo = np.percentile(xyz, lower_percentile, axis=0)
        hi = np.percentile(xyz, upper_percentile, axis=0)
    else:
        lo = np.min(xyz, axis=0)
        hi = np.max(xyz, axis=0)

    extent = np.maximum(hi - lo, 1e-4)
    max_extent = np.max(extent)
    center = (lo + hi) / 2.0
    half_span = max_extent * (0.5 + padding_ratio)

    return {
        'x': (center[0] - half_span, center[0] + half_span),
        'y': (center[1] - half_span, center[1] + half_span),
        'z': (center[2] - half_span, center[2] + half_span)
    }


def pointcloud_to_open3d(point_cloud: np.ndarray):
    """
    Convert numpy point cloud array to Open3D point cloud object (if available) or return numpy array.

    Args:
        point_cloud: (N, 6) array with xyz + rgb

    Returns:
        Open3D PointCloud object (if available) or dictionary containing xyz and rgb
    """
    # Extract xyz and rgb
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:6]

    # Filter out zero-padded points
    valid_mask = ~np.all(xyz == 0, axis=1)
    xyz = xyz[valid_mask]
    rgb = rgb[valid_mask]

    if len(xyz) == 0:
        # If no valid points
        if HAS_OPEN3D and USE_OPEN3D:
            return o3d.geometry.PointCloud()
        else:
            return {'xyz': np.array([]), 'rgb': np.array([])}

    # Process RGB color values
    rgb_max = np.max(rgb)
    if rgb_max > 1.0:
        # Already in 0-255 range, convert to 0-1 range
        rgb_normalized = rgb / 255.0
    else:
        # Already in 0-1 range
        rgb_normalized = rgb.copy()

    # Handle NaN and inf values
    rgb_normalized = np.nan_to_num(rgb_normalized, nan=0.5, posinf=1.0, neginf=0.0)
    rgb_normalized = np.clip(rgb_normalized, 0.0, 1.0)

    # Prefer returning Open3D format (if available)
    if HAS_OPEN3D:
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb_normalized)
        return pcd

    # If Open3D is not available, return dictionary format (for matplotlib)
    return {'xyz': xyz, 'rgb': rgb_normalized}


def render_pointcloud_image(
    pcd,
    width: int = 1200,
    height: int = 800,
    camera_params: Optional[dict] = None,
    dynamic_axis: bool = True,
    auto_crop: bool = False
) -> np.ndarray:
    """
    Render point cloud to image (using offscreen rendering, no DISPLAY required).

    Args:
        pcd: Open3D point cloud object or dictionary containing xyz and rgb
        width: Image width
        height: Image height
        camera_params: Camera parameters (optional)
        dynamic_axis: Whether to dynamically scale axes per frame
        auto_crop: Whether to auto-crop white borders (recommended off for videos to avoid frame jitter)

    Returns:
        RGB image array (height, width, 3)
    """
    # If USE_OPEN3D is False, directly use matplotlib (more stable)
    if not USE_OPEN3D:
        if isinstance(pcd, dict):
            if len(pcd['xyz']) == 0:
                return np.zeros((height, width, 3), dtype=np.uint8)
            return render_pointcloud_with_matplotlib(
                pcd, width, height, camera_params, dynamic_axis, auto_crop
            )
        elif HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
            # Convert to dictionary format
            if len(pcd.points) == 0:
                return np.zeros((height, width, 3), dtype=np.uint8)
            pcd_dict = {
                'xyz': np.asarray(pcd.points),
                'rgb': np.asarray(pcd.colors) if pcd.has_colors() else None
            }
            return render_pointcloud_with_matplotlib(
                pcd_dict, width, height, camera_params, dynamic_axis, auto_crop
            )
        else:
            return np.zeros((height, width, 3), dtype=np.uint8)

    # Check point cloud type and convert to Open3D format (if available and enabled)
    if isinstance(pcd, dict):
        # Is dictionary format, need to convert
        if len(pcd['xyz']) == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        # Convert to Open3D format (if available)
        if HAS_OPEN3D:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd['xyz'])
            pcd_o3d.colors = o3d.utility.Vector3dVector(pcd['rgb'])
            pcd = pcd_o3d
        else:
            # If Open3D is not available, directly use matplotlib
            return render_pointcloud_with_matplotlib(
                pcd, width, height, camera_params, dynamic_axis, auto_crop
            )

    # Open3D point cloud object check
    if HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
        if len(pcd.points) == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Prefer Open3D offscreen rendering (better performance, designed specifically for point clouds)
    if HAS_OPEN3D and USE_OPEN3D:
        try:
            # Ensure pcd is in Open3D format
            if isinstance(pcd, dict):
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(pcd['xyz'])
                pcd_o3d.colors = o3d.utility.Vector3dVector(pcd['rgb'])
                pcd = pcd_o3d
            
            # Check if it is an Open3D point cloud object
            if not isinstance(pcd, o3d.geometry.PointCloud):
                raise TypeError("Point cloud object is not in Open3D format")
            
            # Check if rendering module is available
            if not hasattr(o3d.visualization, 'rendering'):
                raise ImportError("Open3D version does not support rendering module")
            
            # Use offscreen renderer (no DISPLAY required)
            renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
            
            # Create material
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = 2.0
            
            # Add point cloud
            renderer.scene.add_geometry("pointcloud", pcd, material)
            
            # Calculate camera parameters
            bbox = pcd.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = bbox.get_extent()
            max_extent = np.max(extent)
            
            # Calculate camera position
            camera_pos = center + np.array([max_extent * 1.5, max_extent * 1.5, max_extent * 1.5])
            lookat = center
            up = [0, 0, 1]
            
            # Set camera (using Open3D's offscreen rendering API)
            # Based on error messages, need to use field_of_view_type parameter
            camera = renderer.scene.camera
            try:
                # Try using version with field_of_view_type
                camera.set_projection(
                    60.0,  # field_of_view
                    width / height,  # aspect_ratio
                    0.1,  # near_plane
                    max_extent * 10,  # far_plane
                    o3d.visualization.rendering.Camera.FovType.Vertical  # field_of_view_type
                )
            except (AttributeError, TypeError):
                # If failed, try using Projection type
                try:
                    # Calculate frustum parameters
                    tan_fov = np.tan(np.radians(60.0) / 2.0)
                    right = near_plane * tan_fov * (width / height)
                    left = -right
                    top = near_plane * tan_fov
                    bottom = -top
                    camera.set_projection(
                        o3d.visualization.rendering.Camera.Projection.Perspective,
                        left, right, bottom, top,
                        0.1,  # near
                        max_extent * 10  # far
                    )
                except Exception:
                    # If all fail, use default settings
                    pass
            
            camera.look_at(lookat, camera_pos, up)
            
            # Render
            image = renderer.render_to_image()
            image_np = np.asarray(image)
            
            return image_np
        except Exception as e:
            print(f"Open3D offscreen rendering failed: {e}")
            print("   Using matplotlib as fallback...")
            # Fallback to matplotlib
            if isinstance(pcd, dict):
                return render_pointcloud_with_matplotlib(
                    pcd, width, height, camera_params, dynamic_axis, auto_crop
                )
            elif HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
                # Convert to dictionary format
                pcd_dict = {
                    'xyz': np.asarray(pcd.points),
                    'rgb': np.asarray(pcd.colors) if pcd.has_colors() else None
                }
                return render_pointcloud_with_matplotlib(
                    pcd_dict, width, height, camera_params, dynamic_axis, auto_crop
                )

    # Fallback: Use matplotlib (if Open3D is not available or not enabled)
    if HAS_MATPLOTLIB:
        if isinstance(pcd, dict):
            if len(pcd['xyz']) == 0:
                return np.zeros((height, width, 3), dtype=np.uint8)
            return render_pointcloud_with_matplotlib(
                pcd, width, height, camera_params, dynamic_axis, auto_crop
            )
        elif HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
            if len(pcd.points) == 0:
                return np.zeros((height, width, 3), dtype=np.uint8)
            # Convert to dictionary format
            pcd_dict = {
                'xyz': np.asarray(pcd.points),
                'rgb': np.asarray(pcd.colors) if pcd.has_colors() else None
            }
            return render_pointcloud_with_matplotlib(
                pcd_dict, width, height, camera_params, dynamic_axis, auto_crop
            )

    # If all failed, return black image
    print("Cannot render point cloud: matplotlib or open3d needs to be installed")
    return np.zeros((height, width, 3), dtype=np.uint8)


def render_pointcloud_with_matplotlib(
    pcd,
    width: int = 1200,
    height: int = 800,
    camera_params: Optional[dict] = None,
    dynamic_axis: bool = True,
    auto_crop: bool = False
) -> np.ndarray:
    """
    Render point cloud using matplotlib (fallback method).
    """
    # Handle different input formats
    if isinstance(pcd, dict):
        points = pcd['xyz']
        colors = pcd['rgb'] if 'rgb' in pcd else None
    elif HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
        if len(pcd.points) == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    else:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(points) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Create figure
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Draw point cloud
    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=1, alpha=0.8)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c='blue', s=1, alpha=0.8)

    # Set viewing angle
    if camera_params is None:
        ax.view_init(elev=30, azim=45)
    else:
        if 'elev' in camera_params:
            elev = camera_params['elev']
        else:
            elev = 30
        if 'azim' in camera_params:
            azim = camera_params['azim']
        else:
            azim = 45
        ax.view_init(elev=elev, azim=azim)

    # Axis range: prefer global fixed range (can significantly reduce dynamic video jitter)
    try:
        axis_limits = None
        if camera_params is not None and isinstance(camera_params, dict):
            axis_limits = camera_params.get('axis_limits')

        if axis_limits is not None:
            ax.set_xlim(*axis_limits['x'])
            ax.set_ylim(*axis_limits['y'])
            ax.set_zlim(*axis_limits['z'])
        elif dynamic_axis:
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            z_min, z_max = points[:, 2].min(), points[:, 2].max()

            x_range = max(x_max - x_min, 1e-6)
            y_range = max(y_max - y_min, 1e-6)
            z_range = max(z_max - z_min, 1e-6)
            max_range = max(x_range, y_range, z_range)

            pad = 0.05 * max_range  # 5% padding
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_zlim(z_min - pad, z_max + pad)

        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    # Set background to pure white and turn off axes (don't show cube/ticks)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_axis_off()

    # Reduce outer margins to make point cloud area larger
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    # Save to memory
    fig.canvas.draw()
    
    # Use compatible API to get image data
    try:
        # Newer matplotlib versions use buffer_rgba()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert to RGB (remove alpha channel)
        buf = buf[:, :, :3]
    except AttributeError:
        try:
            # Older versions use tostring_rgb()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Use tobytes() as last fallback
            buf = np.frombuffer(fig.canvas.tobytes_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Close figure to release memory
    plt.close(fig)

    # Auto-cropping white borders causes frame size and view jumps, disabled by default
    if auto_crop:
        try:
            non_white_mask = np.any(buf < 250, axis=2)
            if np.any(non_white_mask):
                ys, xs = np.where(non_white_mask)
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()

                H, W, _ = buf.shape
                pad_h = int(0.1 * H)
                pad_w = int(0.1 * W)

                y_min = max(y_min - pad_h, 0)
                y_max = min(y_max + pad_h, H - 1)
                x_min = max(x_min - pad_w, 0)
                x_max = min(x_max + pad_w, W - 1)

                buf = buf[y_min:y_max + 1, x_min:x_max + 1]
        except Exception:
            pass
    
    return buf


def create_dynamic_video(
    zarr_path: str,
    output_dir: Path,
    episode_idx: int,
    fps: int = 30,
    width: int = 1200,
    height: int = 800,
    max_frames: int = 0,
    output_name: Optional[str] = None,
    dynamic_axis: bool = False,
    auto_crop: bool = False,
    cleanup_frames: bool = False
):
    """
    Create dynamic point cloud video for specified task and episode.

    Args:
        zarr_path: Zarr file path
        output_dir: Output directory
        episode_idx: Episode index (0-based)
        fps: Video frame rate
        width: Render width
        height: Render height
        max_frames: Maximum frames to render (0 means entire episode)
        output_name: Output video name (optional, with or without .mp4)
        dynamic_axis: Whether to dynamically scale axes per frame
        auto_crop: Whether to auto-crop white borders
        cleanup_frames: Delete intermediate frame images after video generation
    """
    if not HAS_MATPLOTLIB and not HAS_OPEN3D:
        print("matplotlib or open3d needs to be installed")
        return

    print(f"\n{'='*80}")
    print(f"Creating dynamic point cloud video")
    print(f"Input: {Path(zarr_path).name}")
    print(f"Episode: {episode_idx}")
    print(f"{'='*80}\n")

    if not Path(zarr_path).exists():
        print(f"File does not exist: {zarr_path}")
        return

    point_clouds, episode_ends = load_zarr_data(zarr_path)

    # Get frame range for episode
    if episode_idx >= len(episode_ends):
        print(f"Episode {episode_idx} out of range (total episodes: {len(episode_ends)})")
        return
    
    start_frame, end_frame = get_episode_frames(episode_ends, episode_idx)
    if max_frames > 0:
        end_frame = min(end_frame, start_frame + max_frames - 1)
    num_frames = end_frame - start_frame + 1

    print(f"Frame range: {start_frame} - {end_frame} ({num_frames} frames total)")

    # Create output directory
    output_stem = build_output_stem(zarr_path, episode_idx, output_name)
    video_dir = output_dir / output_stem
    video_dir.mkdir(parents=True, exist_ok=True)
    temp_img_dir = video_dir / "temp_images"
    temp_img_dir.mkdir(exist_ok=True)

    # Calculate camera parameters (using matplotlib format, better compatibility)
    camera_params = {
        'elev': 30,
        'azim': 45
    }

    # Compute global fixed coordinate range for dynamic video to avoid jitter from per-frame scaling
    if not dynamic_axis and num_frames > 0:
        episode_points = point_clouds[start_frame:end_frame + 1]
        axis_limits = compute_axis_limits_for_frames(
            episode_points,
            0,
            num_frames - 1
        )
        if axis_limits is not None:
            camera_params['axis_limits'] = axis_limits
            print(
                "Fixed coordinate range: "
                f"X[{axis_limits['x'][0]:.3f}, {axis_limits['x'][1]:.3f}] "
                f"Y[{axis_limits['y'][0]:.3f}, {axis_limits['y'][1]:.3f}] "
                f"Z[{axis_limits['z'][0]:.3f}, {axis_limits['z'][1]:.3f}]"
            )
        else:
            print("No valid point cloud for computing fixed coordinate range, falling back to dynamic axes")
            dynamic_axis = True

    # Generate image for each frame
    for frame_idx in tqdm(range(start_frame, end_frame + 1), desc="Generating frames"):
        point_cloud = point_clouds[frame_idx]
        pcd = pointcloud_to_open3d(point_cloud)
        
        # Render image
        image = render_pointcloud_image(
            pcd,
            width=width,
            height=height,
            camera_params=camera_params,
            dynamic_axis=dynamic_axis,
            auto_crop=auto_crop
        )

        # Save image
        frame_num = frame_idx - start_frame
        img_file = temp_img_dir / f"frame_{frame_num:06d}.png"
        
        if HAS_OPENCV:
            cv2.imwrite(str(img_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            from PIL import Image
            Image.fromarray(image).save(str(img_file))

    # Generate video
    video_file = video_dir / f"{output_stem}.mp4"
    img_files = sorted(temp_img_dir.glob("frame_*.png"))
    
    if len(img_files) > 0:
        print(f"\nGenerating video file...")
        if create_video_from_images(img_files, video_file, fps):
            print(f"Video generated: {video_file}")
            if cleanup_frames:
                for img_file in img_files:
                    img_file.unlink(missing_ok=True)
                print(f"Intermediate frames deleted: {temp_img_dir}")
            else:
                print(f"Frame images preserved at: {temp_img_dir} (can be used for subsequent comparison images)")
        else:
            print(f"Video generation failed, but image files saved at: {temp_img_dir}")
    else:
        print(f"No image files generated")


def create_video_from_images(img_files: List[Path], output_video: Path, fps: int) -> bool:
    """
    Generate video from image files.

    Args:
        img_files: List of image files
        output_video: Output video file path
        fps: Video frame rate

    Returns:
        bool: Whether video was successfully generated
    """
    if len(img_files) == 0:
        return False

    # Prefer ffmpeg (H.264 + faststart, better compatibility)
    try:
        import subprocess
        subprocess.run([
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(img_files[0].parent / 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            str(output_video)
        ], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # When ffmpeg is not available, fall back to opencv
    if HAS_OPENCV:
        try:
            # Read first image to get target size (note: frame images may vary in size due to cropping)
            first_img = cv2.imread(str(img_files[0]))
            if first_img is None:
                return False
            
            height, width, _ = first_img.shape

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

            for img_file in tqdm(img_files, desc="Compositing video"):
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                # If current frame size differs from first frame, resize to uniform size
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height))
                
                out.write(img)
            
            out.release()

            # Don't delete image files, preserve for subsequent comparison images
            return True
        except Exception as e:
            print(f"opencv method failed: {e}")
    
    return False


def main():
    if not HAS_MATPLOTLIB and not HAS_OPEN3D:
        print("matplotlib or open3d needs to be installed:")
        print("   pip install matplotlib")
        print("   or")
        print("   pip install open3d")
        return

    if HAS_MATPLOTLIB and not USE_OPEN3D:
        print("Using matplotlib for visualization (default, more stable)")
    elif HAS_OPEN3D and USE_OPEN3D:
        print("Using Open3D for visualization")
    elif HAS_MATPLOTLIB:
        print("Using matplotlib for visualization")
    
    parser = argparse.ArgumentParser(
        description='Render a single zarr point cloud sequence to a dynamic video',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--zarr-path',
        type=str,
        required=True,
        help='Input zarr file path'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations/multi_zarr',
        help='Output directory (a subdirectory will be created under it)'
    )
    parser.add_argument(
        '--episode-idx',
        type=int,
        default=0,
        help='Episode index to visualize (default: 0)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Video frame rate (default: 30)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1200,
        help='Render width (default: 1200)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=800,
        help='Render height (default: 800)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=0,
        help='Maximum frames to render (default: 0, meaning entire episode)'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='',
        help='Output video filename (optional, with or without .mp4)'
    )
    parser.add_argument(
        '--dynamic-axis',
        action='store_true',
        help='Enable per-frame dynamic axis scaling (off by default to reduce video jitter)'
    )
    parser.add_argument(
        '--auto-crop',
        action='store_true',
        help='Enable auto-crop white borders (off by default, recommended off for videos)'
    )
    parser.add_argument(
        '--cleanup-frames',
        action='store_true',
        help='Delete intermediate frame images after video generation'
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_dynamic_video(
        zarr_path=args.zarr_path,
        output_dir=output_dir,
        episode_idx=args.episode_idx,
        fps=args.fps,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        output_name=args.output_name,
        dynamic_axis=args.dynamic_axis,
        auto_crop=args.auto_crop,
        cleanup_frames=args.cleanup_frames
    )


if __name__ == '__main__':
    main()
