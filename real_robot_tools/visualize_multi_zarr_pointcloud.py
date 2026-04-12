#!/usr/bin/env python3
"""
Visualize point cloud data from multiple zarr files using matplotlib, supporting 2x3 layout display and dynamic video generation.

Advantages:
- Uses matplotlib rendering, stable and reliable
- Supports RGB colors, fully preserving point cloud color information
- Can easily generate images and videos
- Does not require DISPLAY, suitable for headless servers
- Retains all frame images after video generation for subsequent comparison image composition

Usage:
    # Display point cloud overview of all datasets (2x3 layout)
    python visualize_multi_zarr_pointcloud.py \
        --data-dir data/real_data \
        --output-dir visualizations/multi_zarr \
        --frame-idx 0

    # Generate video and images for Episode 0 of all data sources
    python visualize_multi_zarr_pointcloud.py \
        --data-dir data/real_data \
        --output-dir visualizations/multi_zarr \
        --create-all-episode0 \
        --frame-idx 0 \
        --fps 10
"""

import argparse
import json
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

# In the current environment, Open3D offscreen rendering is slow and produces poor results, forcing use of matplotlib
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

# Task and type mapping
TASKS = ['kettle']
TYPES = ['stereo']  # single: monocular, stereo: binocular (without single suffix)


def find_zarr_files(data_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Find all zarr files and organize them into a dictionary.

    Returns:
        {
            'drawer': {'single': 'path/to/drawer_new_single-50.zarr', 'stereo': 'path/to/drawer_new-50.zarr'},
            'kettle': {...},
            'towel': {...}
        }
    """
    data_path = Path(data_dir)
    zarr_files = {}

    # Special handling: towel's stereo file may have a misspelled name towe_newl-50.zarr
    special_cases = {
        'towel': {
            'stereo': ['towe_newl-*.zarr', 'towel_new-*.zarr', 'towel_old-*.zarr']
        }
    }

    for task in TASKS:
        zarr_files[task] = {}
        # Find monocular files (containing _single)
        single_pattern = f"{task}_new_single-*.zarr"
        single_files = list(data_path.glob(single_pattern))
        if single_files:
            zarr_files[task]['single'] = str(single_files[0])

        # Find stereo files (not containing _single, but containing task name)
        if task in special_cases:
            # Special handling: try multiple possible filename patterns
            stereo_files = []
            for pattern in special_cases[task]['stereo']:
                found = list(data_path.glob(pattern))
                stereo_files.extend([f for f in found if 'single' not in f.name])
            if stereo_files:
                zarr_files[task]['stereo'] = str(stereo_files[0])
        else:
            stereo_pattern = f"{task}_new-*.zarr"
            stereo_files = [f for f in data_path.glob(stereo_pattern)
                           if 'single' not in f.name]
            if stereo_files:
                zarr_files[task]['stereo'] = str(stereo_files[0])
    print(zarr_files)
    return zarr_files


def load_zarr_data(zarr_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load zarr data."""
    root = zarr.open(zarr_path, 'r')

    point_clouds = root['data/point_cloud'][:]
    states = root['data/state'][:]
    actions = root['data/action'][:]
    episode_ends = root['meta/episode_ends'][:]

    return point_clouds, states, actions, episode_ends


def get_episode_frames(episode_ends: np.ndarray, episode_idx: int) -> Tuple[int, int]:
    """
    Get frame range for specified episode.

    Args:
        episode_ends: episode end index array
        episode_idx: episode index (starting from 0)

    Returns:
        (start_frame, end_frame)
    """
    if episode_idx == 0:
        start_frame = 0
    else:
        start_frame = episode_ends[episode_idx - 1] + 1

    end_frame = episode_ends[episode_idx]

    return start_frame, end_frame


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

    # If Open3D not available, return dictionary format (for matplotlib)
    return {'xyz': xyz, 'rgb': rgb_normalized}


def render_pointcloud_image(
    pcd,
    width: int = 1200,
    height: int = 800,
    camera_params: Optional[dict] = None
) -> np.ndarray:
    """
    Render point cloud as image (using offscreen rendering, no DISPLAY required).

    Args:
        pcd: Open3D point cloud object or dictionary containing xyz and rgb
        width: image width
        height: image height
        camera_params: camera parameters (optional)

    Returns:
        RGB image array (height, width, 3)
    """
    # If USE_OPEN3D is False, use matplotlib directly (more stable)
    if not USE_OPEN3D:
        if isinstance(pcd, dict):
            if len(pcd['xyz']) == 0:
                return np.zeros((height, width, 3), dtype=np.uint8)
            return render_pointcloud_with_matplotlib(pcd, width, height, camera_params)
        elif HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
            # Convert to dictionary format
            if len(pcd.points) == 0:
                return np.zeros((height, width, 3), dtype=np.uint8)
            pcd_dict = {
                'xyz': np.asarray(pcd.points),
                'rgb': np.asarray(pcd.colors) if pcd.has_colors() else None
            }
            return render_pointcloud_with_matplotlib(pcd_dict, width, height, camera_params)
        else:
            return np.zeros((height, width, 3), dtype=np.uint8)

    # Check point cloud type and convert to Open3D format (if available and enabled)
    if isinstance(pcd, dict):
        # Is dictionary format, needs conversion
        if len(pcd['xyz']) == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        # Convert to Open3D format (if available)
        if HAS_OPEN3D:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd['xyz'])
            pcd_o3d.colors = o3d.utility.Vector3dVector(pcd['rgb'])
            pcd = pcd_o3d
        else:
            # If Open3D not available, use matplotlib directly
            return render_pointcloud_with_matplotlib(pcd, width, height, camera_params)

    # Open3D point cloud object check
    if HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
        if len(pcd.points) == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)

    # Prefer using Open3D offscreen rendering (better performance, designed specifically for point clouds)
    if HAS_OPEN3D and USE_OPEN3D:
        try:
            # Ensure pcd is in Open3D format
            if isinstance(pcd, dict):
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(pcd['xyz'])
                pcd_o3d.colors = o3d.utility.Vector3dVector(pcd['rgb'])
                pcd = pcd_o3d

            # Check if it's an Open3D point cloud object
            if not isinstance(pcd, o3d.geometry.PointCloud):
                raise TypeError("Point cloud object is not in Open3D format")

            # Check if rendering module exists
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
            # According to error message, need to use field_of_view_type parameter
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
                    # If all failed, use default settings
                    pass

            camera.look_at(lookat, camera_pos, up)

            # Render
            image = renderer.render_to_image()
            image_np = np.asarray(image)

            return image_np
        except Exception as e:
            print(f"[WARN] Open3D offscreen rendering failed: {e}")
            print("   Using matplotlib as fallback...")
            # Fallback to matplotlib
            if isinstance(pcd, dict):
                return render_pointcloud_with_matplotlib(pcd, width, height, camera_params)
            elif HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
                # Convert to dictionary format
                pcd_dict = {
                    'xyz': np.asarray(pcd.points),
                    'rgb': np.asarray(pcd.colors) if pcd.has_colors() else None
                }
                return render_pointcloud_with_matplotlib(pcd_dict, width, height, camera_params)

    # Fallback: use matplotlib (if Open3D not available or not enabled)
    if HAS_MATPLOTLIB:
        if isinstance(pcd, dict):
            if len(pcd['xyz']) == 0:
                return np.zeros((height, width, 3), dtype=np.uint8)
            return render_pointcloud_with_matplotlib(pcd, width, height, camera_params)
        elif HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
            if len(pcd.points) == 0:
                return np.zeros((height, width, 3), dtype=np.uint8)
            # Convert to dictionary format
            pcd_dict = {
                'xyz': np.asarray(pcd.points),
                'rgb': np.asarray(pcd.colors) if pcd.has_colors() else None
            }
            return render_pointcloud_with_matplotlib(pcd_dict, width, height, camera_params)

    # If all failed, return black image
    print("[ERROR] Cannot render point cloud: need to install matplotlib or open3d")
    return np.zeros((height, width, 3), dtype=np.uint8)


def render_pointcloud_with_matplotlib(
    pcd,
    width: int = 1200,
    height: int = 800,
    camera_params: Optional[dict] = None
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

    # Plot point cloud
    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=colors, s=10, alpha=0.8)
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

    # Set tight coordinate range based on point cloud range, let point cloud fill view (only small margin)
    try:
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

    # Reduce margins to make point cloud area larger
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
            # Use tobytes() as last resort
            buf = np.frombuffer(fig.canvas.tobytes_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Close figure to release memory
    plt.close(fig)

    # Auto-crop based on non-white pixels (crop about 20-30% of whitespace), don't restore original resolution
    try:
        # Background is pure white (255,255,255), find all non-white pixels
        non_white_mask = np.any(buf < 250, axis=2)
        if np.any(non_white_mask):
            ys, xs = np.where(non_white_mask)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            H, W, _ = buf.shape
            # Use 10% of image size as padding, crop about 20-30% of outer ring
            pad_h = int(0.1 * H)
            pad_w = int(0.1 * W)

            y_min = max(y_min - pad_h, 0)
            y_max = min(y_max + pad_h, H - 1)
            x_min = max(x_min - pad_w, 0)
            x_max = min(x_max + pad_w, W - 1)

            buf = buf[y_min:y_max + 1, x_min:x_max + 1]
    except Exception:
        # If cropping fails, keep original image
        pass

    return buf


def create_overview_image(
    zarr_files: Dict[str, Dict[str, str]],
    frame_idx: int,
    output_file: Path,
    width: int = 1800,
    height: int = 1200
):
    """
    Create 2x3 layout point cloud overview image.

    Args:
        zarr_files: zarr file path dictionary
        frame_idx: frame index to display
        output_file: output image file path
        width: total image width
        height: total image height
    """
    if not HAS_MATPLOTLIB and not HAS_OPEN3D:
        print("[ERROR] Need to install matplotlib or open3d")
        return

    print(f"\nCreating 2x3 layout overview image (frame {frame_idx})...")

    # Size of each subplot
    sub_width = width // 3
    sub_height = height // 2

    # Create total image
    overview_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Store camera parameters (for consistent viewing angle)
    camera_params = None

    # Load data and render each subplot
    for task_idx, task in enumerate(TASKS):
        for type_idx, type_name in enumerate(TYPES):
            row = type_idx
            col = task_idx

            zarr_path = zarr_files.get(task, {}).get(type_name)
            if zarr_path is None:
                print(f"[WARN] Skipping: {task} ({type_name}) - file not found")
                continue

            try:
                point_clouds, states, actions, episode_ends = load_zarr_data(zarr_path)

                actual_frame_idx = frame_idx
                if actual_frame_idx >= len(point_clouds):
                    print(f"[WARN] Frame index {actual_frame_idx} out of range, using last frame")
                    actual_frame_idx = len(point_clouds) - 1

                point_cloud = point_clouds[actual_frame_idx]
                pcd = pointcloud_to_open3d(point_cloud)

                # Check if point cloud is empty
                if isinstance(pcd, dict):
                    if len(pcd['xyz']) == 0:
                        continue
                elif HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
                    if len(pcd.points) == 0:
                        continue
                else:
                    continue

                # Save camera parameters from first subplot for subsequent subplots
                if camera_params is None:
                    # Calculate camera parameters (using matplotlib format for better compatibility)
                    camera_params = {
                        'elev': 30,
                        'azim': 45
                    }

                # Render subplot
                sub_image = render_pointcloud_image(
                    pcd,
                    width=sub_width,
                    height=sub_height,
                    camera_params=camera_params
                )

                # Place subplot into total image
                y_start = row * sub_height
                y_end = y_start + sub_height
                x_start = col * sub_width
                x_end = x_start + sub_width

                overview_image[y_start:y_end, x_start:x_end] = sub_image

                # Add title
                if HAS_OPENCV:
                    cv2.putText(
                        overview_image,
                        f"{task} ({type_name})",
                        (x_start + 10, y_start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

            except Exception as e:
                print(f"[ERROR] Processing failed {task} ({type_name}): {e}")
                continue

    # Save image
    if HAS_OPENCV:
        cv2.imwrite(str(output_file), cv2.cvtColor(overview_image, cv2.COLOR_RGB2BGR))
        print(f"[OK] Overview image saved: {output_file}")
    else:
        from PIL import Image
        Image.fromarray(overview_image).save(str(output_file))
        print(f"[OK] Overview image saved: {output_file}")


def create_dynamic_video(
    zarr_files: Dict[str, Dict[str, str]],
    output_dir: Path,
    task: str,
    type_name: str,
    episode: int,
    fps: int = 10
):
    """
    Create dynamic point cloud video for specified task and episode.

    Args:
        zarr_files: zarr file path dictionary
        output_dir: output directory
        task: task name (drawer/kettle/towel)
        type_name: type (single/stereo)
        episode: episode index (starting from 0)
        fps: video frame rate
    """
    if not HAS_MATPLOTLIB and not HAS_OPEN3D:
        print("[ERROR] Need to install matplotlib or open3d")
        return

    print(f"\n{'='*80}")
    print(f"Creating dynamic point cloud video")
    print(f"Task: {task}, Type: {type_name}, Episode: {episode}")
    print(f"{'='*80}\n")

    zarr_path = zarr_files.get(task, {}).get(type_name)
    if zarr_path is None:
        print(f"[ERROR] File not found: {task} ({type_name})")
        return

    print(f"Loading data: {Path(zarr_path).name}")
    point_clouds, states, actions, episode_ends = load_zarr_data(zarr_path)

    # Get frame range for episode
    if episode >= len(episode_ends):
        print(f"[ERROR] Episode {episode} out of range (total episodes: {len(episode_ends)})")
        return

    start_frame, end_frame = get_episode_frames(episode_ends, episode)
    end_frame -=2
    num_frames = end_frame - start_frame + 1

    print(f"Episode {episode}: frame {start_frame} - {end_frame} (total {num_frames} frames)")

    # Create output directory
    video_dir = output_dir / f"video_{task}_{type_name}_ep{episode}"
    video_dir.mkdir(parents=True, exist_ok=True)
    temp_img_dir = video_dir / "temp_images"
    temp_img_dir.mkdir(exist_ok=True)

    # Calculate camera parameters (using matplotlib format for better compatibility)
    camera_params = {
        'elev': 30,
        'azim': -45
    }

    # Generate image for each frame
    for frame_idx in tqdm(range(start_frame, end_frame + 1), desc="Generating frames"):
        point_cloud = point_clouds[frame_idx]
        pcd = pointcloud_to_open3d(point_cloud)

        # Render image
        image = render_pointcloud_image(
            pcd,
            width=1200,
            height=800,
            camera_params=camera_params
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
    video_file = video_dir / f"video_{task}_{type_name}_ep{episode}.mp4"
    img_files = sorted(temp_img_dir.glob("frame_*.png"))

    if len(img_files) > 0:
        print(f"\nGenerating video file...")
        if create_video_from_images(img_files, video_file, fps):
            print(f"[OK] Video generated: {video_file}")
            print(f"[OK] Frame images retained at: {temp_img_dir} (can be used for subsequent comparison image composition)")
        else:
            print(f"[WARN] Video generation failed, but image files saved at: {temp_img_dir}")
    else:
        print(f"[WARN] No image files generated")


def create_video_from_images(img_files: List[Path], output_video: Path, fps: int) -> bool:
    """
    Generate video from image files.

    Args:
        img_files: image file list
        output_video: output video file path
        fps: video frame rate

    Returns:
        bool: whether video was successfully generated
    """
    if len(img_files) == 0:
        return False

    # Use opencv to compose video
    if HAS_OPENCV:
        try:
            # Read first image to get target size (note: frame images may have inconsistent sizes due to cropping)
            first_img = cv2.imread(str(img_files[0]))
            if first_img is None:
                return False

            height, width, _ = first_img.shape

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

            for img_file in tqdm(img_files, desc="Composing video"):
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                # If current frame size differs from first frame, resize to uniform size
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height))

                out.write(img)

            out.release()

            # Don't delete image files, keep for subsequent comparison image composition
            return True
        except Exception as e:
            print(f"[WARN] opencv method failed: {e}")

    # Use ffmpeg (if available)
    try:
        import subprocess
        subprocess.run([
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(img_files[0].parent / 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_video)
        ], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return False


def create_all_episode0_videos(
    zarr_files: Dict[str, Dict[str, str]],
    output_dir: Path,
    fps: int = 10,
    overview_frame_idx: int = 0
):
    """
    Generate videos and images for Episode 0 of all 6 data sources, and create 2x3 overview image.
    """
    if not HAS_MATPLOTLIB and not HAS_OPEN3D:
        print("[ERROR] Need to install matplotlib or open3d")
        return

    print(f"\n{'='*80}")
    print(f"Generating visualization for Episode 0 of all data sources")
    print(f"Overview image frame index: {overview_frame_idx}")
    print(f"{'='*80}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate video for each data source
    for task in TASKS:
        for type_name in TYPES:
            zarr_path = zarr_files.get(task, {}).get(type_name)
            if zarr_path is None:
                print(f"[WARN] Skipping: {task} ({type_name}) - file not found")
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {task} ({type_name})")
            print(f"{'='*60}")

            try:
                create_dynamic_video(
                    zarr_files,
                    output_dir,
                    task,
                    type_name,
                    episode=0,
                    fps=fps
                )
            except Exception as e:
                print(f"[ERROR] Processing failed {task} ({type_name}): {e}")
                import traceback
                traceback.print_exc()
                continue

    # Generate 2x3 overview image
    overview_file = output_dir / f"all_episode0_overview_frame_{overview_frame_idx:06d}.png"
    create_overview_image(
        zarr_files,
        overview_frame_idx,
        overview_file
    )

    print(f"\n{'='*80}")
    print(f"[OK] Episode 0 visualization for all data sources completed!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


def main():
    if not HAS_MATPLOTLIB and not HAS_OPEN3D:
        print("[ERROR] Need to install matplotlib or open3d:")
        print("   pip install matplotlib")
        print("   or")
        print("   pip install open3d")
        return

    if HAS_MATPLOTLIB and not USE_OPEN3D:
        print("[OK] Using matplotlib for visualization (default, more stable)")
    elif HAS_OPEN3D and USE_OPEN3D:
        print("[OK] Using Open3D for visualization")
    elif HAS_MATPLOTLIB:
        print("[OK] Using matplotlib for visualization")

    parser = argparse.ArgumentParser(
        description='Visualize point cloud data from multiple zarr files using matplotlib',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/real_data',
        help='Directory containing zarr files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations/multi_zarr',
        help='Output directory'
    )
    parser.add_argument(
        '--frame-idx',
        type=int,
        default=0,
        help='Frame index to display (for overview image)'
    )
    parser.add_argument(
        '--create-all-episode0',
        action='store_true',
        help='Generate videos and images for Episode 0 of all 6 data sources, and create 2x3 overview image'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Video frame rate (default: 10)'
    )

    args = parser.parse_args()

    # Find zarr files
    zarr_files = find_zarr_files(args.data_dir)

    print("\nFound zarr files:")
    for task in TASKS:
        for type_name in TYPES:
            path = zarr_files.get(task, {}).get(type_name)
            if path:
                print(f"  [OK] {task} ({type_name}): {Path(path).name}")
            else:
                print(f"  [ERROR] {task} ({type_name}): not found")

    output_dir = Path(args.output_dir)

    if args.create_all_episode0:
        # Generate videos and overview image for Episode 0 of all data sources
        create_all_episode0_videos(
            zarr_files,
            output_dir,
            fps=args.fps,
            overview_frame_idx=args.frame_idx
        )
    else:
        # Create overview image
        overview_file = output_dir / f"multi_zarr_overview_frame_{args.frame_idx:06d}.png"
        create_overview_image(
            zarr_files,
            args.frame_idx,
            overview_file
        )


if __name__ == '__main__':
    main()
