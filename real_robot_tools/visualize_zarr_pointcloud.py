#!/usr/bin/env python3
"""
Visualize point clouds from zarr files with interactive HTML output.

Usage:
    # Visualize specific frames
    python visualize_zarr_pointcloud.py \
        --zarr-path data/zarr/test_01/test_task-1.zarr \
        --output-dir visualizations/zarr_pointclouds \
        --frames 0 50 100 150

    # Visualize multiple frames with stride
    python visualize_zarr_pointcloud.py \
        --zarr-path data/zarr/test_01/test_task-1.zarr \
        --output-dir visualizations/zarr_pointclouds \
        --num-frames 10 \
        --stride 30

    # Create video from frames
    python visualize_zarr_pointcloud.py \
        --zarr-path data/zarr/test_01/test_task-1.zarr \
        --output-dir visualizations/zarr_pointclouds \
        --create-video \
        --fps 10
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
import zarr
from tqdm import tqdm


def load_zarr_data(zarr_path: str):
    """Load zarr data."""
    root = zarr.open(zarr_path, 'r')

    point_clouds = root['data/point_cloud'][:]
    states = root['data/state'][:]
    actions = root['data/action'][:]
    episode_ends = root['meta/episode_ends'][:]

    return point_clouds, states, actions, episode_ends


def create_pointcloud_figure(
    point_cloud: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    frame_idx: int,
    total_frames: int,
    title: str = "Point Cloud Visualization"
) -> go.Figure:
    """
    Create a Plotly figure for point cloud visualization.

    Args:
        point_cloud: (N, 6) array with xyz + rgb
        state: (state_dim,) current state
        action: (action_dim,) current action
        frame_idx: Current frame index
        total_frames: Total number of frames
        title: Figure title

    Returns:
        Plotly figure object
    """
    # Extract xyz and rgb
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:6]

    # Filter out zero-padded points
    valid_mask = ~np.all(xyz == 0, axis=1)
    xyz = xyz[valid_mask]
    rgb = rgb[valid_mask]

    # Convert RGB to hex colors for Plotly
    rgb_255 = (rgb * 255).astype(int)
    colors = [f'rgb({r},{g},{b})' for r, g, b in rgb_255]

    # Create 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors,
                opacity=0.8
            ),
            text=[f'x: {x:.3f}<br>y: {y:.3f}<br>z: {z:.3f}' 
                  for x, y, z in xyz],
            hovertemplate='%{text}<extra></extra>'
        )
    ])

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title}<br>Frame {frame_idx}/{total_frames} | Points: {len(xyz):,}",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        width=1200,
        height=800,
        showlegend=False
    )

    # Add state and action information as annotations
    state_str = "State: [" + ", ".join([f"{s:.2f}" for s in state]) + "]"
    action_str = "Action: [" + ", ".join([f"{a:.2f}" for a in action]) + "]"

    fig.add_annotation(
        text=state_str,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="white",
        opacity=0.8,
        xanchor='left',
        yanchor='top',
        font=dict(size=10, family="monospace")
    )

    fig.add_annotation(
        text=action_str,
        xref="paper", yref="paper",
        x=0.02, y=0.93,
        showarrow=False,
        bgcolor="white",
        opacity=0.8,
        xanchor='left',
        yanchor='top',
        font=dict(size=10, family="monospace")
    )

    return fig


def visualize_zarr_pointclouds(
    zarr_path: str,
    output_dir: str,
    frames: Optional[List[int]] = None,
    num_frames: Optional[int] = None,
    stride: int = 1,
    create_video: bool = False,
    fps: int = 10
):
    """
    Visualize point clouds from zarr file.

    Args:
        zarr_path: Path to zarr file
        output_dir: Output directory for HTML files
        frames: Specific frame indices to visualize
        num_frames: Number of frames to visualize (evenly spaced)
        stride: Frame stride if num_frames not specified
        create_video: Create a video from frames
        fps: Frames per second for video
    """
    print(f"\n{'='*80}")
    print(f"Visualizing Point Clouds from Zarr")
    print(f"Zarr path: {zarr_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading zarr data...")
    point_clouds, states, actions, episode_ends = load_zarr_data(zarr_path)
    total_frames = len(point_clouds)

    print(f"Total frames: {total_frames}")
    print(f"Point cloud shape: {point_clouds.shape}")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")

    # Determine which frames to visualize
    if frames is not None:
        frame_indices = frames
    elif num_frames is not None:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        frame_indices = list(range(0, total_frames, stride))

    frame_indices = [idx for idx in frame_indices if 0 <= idx < total_frames]

    print(f"\nVisualizing {len(frame_indices)} frames: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate HTML files for each frame
    html_files = []
    for frame_idx in tqdm(frame_indices, desc="Generating visualizations"):
        point_cloud = point_clouds[frame_idx]
        state = states[frame_idx]
        action = actions[frame_idx]

        # Create figure
        fig = create_pointcloud_figure(
            point_cloud, state, action, frame_idx, total_frames - 1,
            title=f"Zarr Point Cloud - {Path(zarr_path).stem}"
        )

        # Save to HTML
        html_file = output_path / f"frame_{frame_idx:06d}.html"
        fig.write_html(str(html_file))
        # Also write a sidecar .npz with raw xyz and rgb arrays for robust downstream use
        try:
            xyz = point_cloud[:, :3]
            valid_mask = ~np.all(xyz == 0, axis=1)
            xyz = xyz[valid_mask]
            rgb = point_cloud[:, 3:6][valid_mask]
            # rgb stored as 0..255 ints
            rgb_255 = (rgb * 255).astype(int)
            npz_file = html_file.with_suffix('.npz')
            np.savez_compressed(str(npz_file), x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], rgb=rgb_255)
        except Exception:
            # don't fail visualization generation for sidecar write errors
            pass
        html_files.append(html_file)

    # Create index HTML with navigation
    create_index_html(output_path, html_files, zarr_path, frame_indices)

    print(f"\n{'='*80}")
    print(f"✅ Visualization complete!")
    print(f"Generated {len(html_files)} HTML files in: {output_path}")
    print(f"Open index.html to browse all frames")
    print(f"{'='*80}\n")

    # Create video if requested
    if create_video:
        print("\nCreating video...")
        create_video_from_htmls(html_files, output_path / "pointcloud_video.mp4", fps)


def create_index_html(
    output_dir: Path,
    html_files: List[Path],
    zarr_path: str,
    frame_indices: List[int]
):
    """Create an index HTML page with navigation."""
    index_html = output_dir / "index.html"

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zarr Point Cloud Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .info {{
            background-color: white;
            padding: 15px;
            margin: 20px auto;
            max-width: 1200px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            max-width: 1200px;
            margin: 20px auto;
        }}
        .frame-card {{
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .frame-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .frame-link {{
            color: #007bff;
            text-decoration: none;
            font-weight: bold;
            font-size: 18px;
        }}
        .frame-link:hover {{
            text-decoration: underline;
        }}
        .stats {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <h1>🎯 Zarr Point Cloud Visualization</h1>

    <div class="info">
        <h2>Dataset Information</h2>
        <p><strong>Zarr Path:</strong> {zarr_path}</p>
        <p><strong>Total Frames Visualized:</strong> {len(html_files)}</p>
        <p><strong>Frame Indices:</strong> {', '.join(map(str, frame_indices[:20]))}{'...' if len(frame_indices) > 20 else ''}</p>
    </div>

    <div class="grid">
"""

    for html_file, frame_idx in zip(html_files, frame_indices):
        html_content += f"""
        <div class="frame-card">
            <a class="frame-link" href="{html_file.name}" target="_blank">
                Frame {frame_idx}
            </a>
            <div class="stats">Click to view</div>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    with open(index_html, 'w') as f:
        f.write(html_content)

    print(f"Created index page: {index_html}")


def create_video_from_htmls(html_files: List[Path], output_video: Path, fps: int):
    """Create a video from HTML visualizations (requires selenium and ffmpeg)."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from PIL import Image
        import subprocess

        print("Capturing screenshots...")

        # Setup headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1200,800")
        driver = webdriver.Chrome(options=chrome_options)

        temp_dir = output_video.parent / "temp_frames"
        temp_dir.mkdir(exist_ok=True)

        for i, html_file in enumerate(tqdm(html_files, desc="Capturing frames")):
            driver.get(f"file://{html_file.absolute()}")
            driver.implicitly_wait(2)
            screenshot = temp_dir / f"frame_{i:06d}.png"
            driver.save_screenshot(str(screenshot))

        driver.quit()

        # Create video using ffmpeg
        print("Creating video...")
        subprocess.run([
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(temp_dir / 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_video)
        ], check=True)

        # Cleanup
        for f in temp_dir.glob("*.png"):
            f.unlink()
        temp_dir.rmdir()

        print(f"Video saved to: {output_video}")

    except ImportError:
        print("⚠️  Video creation requires selenium and ffmpeg. Skipping video generation.")
    except Exception as e:
        print(f"⚠️  Failed to create video: {e}")


def main():
    parser = argparse.ArgumentParser(description='Visualize point clouds from zarr files')
    parser.add_argument('--zarr-path', type=str, required=True,
                        help='Path to zarr file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for HTML files')
    parser.add_argument('--frames', type=int, nargs='+',
                        help='Specific frame indices to visualize')
    parser.add_argument('--num-frames', type=int,
                        help='Number of frames to visualize (evenly spaced)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Frame stride if num-frames not specified (default: 1)')
    parser.add_argument('--create-video', action='store_true',
                        help='Create video from frames (requires selenium and ffmpeg)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for video (default: 10)')

    args = parser.parse_args()

    visualize_zarr_pointclouds(
        args.zarr_path,
        args.output_dir,
        frames=args.frames,
        num_frames=args.num_frames,
        stride=args.stride,
        create_video=args.create_video,
        fps=args.fps
    )


if __name__ == '__main__':
    main()