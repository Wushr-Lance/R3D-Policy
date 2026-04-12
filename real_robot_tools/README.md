# Real Robot Data Processing & Deployment Tools

This directory contains core tools for converting LeRobot format real-robot data into R3D training format, as well as real-robot deployment inference scripts.

## Directory Structure

```
real_robot_tools/
├── deploy/                          # Deployment scripts
│   ├── control_robot_dp3.py         # Main deployment script with policy inference
│   └── control_utils.py             # Control utilities and robot communication
├── scripts/                         # Shell scripts
│   ├── process_depths_cdm.sh        # Batch CDM depth denoising
│   └── batch_render_zarr_dynamic_videos.sh  # Batch video rendering
├── convert_parquet_to_zarr.py       # LeRobot to Zarr conversion
├── process_depth_with_cdm.py        # CDM depth optimization
├── realtime_pointcloud.py           # Real-time point cloud generation
├── visualize_zarr_pointcloud.py     # Point cloud visualization (HTML)
├── render_zarr_dynamic_video.py     # Dynamic video rendering
├── html_pointclouds_to_videos.py    # HTML to video conversion
├── visualize_multi_zarr_pointcloud.py # Multi-zarr visualization
├── calib.json                       # Camera extrinsic calibration
└── inst.json                        # Camera intrinsic parameters
```

## Core Workflow

### 1. CDM Depth Denoising
```bash
bash scripts/process_depths_cdm.sh data/task1 data/cdm_d435.ckpt
```

### 2. Convert to Zarr Format
```bash
python convert_parquet_to_zarr.py \
    --input-dir data/task1 \
    --calib-file calib.json \
    --inst-file inst.json \
    --output-dir data/zarr \
    --task-name task1 \
    --num-episodes 50 \
    --use-cdm-depth
```

### 3. Visualize Point Clouds
```bash
python visualize_zarr_pointcloud.py \
    --zarr-path data/zarr/task1/task1-50.zarr \
    --output-dir visualizations/task1 \
    --num-frames 5
```

## Deployment

### Debug with Zarr Replay
```bash
export REPLAY_ZARR="data/zarr/task1/task1-50.zarr"
python deploy/control_robot_dp3.py --robot.type=gello --control.type=record ...
```

### Real-time Inference
```bash
export DP3_CHECKPOINT="path/to/checkpoint.ckpt"
python deploy/control_robot_dp3.py --robot.type=gello --control.type=record ...
```

## Configuration Files

- **calib.json**: Camera extrinsic matrices (transformation to robot base)
- **inst.json**: Camera intrinsic matrices and ROI cropping ranges

For detailed documentation, see [REAL_ROBOT_DEPLOYMENT.md](../REAL_ROBOT_DEPLOYMENT.md).
