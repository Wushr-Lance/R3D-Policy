# Real Robot Deployment Guide for R3D-Policy

This guide provides comprehensive instructions for deploying R3D-Policy on real robots, including environment setup, data processing, model training, and real-time inference.

> **Note**: For detailed information about the robot hardware setup, please refer to our paper.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Collection (LeRobot)](#2-data-collection-lerobot)
3. [Data Processing Pipeline](#3-data-processing-pipeline)
4. [Task Configuration](#4-task-configuration)
5. [Model Training](#5-model-training)
6. [Real Robot Deployment](#6-real-robot-deployment)
7. [Visualization Tools](#7-visualization-tools)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Environment Setup

### 1.1 Base Environment

First, set up the base R3D environment following the main [README.md](./README.md):

```bash
# Create conda environment
conda create -n r3d python=3.10
conda activate r3d

# Install PyTorch with CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install R3D package
cd R3D && pip install -e . && cd ..

# Install additional dependencies
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0
pip install dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4
pip install moviepy imageio av matplotlib termcolor
pip install timm sentence-transformers==3.2.1 huggingface_hub==0.23.2 open_clip_torch

# Install PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install CuRobo (for robot kinematics)
cd R3D/r3d/env/robotwin2/envs
git clone https://github.com/NVlabs/curobo.git
cd curobo && pip install -e . --no-build-isolation && cd ../../../../..
```

### 1.2 LeRobot Framework Installation

We use a modified LeRobot framework for robot teleoperation and data collection. Install it from our repository:

```bash
# Clone the modified LeRobot framework
git clone https://github.com/BoyiZhao/lerobot_R3D.git
cd lerobot_R3D

# Install the package
pip install -e .

# Set up USB serial port permissions (for Gello)
sudo chmod 666 /dev/ttyUSB*
# Or use the provided script:
bash init.bash
```

### 1.3 CDM (Camera Depth Models) Installation

CDM is essential for converting noisy real-world depth maps into simulation-like clean depth maps, significantly improving sim-to-real transfer.

```bash
# Clone and install the CDM inference library
git clone https://github.com/ByteDance-Seed/manip-as-in-sim-suite.git
cd manip-as-in-sim-suite/cdm
pip install -e .
cd ../..

# Download the checkpoint optimized for Intel RealSense D435
# From: https://huggingface.co/depth-anything/camera-depth-model-d435/blob/main/cdm_d435.ckpt
# Place it at: data/cdm_d435.ckpt
```

### 1.4 Hardware Dependencies

For real robot operation, ensure you have:

- **Robot**: xArm6 collaborative robot with Gello leader arm for teleoperation
- **Cameras**: Intel RealSense D435/D435i (2-3 units recommended)
- **Motors**: Dynamixel motors for Gello leader arm

Install hardware SDKs:
```bash
# Intel RealSense SDK
pip install pyrealsense2

# Dynamixel SDK (usually included with LeRobot)
pip install dynamixel-sdk
```

---

## 2. Data Collection (LeRobot)

### 2.1 Camera Setup

Configure your cameras in the LeRobot config. Our setup uses:
- `cam_left`: Left side camera (RealSense D435i)
- `cam_right`: Right side camera (RealSense D435i)
- `cam_arm`: Wrist-mounted camera (optional)

### 2.2 Teleoperation

Start teleoperation to verify robot control:

```bash
cd lerobot_R3D

# Teleoperate without recording (testing)
python lerobot/scripts/control_robot.py \
    --robot.type=gello \
    --robot.cameras='{}' \
    --control.type=teleoperate \
    --control.fps=30

# Teleoperate with cameras
python lerobot/scripts/control_robot.py \
    --robot.type=gello \
    --robot.cameras="{cam_left: {type: intelrealsense, serial_number: YOUR_SN1}, cam_right: {type: intelrealsense, serial_number: YOUR_SN2}}" \
    --control.type=teleoperate \
    --control.fps=30
```

### 2.3 Recording Demonstrations

Record demonstration episodes:

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=gello \
    --robot.cameras="{cam_left: {...}, cam_right: {...}}" \
    --control.type=record \
    --control.repo_id=your_dataset/task1 \
    --control.fps=30 \
    --control.num_episodes=50 \
    --control.warmup_time_s=3 \
    --control.episode_time_s=60 \
    --control.reset_time_s=5
```

Data will be saved in LeRobot format with:
- `data/chunk-*/episode_*.parquet` - Robot states and actions
- `images/observation.images.*/` - RGB images
- `images/observation.depths.*/` - Depth images (uint16)

---

## 3. Data Processing Pipeline

### 3.1 Overview

The data processing pipeline converts LeRobot format data to R3D training format:

```
LeRobot Data (RGB-D + Parquet)
    ↓
CDM Depth Optimization (optional but recommended)
    ↓
Point Cloud Generation (coordinate transformation to BASE frame)
    ↓
Zarr Dataset (training format)
    ↓
R3D Training
```

### 3.2 Step 1: CDM Depth Denoising

CDM significantly improves depth quality by reducing noise and filling holes:

```bash
cd R3D-Policy

# Process all episodes in a task directory
bash real_robot_tools/scripts/process_depths_cdm.sh \
    data/task1 \
    data/cdm_d435.ckpt

# This creates 'observation.depths_after_cdm.*' directories with processed depth maps
```

### 3.3 Step 2: Convert to Zarr Format

Convert the processed data to Zarr format with point clouds:

```bash
python real_robot_tools/convert_parquet_to_zarr.py \
    --input-dir data/task1 \
    --calib-file real_robot_tools/calib.json \
    --inst-file real_robot_tools/inst.json \
    --output-dir data/zarr \
    --task-name task1 \
    --num-episodes 50 \
    --num-points 8192 \
    --use-cdm-depth \
    --num-workers 4
```

**Key Parameters**:
- `--input-dir`: Root directory containing episode folders
- `--calib-file`: Camera extrinsic calibration (transformation matrices)
- `--inst-file`: Camera intrinsic parameters
- `--num-points`: Number of points in downsampled point cloud (8192 recommended)
- `--use-cdm-depth`: Use CDM-processed depth maps
- `--num-workers`: Number of parallel workers (4-8 recommended)

### 3.4 Step 3: Visualize Point Clouds

Verify the point cloud quality before training:

```bash
python real_robot_tools/visualize_zarr_pointcloud.py \
    --zarr-path data/zarr/task1/task1-50.zarr \
    --output-dir visualizations/task1 \
    --num-frames 5
```

This generates interactive HTML visualizations to check:
- Point cloud coverage
- Coordinate alignment
- Cropping bounds
- Color quality

---

## 4. Task Configuration

### 4.1 Creating Task Configs

Task configurations define dataset paths, observation shapes, and training parameters. Create a config file in `R3D/r3d/config/task/`:

Example `real_robot_task1.yaml`:

```yaml
defaults:
  - _self_

name: real_robot_task1
horizon: 16
n_obs_steps: 2
n_action_steps: 8
obs_as_global_cond: true

shape_meta:
  obs:
    point_cloud:
      shape: [8192, 6]  # N points × (xyz + rgb)
      type: spatial
    agent_pos:
      shape: [7]  # Robot joints + gripper
      type: low_dim
  action:
    shape: [7]  # Same as agent_pos

dataset:
  zarr_path: data/zarr/task1/task1-50.zarr
  val_ratio: 0.1
  max_train_episodes: null

env_runner:
  _target_: r3d.env_runner.base_runner.BaseEnvRunner
  n_train: 0  # No simulation evaluation for real robot
  n_test: 0
```

### 4.2 Key Configuration Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `horizon` | Prediction horizon | 16 |
| `n_obs_steps` | Observation history steps | 2 |
| `n_action_steps` | Action execution steps | 8 |
| `num_points` | Point cloud size | 8192 |
| `agent_pos shape` | Robot DOF | 7 (single-arm) or 14 (dual-arm) |

---

## 5. Model Training

### 5.1 Training Command

Use the provided training script:

```bash
# Single GPU training
bash scripts/train_real_robot.sh dp3_uni3d_pretrained_robotwin2 task1 exp_v1 0 0

# Multi-GPU training (DDP)
bash scripts/train_real_robot.sh dp3_uni3d_pretrained_robotwin2 task1 exp_v1 0 0,1,2,3
```

**Arguments**:
1. `dp3_uni3d_pretrained_robotwin2`: Algorithm config (recommended with pretrained Uni3D encoder)
2. `task1`: Task name (matches config file)
3. `exp_v1`: Experiment label
4. `0`: Random seed
5. `0` or `0,1,2,3`: GPU ID(s)

### 5.2 Algorithm Configs

Available algorithm configurations:

| Config | Description | Recommended |
|--------|-------------|-------------|
| `dp3_uni3d_pretrained_robotwin2` | DP3 with pretrained Uni3D encoder | Yes (best performance) |
| `dp3_robotwin2` | Baseline DP3 | For comparison |
| `dp3_uni3d_scratch_robotwin2` | DP3 with Uni3D from scratch | Requires more data |

### 5.3 Monitoring Training

Training outputs are saved to:
```
R3D/r3d/data/outputs/real_robot_<task>-<config>-<label>_seed<seed>/
├── checkpoints/
│   ├── 100.ckpt
│   ├── 200.ckpt
│   └── ...
├── logs/
└── config.yaml
```

Use Weights & Biases for real-time monitoring (enabled by default).

---

## 6. Real Robot Deployment

### 6.1 Perceptual Chain Debug (Zarr Replay)

Before deploying the policy, verify the perceptual pipeline:

```bash
conda activate r3d

export REPLAY_ZARR="data/zarr/task1/task1-50.zarr"
export REPLAY_EPISODE="0"

python real_robot_tools/deploy/control_robot_dp3.py \
    --robot.type=gello \
    --control.type=record \
    --control.repo_id=test/debug \
    --control.fps=30 \
    --control.num_episodes=1 \
    --control.warmup_time_s=1 \
    --control.episode_time_s=50 \
    --control.reset_time_s=1
```

This replays recorded data through the real-time pipeline to verify point cloud generation matches training.

### 6.2 Policy Execution

Deploy the trained policy:

```bash
# Set checkpoint path
export DP3_CHECKPOINT="R3D/r3d/data/outputs/real_robot_task1-dp3_uni3d_pretrained_robotwin2-exp_v1_seed0/checkpoints/1000.ckpt"

# Run policy
python real_robot_tools/deploy/control_robot_dp3.py \
    --robot.type=gello \
    --control.type=record \
    --control.repo_id=eval/task1 \
    --control.single_task=true \
    --control.fps=30 \
    --control.num_episodes=10 \
    --control.warmup_time_s=5 \
    --control.episode_time_s=60 \
    --control.reset_time_s=5
```

### 6.3 Deployment Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `control.fps` | Control frequency | 30 Hz |
| `control.warmup_time_s` | Time before episode starts | 5s |
| `control.episode_time_s` | Maximum episode duration | 60s |
| `control.reset_time_s` | Time for robot reset | 5s |

---

## 7. Visualization Tools

### 7.1 Point Cloud Visualization

Generate interactive HTML visualizations:

```bash
python real_robot_tools/visualize_zarr_pointcloud.py \
    --zarr-path data/zarr/task1/task1-50.zarr \
    --output-dir visualizations/task1 \
    --num-frames 5
```

### 7.2 Dynamic Video Rendering

Create videos showing point cloud sequences:

```bash
python real_robot_tools/render_zarr_dynamic_video.py \
    --zarr-path data/zarr/task1/task1-50.zarr \
    --output-dir visualizations/videos \
    --episode-idx 0 \
    --fps 30
```

### 7.3 Batch Video Processing

Process multiple zarr files:

```bash
bash real_robot_tools/scripts/batch_render_zarr_dynamic_videos.sh \
    --source task1=data/zarr/task1/task1-50.zarr \
    --source task2=data/zarr/task2/task2-50.zarr \
    --output-dir visualizations/all_tasks \
    --episode-idx 0 \
    --fps 30
```

---

## 8. Troubleshooting

### Common Issues

**1. Sparse or noisy point clouds**
- Enable CDM depth processing
- Check camera calibration files
- Adjust cropping ranges in `inst.json`

**2. Point cloud coordinate mismatch**
- Verify `calib.json` transformation matrices
- Check that cameras are properly calibrated
- Visualize with `visualize_zarr_pointcloud.py`

**3. Slow inference**
- Use GPU for CDM processing
- Reduce `num_points` if necessary
- Store data on SSD

**4. Robot communication errors**
- Check USB permissions: `sudo chmod 666 /dev/ttyUSB*`
- Verify Dynamixel motor connections
- Check xArm network connection

**5. Multi-GPU training issues**
- Script auto-selects free ports
- Check firewall settings
- Ensure all GPUs are visible: `nvidia-smi`

### Camera Calibration

If you need to recalibrate cameras, update these files:

- `real_robot_tools/calib.json`: Extrinsic matrices (T_camera_to_base)
- `real_robot_tools/inst.json`: Intrinsic matrices and ROI cropping ranges

Default ROI cropping: `x: [-0.5, 0.82]`, `y: [-0.7, 0.8]`

---

## Core Files Reference

| File | Purpose |
|------|---------|
| `real_robot_tools/deploy/control_robot_dp3.py` | Main deployment script |
| `real_robot_tools/deploy/control_utils.py` | Control utilities |
| `real_robot_tools/convert_parquet_to_zarr.py` | Data conversion |
| `real_robot_tools/process_depth_with_cdm.py` | CDM depth processing |
| `real_robot_tools/realtime_pointcloud.py` | Real-time point cloud generation |
| `real_robot_tools/visualize_zarr_pointcloud.py` | Point cloud visualization |
| `scripts/train_real_robot.sh` | Training script |
| `R3D/r3d/config/task/real_robot_task*.yaml` | Task configurations |

---

## Citation

If you use this code for real robot deployment, please cite our paper:

```bibtex
@article{r3d2025,
  title={R3D-Policy: Revisiting 3D Policy Learning},
  author={...},
  journal={...},
  year={2025}
}
```
