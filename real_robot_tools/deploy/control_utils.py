# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# Utilities
########################################################################################

import pickle
import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
import h5py
import cv2
import torch
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method
from lerobot.common.datasets.my_utils import get_point_cloud

# Add realtime pointcloud path
import sys
import pathlib
ROOT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "real_robot_to_3dpolicy_tools"))

try:
    from realtime_pointcloud import create_realtime_generator
    import numpy as np
    REALTIME_PC_AVAILABLE = True
except ImportError:
    print("Warning: realtime_pointcloud module not available")
    REALTIME_PC_AVAILABLE = False


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def apply_action_scaling(action_chunk, joint_scales=None, global_scale=1.0, add_variation=True):
    """
    Apply scaling to action chunk - solve j5, j6 fixed issue.

    Args:
        action_chunk: raw action chunk [n_actions, 7]
        joint_scales: scaling factor for each joint [j1,j2,j3,j4,j5,j6,gripper]
        global_scale: global scaling factor
        add_variation: whether to add artificial variation to fixed joints

    Returns:
        scaled_action_chunk: scaled action chunk
    """
    if action_chunk is None:
        return None

    action_chunk = action_chunk.copy()

    # First add artificial variation to fixed joints
    if add_variation:
        joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'gripper']

        for i, joint in enumerate(joint_names):
            if i < action_chunk.shape[1]:
                joint_actions = action_chunk[:, i]
                std_val = np.std(joint_actions)

                # If joint is completely fixed (std close to 0), add small variation
                if std_val < 1e-6:
                    n_actions = action_chunk.shape[0]

                    # Set different variation patterns for different joints
                    if joint == 'j5':
                        # Add sinusoidal variation to j5
                        variation = np.sin(np.linspace(0, 2*np.pi, n_actions)) * 0.05
                        print(f"[Action Scaling] Adding sinusoidal variation to {joint}: +/-0.05")
                    elif joint == 'j6':
                        # Add sawtooth variation to j6
                        variation = np.linspace(-0.03, 0.03, n_actions)
                        print(f"[Action Scaling] Adding linear variation to {joint}: +/-0.03")
                    else:
                        # Add small random variation to other fixed joints
                        variation = np.random.normal(0, 0.01, n_actions)
                        print(f"[Action Scaling] Adding random variation to {joint}: std=0.01")

                    action_chunk[:, i] += variation

    # Apply joint-specific scaling
    if joint_scales is not None:
        joint_scales = np.array(joint_scales)
        if len(joint_scales) == action_chunk.shape[1]:
            action_chunk = action_chunk * joint_scales[np.newaxis, :]
            print(f"[Action Scaling] Applied joint scales: {joint_scales}")
        else:
            print(f"[Action Scaling] Warning: joint_scales length {len(joint_scales)} != action dim {action_chunk.shape[1]}")

    # Apply global scaling
    if global_scale != 1.0:
        action_chunk = action_chunk * global_scale
        print(f"[Action Scaling] Applied global scale: {global_scale}")

    # Show scaling effect
    joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'gripper']
    print("[Action Scaling] Final action ranges:")
    for i, joint in enumerate(joint_names):
        if i < action_chunk.shape[1]:
            joint_actions = action_chunk[:, i]
            min_val, max_val = np.min(joint_actions), np.max(joint_actions)
            std_val = np.std(joint_actions)
            print(f"  {joint}: [{min_val:.6f}, {max_val:.6f}] std={std_val:.6f}")

    return action_chunk


def predict_action_dp3(observation, policy, policy_meta, obs_history, generator=None):
    """
    DP3 Policy prediction - converts observation to point cloud and predicts action.
    
    Args:
        observation: Robot observation dict (contains RGB, Depth, State)
        policy: DP3 policy model with dp3_meta attached
        policy_meta: Dict with shape_meta, n_obs_steps, n_action_steps, device
        obs_history: collections.deque storing recent observations (point_cloud, agent_pos)
        generator: RealtimePointCloudGenerator (will be created if None)
        
    Returns:
        action_chunk: numpy array (n_action_steps, action_dim) - full action chunk
        generator: RealtimePointCloudGenerator (for reuse)
        obs_history: Updated observation history deque
    """
    if not REALTIME_PC_AVAILABLE:
        raise RuntimeError("realtime_pointcloud module not available! Check imports.")
    
    import pathlib
    import torch
    import numpy as np
    
    # Track if this is the first call (generator is None means first call)
    is_first_call = (generator is None)
    
    # 1. Extract RGB and Depth images from observation
    rgb_dict = {}
    depth_dict = {}
    
    for key in observation:
        if 'observation.images.' in key:
            cam_name = key.split('.')[-1]  # e.g., 'cam_left', 'cam_right', 'cam_arm'
            img_tensor = observation[key]
            
            # Debug: print shape on first call
            if generator is None:
                print(f"[DP3 Debug] {key}: shape={img_tensor.shape}, size={img_tensor.numel()}")
            
            # Handle different tensor shapes
            # LeRobot may provide (T, H, W, C), (C, H, W), or (H, W, C)
            if img_tensor.ndim == 4:
                # (T, H, W, C) - take the last frame
                img_np = img_tensor[-1].numpy()  # (H, W, C)
            elif img_tensor.ndim == 3 and img_tensor.shape[0] == 3:
                # (C, H, W) -> (H, W, C)
                img_np = img_tensor.permute(1, 2, 0).numpy()
            else:
                # Already (H, W, C) or other format
                img_np = img_tensor.numpy()
            
            # Convert to uint8 if needed (LeRobot may provide float32 [0,1])
            if is_first_call:
                print(f"[DP3 DEBUG] {cam_name} RGB before conversion: dtype={img_np.dtype}, range=[{img_np.min():.3f}, {img_np.max():.3f}]")
            
            if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                img_np = (img_np * 255).astype(np.uint8)
            elif img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)
            
            if is_first_call:
                print(f"[DP3 DEBUG] {cam_name} RGB after conversion: dtype={img_np.dtype}, range=[{img_np.min()}, {img_np.max()}]")
            
            rgb_dict[cam_name] = img_np  # (H, W, 3) uint8
            
        elif 'observation.depths.' in key:
            cam_name = key.split('.')[-1]
            depth_tensor = observation[key]
            
            # Debug: print shape on first call
            if generator is None:
                print(f"[DP3 Debug] {key}: shape={depth_tensor.shape}, size={depth_tensor.numel()}")
            
            # Handle different tensor shapes
            if depth_tensor.ndim == 3:
                # (T, H, W) - take the last frame
                depth_np = depth_tensor[-1].numpy()  # (H, W)
            else:
                # Already (H, W)
                depth_np = depth_tensor.numpy()
            
            # Convert to uint16 if needed (ensure proper depth format)
            if is_first_call:
                print(f"[DP3 DEBUG] {cam_name} Depth before conversion: dtype={depth_np.dtype}, range=[{depth_np.min():.3f}, {depth_np.max():.3f}]")
            
            if depth_np.dtype != np.uint16:
                depth_np = depth_np.astype(np.uint16)
            
            if is_first_call:
                print(f"[DP3 DEBUG] {cam_name} Depth after conversion: dtype={depth_np.dtype}, range=[{depth_np.min()}, {depth_np.max()}]")
            
            depth_dict[cam_name] = depth_np  # (H, W) uint16
    
    # 2. Get agent state (joint positions) and end-effector pose
    full_state = observation['observation.state'].numpy()  # Full state vector (joint angles)
    
    # Fix: correctly extract and convert end-effector pose
    # observation.ee_pos is the actual end-effector pose, not observation.state!
    # Important: ee_pos needs to be converted the same way as in the convert script!
    if 'observation.ee_pos' in observation:
        ee_pos_raw = observation['observation.ee_pos'].numpy()
        # ee_pos_raw should be (n_obs_steps, 6) or (6,), take the latest one
        if ee_pos_raw.ndim == 2:
            ee_pos_raw_frame = ee_pos_raw[-1]  # Latest timestep
        else:
            ee_pos_raw_frame = ee_pos_raw
        
        # Convert same as in convert script (convert_real_robot_data.py line 458-460)
        # LeRobot ee_pos format: [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
        # realtime_pointcloud expected format is also: [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
        # But convert script did conversion before use, we stay consistent
        ee_pos_data = ee_pos_raw_frame.copy()
        
        if is_first_call:
            print(f"[DP3 DEBUG] [OK] Extracted end-effector pose from observation.ee_pos (LeRobot raw format):")
            print(f"[DP3 DEBUG]    Pos (mm): X={ee_pos_data[0]:.1f}, Y={ee_pos_data[1]:.1f}, Z={ee_pos_data[2]:.1f}")
            print(f"[DP3 DEBUG]    Ori (deg): Roll={ee_pos_data[3]:.1f}, Pitch={ee_pos_data[4]:.1f}, Yaw={ee_pos_data[5]:.1f}")
            print(f"[DP3 DEBUG]    This format will be passed to realtime_pointcloud, which handles mm->m and deg->rad internally")
            
            # Calculate converted values (same as inside realtime_pointcloud)
            import scipy.spatial.transform as sst
            xyz_m = ee_pos_data[:3] / 1000.0
            rpy_rad = np.deg2rad(ee_pos_data[3:6])
            print(f"[DP3 DEBUG]    Converted Pos (m): X={xyz_m[0]:.3f}, Y={xyz_m[1]:.3f}, Z={xyz_m[2]:.3f}")
            print(f"[DP3 DEBUG]    Converted Ori (rad): Roll={rpy_rad[0]:.3f}, Pitch={rpy_rad[1]:.3f}, Yaw={rpy_rad[2]:.3f}")
    else:
        # If no ee_pos, head camera cannot be converted correctly
        ee_pos_data = np.zeros(6)
        if is_first_call:
            print(f"[DP3 DEBUG] [WARN] No ee_pos in observation, head camera coordinate frame might be wrong!")
    
    # **CRITICAL DEBUGGING**: Analyze the actual input data format
    if is_first_call:
        print(f"[DP3 DEBUG] **Full observation keys analysis**:")
        print(f"[DP3 DEBUG] observation.keys(): {list(observation.keys())}")
        for key in observation.keys():
            if hasattr(observation[key], 'shape'):
                print(f"[DP3 DEBUG]   {key}: shape={observation[key].shape}, dtype={observation[key].dtype}")
            else:
                print(f"[DP3 DEBUG]   {key}: type={type(observation[key])}")
    def analyze_lerobot_state_format(observation):
        """Analyze LeRobot observation.state format and find the correct joint angle extraction method"""
        full_state = observation['observation.state'].numpy()

        print(f"[LeRobot State Analysis] **Detailed format analysis**")
        print(f"  shape: {full_state.shape}")
        print(f"  dtype: {full_state.dtype}")
        print(f"  Full data: {full_state.flatten()}")

        # Training data reference range (real range analyzed from zarr)
        training_ranges = {
            0: (0.0, 0.435),      # j1: training data range [0.000000, 0.435088]
            1: (-0.262, 0.613),   # j2: training data range [-0.261799, 0.613433]
            2: (-1.899, -0.960),  # j3: training data range [-1.898933, -0.959931]
            3: (-3.124, -3.103),  # j4: training data range [-3.124139, -3.103388]
            4: (-1.257, -1.091),  # j5: training data range [-1.256636, -1.090747] <- Key
            5: (-0.035, 0.152),   # j6: training data range [-0.034907, 0.152128] <- Key
            6: (0.002, 0.141)     # gripper: training data range [0.001745, 0.140674]
        }

        def check_range_match(data, name):
            """Check if data matches training data range"""
            score = 0
            print(f"  {name}:")
            for i in range(min(7, len(data))):
                val = data[i]
                min_range, max_range = training_ranges[i]
                in_range = min_range <= val <= max_range
                if in_range:
                    score += 1
                status = '[PASS]' if in_range else '[FAIL]'
                print(f"    j{i}: {val:.6f} {status} training range[{min_range:.3f}, {max_range:.3f}]")
            print(f"    Match score: {score}/7")
            return score

        # Extract flat data
        if len(full_state.shape) == 2:  # (1, N)
            state_flat = full_state[0]
        else:  # (N,)
            state_flat = full_state

        print(f"  Flattened data length: {len(state_flat)}")
        print(f"  Full flattened data: {state_flat}")

        best_method = None
        best_score = -1

        # Method 1: First 7 dimensions
        if len(state_flat) >= 7:
            front_7 = state_flat[:7]
            score1 = check_range_match(front_7, "Method 1 - First 7 dims")
            if score1 > best_score:
                best_score = score1
                best_method = ("First 7 dims", front_7)

        # Method 2: Last 7 dimensions
        if len(state_flat) >= 7:
            back_7 = state_flat[-7:]
            score2 = check_range_match(back_7, "Method 2 - Last 7 dims")
            if score2 > best_score:
                best_score = score2
                best_method = ("Last 7 dims", back_7)

        # Method 3: Dimensions 7-13 (if 14+ dims)
        if len(state_flat) >= 14:
            mid_7 = state_flat[7:14]
            score3 = check_range_match(mid_7, "Method 3 - Middle 7 dims (7-13)")
            if score3 > best_score:
                best_score = score3
                best_method = ("Middle 7 dims", mid_7)

        # Method 4: Try other possible positions
        if len(state_flat) >= 21:
            for start_idx in [14, 15, 16]:
                if start_idx + 7 <= len(state_flat):
                    candidate = state_flat[start_idx:start_idx+7]
                    score = check_range_match(candidate, f"Method 4 - {start_idx} to {start_idx+6} dims")
                    if score > best_score:
                        best_score = score
                        best_method = (f"{start_idx} to {start_idx+6} dims", candidate)

        print(f"\n**Best extraction method analysis**:")
        if best_method:
            method_name, best_data = best_method
            print(f"  Best method: {method_name} (score: {best_score}/7)")
            print(f"  Extracted data: {best_data}")

            if best_score >= 5:  # Reliable if at least 5/7 match
                print(f"  [OK] Reliable extraction method found!")
                return best_data
            else:
                print(f"  [WARN] Low match score, might need other conversion")
        else:
            print(f"  [ERROR] No matching extraction method found")

        # If no good match, show more analysis info
        print(f"\n**Detailed format speculation**:")
        if len(state_flat) >= 14:
            print(f"  Assumed format: [joint_pos 7 + joint_vel 7 + ...]")
            print(f"    Possible joint positions: {state_flat[:7]}")
            print(f"    Possible joint velocities: {state_flat[7:14]}")

        if len(state_flat) >= 21:
            print(f"  Assumed format: [EE pose 6 + joint_pos 7 + ...]")
            print(f"    Possible EE pose: {state_flat[:6]}")
            print(f"    Possible joint positions: {state_flat[6:13]}")

        # Fallback to first 7 dims by default, with warning
        fallback_data = state_flat[:7] if len(state_flat) >= 7 else np.zeros(7)
        print(f"  [FALLBACK] Using first 7 dims {fallback_data}")
        return fallback_data
    
    # Detailed analysis on first call
    if is_first_call:
        extracted_state = analyze_lerobot_state_format(observation)
        print(f"[DP3 DEBUG] 📤 **LeRobot raw state**: {extracted_state}")
    else:
        # Non-first call, use fast extraction (may need adjustment based on first call analysis)
        if full_state.shape[-1] >= 7:
            extracted_state = full_state[0, :7]
        else:
            extracted_state = np.pad(full_state[0], (0, 7 - full_state.shape[-1]))
    
    # Correct data processing flow:
    # LeRobot provides state in degrees -> convert to training format -> model inference -> convert back to control format
    # print(f"[DP3 Data Processing] 🔧 **Correct data flow: LeRobot degrees -> training format -> inference -> control format**")
    # print(f"[DP3 Data Processing] LeRobot raw degree data: {extracted_state}")
    
    # Convert to training format (degrees to radians etc.)
    state = extracted_state
    
    # print(f"[DP3 Data Processing] 📤 **Will convert to training format for model inference**")
    # print(f"[DP3 Data Processing] ℹ️ Will convert back to LeRobot control format after inference")
    
    # ee_pos_data already extracted correctly (from observation.ee_pos)
    
    # 3. Initialize generator if needed (only once per episode)
    if generator is None:
        print("[DP3] Creating realtime pointcloud generator...")
        calib_path = pathlib.Path(__file__).parent / 'real_robot_to_3dpolicy_tools' / 'calib.json'
        inst_path = pathlib.Path(__file__).parent / 'real_robot_to_3dpolicy_tools' / 'inst.json'
        generator = create_realtime_generator(
            calib_path=str(calib_path),
            inst_path=str(inst_path),
            device=policy_meta['device'],
            crop_x_range=(-0.5, 0.82),  # Match training data cropping
            crop_y_range=(-0.7, 0.8),
            crop_z_range=(-np.inf, np.inf)
        )
        print("[DP3] Generator initialized")
    
    # 4. Generate point cloud: (N, 6) with xyz + rgb
    pointcloud = generator.generate(
        rgb_dict, 
        depth_dict, 
        ee_pos_data, 
        num_points=8192  # Match training
    )
    
    # Save point cloud and visualize (on first call)
    if is_first_call:
        print(f"\n**Saving point cloud for debugging**")
        try:
            # Save raw numpy array
            pointcloud_save_path = "/tmp/dp3_deployment_pointcloud.npy"
            np.save(pointcloud_save_path, pointcloud)
            print(f"  [OK] Saved point cloud array to: {pointcloud_save_path}")
            print(f"     Shape: {pointcloud.shape}")
            print(f"     XYZ Range: X[{pointcloud[:, 0].min():.3f}, {pointcloud[:, 0].max():.3f}], "
                  f"Y[{pointcloud[:, 1].min():.3f}, {pointcloud[:, 1].max():.3f}], "
                  f"Z[{pointcloud[:, 2].min():.3f}, {pointcloud[:, 2].max():.3f}]")

            # Generate HTML visualization
            from real_robot_to_3dpolicy_tools.realtime_pointcloud import visualize_pointcloud_plotly
            html_save_path = "/tmp/dp3_deployment_pointcloud.html"
            visualize_pointcloud_plotly(
                pointcloud,
                output_path=html_save_path,
                title=f"DP3 Deployment Point Cloud (frame 0)"
            )
            print(f"  [OK] Generated HTML visualization: {html_save_path}")
            print(f"  Open in browser to check point cloud coordinate frame")
        except Exception as e:
            print(f"  [WARN] Failed to save point cloud: {e}")
    
    # 5. Determine whether to use color based on policy configuration
    # Check if policy uses color by looking at the expected input shape
    use_pc_color = policy_meta['shape_meta'].obs.point_cloud.shape[1] == 6
    
    if use_pc_color:
        # Keep xyz + rgb (N, 6)
        pc_data = pointcloud  # (N, 6)
    else:
        # Extract xyz only (N, 3)
        pc_data = pointcloud[:, :3]  # (N, 3)
    
    # 6. Use temporal state data directly - no need to flatten or repeat
    # observation['observation.state'] itself is full n_obs_steps temporal data (T, D)
    n_obs_steps = policy_meta['n_obs_steps']
    pc_tensor = torch.from_numpy(pc_data).float()
    
    # Analyze full temporal state
    if is_first_call:
        print(f"\n**Temporal State Analysis** (n_obs_steps={n_obs_steps}):")
        print(f"  observation['observation.state'].shape: {full_state.shape}")
        if full_state.ndim == 2 and full_state.shape[0] == n_obs_steps:
            print(f"  [OK] Perfect match! Temporal dimension {full_state.shape[0]} == n_obs_steps {n_obs_steps}")
            print(f"  State dimension: {full_state.shape[1]}")
            for t in range(full_state.shape[0]):
                print(f"  Timestep {t}: {full_state[t]}")
        else:
            print(f"  [WARN] Dimension mismatch: {full_state.shape}")
    
    # Build obs_history directly from temporal data
    from collections import deque
    obs_history = deque(maxlen=n_obs_steps)
    
    if full_state.ndim == 2 and full_state.shape[0] == n_obs_steps:
        # Perfect case: use full temporal state data directly, convert to training format
        print(f"[DP3 Debug] [OK] Using full temporal state data, converting to training format") if is_first_call else None
        
        # Convert state to training data format for each timestep
        for t in range(n_obs_steps):
            raw_state = full_state[t]  # (D,) original joint state (degrees)
            
            # Convert to training data format - following convert script logic
            # convert_real_robot_data.py line 530:
            # state_rad = np.deg2rad(state)
            # vector_ds = np.concatenate([state_rad[:6], [state_rad[6] / 100.0]])
            converted_state = np.zeros(7)
            # converted_state[:6] = np.deg2rad(raw_state[:6])  # first 6 joints: deg -> rad
            # converted_state[6] = np.deg2rad(raw_state[6]) / 100.0  # gripper: deg -> rad -> normalized
            converted_state = raw_state
            
            # Build observation (use training format)
            agent_pos_tensor = torch.from_numpy(converted_state[:7]).float()
            obs = {'point_cloud': pc_tensor, 'agent_pos': agent_pos_tensor}
            obs_history.append(obs)
            
            if is_first_call:
                print(f"  Timestep {t}: raw degrees={raw_state[:4]} -> training format(rad)={converted_state[:4]}")
                if len(raw_state) > 4:
                    print(f"               j5: {raw_state[4]:.3f}°->{converted_state[4]:.3f}rad, j6: {raw_state[5]:.3f}°->{converted_state[5]:.3f}rad")
    
    else:
        # Fallback: handle non-standard format
        print(f"[DP3 Debug] [WARN] Using fallback state processing, converting to training format") if is_first_call else None
        
        # Use analyze_lerobot_state_format logic
        if is_first_call:
            extracted_state = analyze_lerobot_state_format(observation)
        else:
            # Fast extraction (avoid repeated analysis)
            if full_state.shape[-1] >= 7:
                extracted_state = full_state[0, :7] if full_state.ndim == 2 else full_state[:7]
            else:
                extracted_state = np.pad(full_state.flatten(), (0, 7 - full_state.size))[:7]
        
        # Convert to training data format - following convert script logic
        converted_state = np.zeros(7)
        converted_state[:6] = np.deg2rad(extracted_state[:6])  # first 6 joints: deg -> rad
        converted_state[6] = np.deg2rad(extracted_state[6]) / 100.0  # gripper: deg -> rad -> normalized
        
        # Fill history with converted state
        agent_pos_tensor = torch.from_numpy(converted_state[:7]).float()
        current_obs = {'point_cloud': pc_tensor, 'agent_pos': agent_pos_tensor}
        
        for _ in range(n_obs_steps):
            obs_history.append(current_obs.copy())
    
    # 9. Build model input
    point_clouds = torch.stack([obs['point_cloud'] for obs in obs_history], dim=0)  # (T, N, C)
    agent_poses = torch.stack([obs['agent_pos'] for obs in obs_history], dim=0)  # (T, D)
    
    # Add batch dimension and move to device
    obs_dict_input = {
        'point_cloud': point_clouds.unsqueeze(0).to(policy_meta['device']),  # (1, T, N, C)
        'agent_pos': agent_poses.unsqueeze(0).to(policy_meta['device'])  # (1, T, D) - Keep agent_pos key for normalizer
    }
    
    # CRITICAL INPUT RANGE ANALYSIS: Check if inference input matches training data ranges
    if is_first_call:
        print(f"\n**CRITICAL: Inference Input Range Analysis**")
        print(f"===============================================")

        # 1. Analyze point cloud input range
        pc_data = obs_dict_input['point_cloud'][0, -1]  # Latest timestep (N, C)
        print(f"Point Cloud Analysis:")
        print(f"  shape: {pc_data.shape}")
        if pc_data.shape[1] >= 3:
            xyz = pc_data[:, :3]  # (N, 3)
            print(f"  XYZ ranges:")
            print(f"    X: [{xyz[:, 0].min():.6f}, {xyz[:, 0].max():.6f}] mean={xyz[:, 0].mean():.6f}")
            print(f"    Y: [{xyz[:, 1].min():.6f}, {xyz[:, 1].max():.6f}] mean={xyz[:, 1].mean():.6f}")
            print(f"    Z: [{xyz[:, 2].min():.6f}, {xyz[:, 2].max():.6f}] mean={xyz[:, 2].mean():.6f}")
        if pc_data.shape[1] >= 6:
            rgb = pc_data[:, 3:6]  # (N, 3)
            print(f"  RGB ranges:")
            print(f"    R: [{rgb[:, 0].min():.6f}, {rgb[:, 0].max():.6f}] mean={rgb[:, 0].mean():.6f}")
            print(f"    G: [{rgb[:, 1].min():.6f}, {rgb[:, 1].max():.6f}] mean={rgb[:, 1].mean():.6f}")
            print(f"    B: [{rgb[:, 2].min():.6f}, {rgb[:, 2].max():.6f}] mean={rgb[:, 2].mean():.6f}")

        # 2. Analyze agent_pos input range and compare with training data
        agent_pos_data = obs_dict_input['agent_pos'][0, -1]  # Latest timestep (D,)
        print(f"\nAgent Position Key Analysis:")
        print(f"  shape: {agent_pos_data.shape}")
        print(f"  Inference input values: {agent_pos_data.detach().cpu().numpy()}")
        
        # Training data real range (analyzed from zarr)
        training_ranges_real = {
            0: (0.000000, 0.435088),   # j1 
            1: (-0.261799, 0.613433),  # j2 
            2: (-1.898933, -0.959931), # j3 
            3: (-3.124139, -3.103388), # j4 
            4: (-1.256636, -1.090747), # j5 <- Key joint
            5: (-0.034907, 0.152128),  # j6 <- Key joint  
            6: (0.001745, 0.140674)    # gripper
        }
        
        joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'gripper']
        print(f"\nPer-joint Range Comparison Analysis:")
        range_match_score = 0

        for i in range(min(7, agent_pos_data.shape[0])):
            joint_name = joint_names[i]
            input_val = agent_pos_data[i].item()
            train_min, train_max = training_ranges_real[i]
            in_range = train_min <= input_val <= train_max

            if in_range:
                range_match_score += 1
                status = '[PASS]'
            else:
                status = '[FAIL]'

            print(f"  {joint_name}: inference={input_val:.6f} {status} training range[{train_min:.6f}, {train_max:.6f}]")

            # Focus on j5/j6
            if joint_name in ['j5', 'j6']:
                if in_range:
                    print(f"    [OK] {joint_name} is in correct range!")
                else:
                    distance_to_range = min(abs(input_val - train_min), abs(input_val - train_max))
                    print(f"    [WARN] {joint_name} is out of training range! Nearest distance={distance_to_range:.6f}")

        print(f"\nRange match score: {range_match_score}/7")
        if range_match_score < 5:
            print(f"[ERROR] Inference input range seriously mismatches training data! This could be root cause of fixed j5/j6!")
        elif range_match_score < 7:
            print(f"[WARN] Inference input range partially mismatches training data")
        else:
            print(f"[OK] Inference input range completely matches training data")

        # 3. Check normalizer expected range
        if hasattr(policy, 'normalizer') and policy.normalizer is not None:
            try:
                print(f"\nNormalizer Param Analysis:")
                normalizer = policy.normalizer
                if hasattr(normalizer, 'params_dict'):
                    agent_pos_stats = getattr(normalizer.params_dict, 'agent_pos', None)
                    if agent_pos_stats is not None and hasattr(agent_pos_stats, 'input_stats'):
                        norm_mean = agent_pos_stats.input_stats.mean[:7]
                        norm_std = agent_pos_stats.input_stats.std[:7]
                        print(f"  Normalizer Mean: {norm_mean.detach().cpu().numpy()}")
                        print(f"  Normalizer Std:  {norm_std.detach().cpu().numpy()}")

                        # Calculate normalized values
                        normalized_input = (agent_pos_data[:7] - norm_mean) / norm_std
                        print(f"  Normalized input: {normalized_input.detach().cpu().numpy()}")

                        # Check for reasonable normalization range (usually -3 to 3)
                        extreme_values = (normalized_input.abs() > 3).any()
                        if extreme_values:
                            print(f"  [WARN] Normalized input has extreme values (>3 std), might affect inference!")
                        else:
                            print(f"  [OK] Normalized values are in normal range")
            except Exception as e:
                print(f"  [ERROR] Could not get normalizer info: {e}")
        
        print(f"===============================================\n")
    
    # CRITICAL FIX: Check model configuration and adjust data format accordingly
    # Different DP3 models may have been trained with different data formats
    # We need to detect the model's expected format and adjust our input
    
    # Try to detect if model uses extract_global_feature=True
    extract_global_feature = getattr(policy, 'pc_encoder_extract_global_feature', None)
    if extract_global_feature is None and hasattr(policy, 'obs_encoder'):
        extract_global_feature = getattr(policy.obs_encoder, 'pc_encoder_extract_global_feature', None)
    
    if is_first_call and extract_global_feature is not None:
        print(f"[DP3 Debug] Detected model with extract_global_feature={extract_global_feature}")
        if extract_global_feature:
            print(f"[DP3 Debug] Using global feature extraction mode - will handle state format accordingly")
    
    # 10. Policy inference 
    print(f"[DP3 Debug] Before policy prediction:")
    print(f"[DP3 Debug] point_cloud shape: {obs_dict_input['point_cloud'].shape}")
    print(f"[DP3 Debug] agent_pos shape: {obs_dict_input['agent_pos'].shape}")  # Keep agent_pos key
    
    # DEBUG: Print policy configuration details
    if is_first_call:
        print(f"[DP3 Debug] Policy device: {policy.device}")
        print(f"[DP3 Debug] Policy num_inference_steps: {getattr(policy, 'num_inference_steps', 'NOT_SET')}")
        # Check if using EMA
        ema_status = "EMA" if hasattr(policy, 'ema') and policy.ema is not None else "Standard"
        print(f"[DP3 Debug] Model type: {ema_status}")
        
        # Print normalizer stats for agent_pos (first few values)
        if hasattr(policy, 'normalizer') and policy.normalizer is not None:
            try:
                normalizer = policy.normalizer
                if hasattr(normalizer, 'params_dict') or hasattr(normalizer, 'get_input_stats'):
                    print(f"[DP3 Debug] Normalizer available")
                    # Try different ways to access normalizer stats
                    if hasattr(normalizer, 'params_dict'):
                        agent_pos_stats = getattr(normalizer.params_dict, 'agent_pos', None)
                        if agent_pos_stats is not None:
                            if hasattr(agent_pos_stats, 'input_stats'):
                                mean = agent_pos_stats.input_stats.mean
                                std = agent_pos_stats.input_stats.std
                                print(f"[DP3 Debug] agent_pos normalizer mean[:7]: {mean[:7] if len(mean) >= 7 else mean}")
                                print(f"[DP3 Debug] agent_pos normalizer std[:7]: {std[:7] if len(std) >= 7 else std}")
                                # Specifically check j5/j6 normalizer values (indices 4,5)
                                print(f"[DP3 Debug] j5 (idx=4) normalizer mean={mean[4]:.6f}, std={std[4]:.6f}")
                                print(f"[DP3 Debug] j6 (idx=5) normalizer mean={mean[5]:.6f}, std={std[5]:.6f}")
            except Exception as e:
                print(f"[DP3 Debug] Could not access normalizer stats: {e}")
                    
        # Print input agent_pos values (raw and normalized)
        agent_pos_raw = obs_dict_input['agent_pos'][0, -1, :]  # Last timestep, batch=0
        print(f"[DP3 Debug] Raw agent_pos (last timestep): {agent_pos_raw}")
    
    with torch.no_grad():
        action = policy.predict_action(obs_dict_input)  # (1, n_action_steps, action_dim)
        # print("action in inference time",action)
        print("success")
        if is_first_call:
            print(f"[DP3 Debug] [OK] Policy prediction successful!")
            print(f"[DP3 Debug] Action keys: {list(action.keys())}")
            print(f"[DP3 Debug] Action shape: {action['action'].shape}")
    
    # DEBUG: Print action statistics before extraction
    action_tensor = action['action']  # (1, n_action_steps, action_dim)
    if is_first_call:
        print(f"[DP3 Debug] Action tensor shape: {action_tensor.shape}")
        print(f"[DP3 Debug] Action tensor device: {action_tensor.device}")
        print(f"[DP3 Debug] Action tensor dtype: {action_tensor.dtype}")
        
        # Print action statistics for each joint
        action_np_temp = action_tensor.squeeze(0).detach().cpu().numpy()
        joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'gripper']
        print(f"[DP3 Debug] Raw model output statistics (before any processing):")
        for i, joint in enumerate(joint_names):
            if i < action_np_temp.shape[1]:
                joint_actions = action_np_temp[:, i]
                mean_val = np.mean(joint_actions)
                std_val = np.std(joint_actions)
                min_val, max_val = np.min(joint_actions), np.max(joint_actions)
                print(f"[DP3 Debug]   {joint}: mean={mean_val:.6f}, std={std_val:.6f}, range=[{min_val:.6f}, {max_val:.6f}]")
                # Flag if completely fixed
                if std_val < 1e-6:
                    print(f"[DP3 Debug]   [WARN] {joint} appears FIXED (std~=0)")
    
    # 11. Extract action chunk
    action_chunk = action['action'].squeeze(0).detach().cpu().numpy()  # (n_action_steps, action_dim)
    
    # 12. Action post-processing - model output is already in training data format, no denormalization needed
    # Clarification: Issue is not normalizer but data conversion.
    # Training actions were already processed, so model output follows that format.
    # Need to convert model output actions back to LeRobot control format.
    # if is_first_call:
    #     print(f"[DP3 Debug] Action post-processing:")
    #     print(f"  Model output action shape: {action_chunk.shape}")
    #     print(f"  Model output j5 range: [{action_chunk[:, 4].min():.6f}, {action_chunk[:, 4].max():.6f}]")
    #     print(f"  Model output j6 range: [{action_chunk[:, 5].min():.6f}, {action_chunk[:, 5].max():.6f}]")
    #     print(f"  ℹ️ Model output format is training data format, needs conversion back to LeRobot control format")
    
    # 13. Inverse action conversion - training format back to LeRobot control format
    # convert script training data format:
    # - first 6 joints: action_rad = np.deg2rad(action)
    # - gripper: action_rad[6] / 100.0 normalized
    # Inverse:
    # - first 6 joints: action = np.rad2deg(action_rad)
    # - gripper: action = action_rad[6] * 100.0 (gripper itself might not be degree unit!)
    action_for_control = np.zeros_like(action_chunk)
    # action_for_control[:, :6] = np.rad2deg(action_chunk[:, :6])  # first 6 joints: rad -> deg
    # action_for_control[:, 6] = action_chunk[:, 6] * 100.0  # gripper: denormalize (no rad2deg)
    action_for_control = action_chunk
    
    # Detailed debug output - print full action chunk for analysis
    # print(f"\n{'='*80}")
    # print(f"🔍 **Detailed Action Chunk Analysis** (printed every prediction)")
    # print(f"{'='*80}")
    # print(f"Model output (training format, rad):")
    # print(f"  shape: {action_chunk.shape}")
    # for i in range(min(3, action_chunk.shape[0])):  # print first 3 steps
    #     print(f"  [{i}]: {action_chunk[i]}")
    
    # print(f"\nAfter inverse conversion (control format, deg):")
    # print(f"  shape: {action_for_control.shape}")
    # for i in range(min(3, action_for_control.shape[0])):
    #     print(f"  [{i}]: {action_for_control[i]}")
    
    # Compare current state and first action
    # print(f"\n📊 Current state vs First action comparison:")
    # if len(obs_history) > 0:
    #     # Get latest state from obs_history (training format, rad)
    #     latest_state_train = obs_history[-1]['agent_pos'].cpu().numpy()
    #     # Convert back to deg for comparison
    #     latest_state_deg = np.zeros(7)
    #     latest_state_deg[:6] = np.rad2deg(latest_state_train[:6])
    #     latest_state_deg[6] = np.rad2deg(latest_state_train[6] * 100.0)
        
    #     print(f"  Current state (deg): {latest_state_deg}")
    #     print(f"  First action (deg): {action_for_control[0]}")
    #     print(f"  Diff (action-state): {action_for_control[0] - latest_state_deg}")
    
    # print(f"{'='*80}\n")
    
    # if is_first_call:
    #     print(f"[DP3 Debug] Inverse action conversion (first call details):")
    #     print(f"  training format(rad) -> control format(deg)")
    #     print(f"  j5: {action_chunk[:3, 4]} -> {action_for_control[:3, 4]}")
    #     print(f"  j6: {action_chunk[:3, 5]} -> {action_for_control[:3, 5]}")
    #     print(f"  gripper: {action_chunk[:3, 6]} -> {action_for_control[:3, 6]}")
    
    # DEBUG: Final action chunk analysis
    if is_first_call:
        print(f"[DP3 Debug] Action chunk shape: {action_for_control.shape}")
        print(f"[DP3 Debug] Final action chunk statistics (LeRobot control format, degrees):")
        joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'gripper']
        for i, joint in enumerate(joint_names):
            if i < action_for_control.shape[1]:
                joint_actions = action_for_control[:, i]
                mean_val = np.mean(joint_actions)
                std_val = np.std(joint_actions)
                min_val, max_val = np.min(joint_actions), np.max(joint_actions)
                print(f"[DP3 Debug]   {joint}: mean={mean_val:.6f}, std={std_val:.6f}, range=[{min_val:.6f}, {max_val:.6f}]")
                
                # Special attention to j5/j6
                if joint in ['j5', 'j6'] and std_val < 1e-6:
                    print(f"[DP3 Debug]   [ALERT] {joint} IS FIXED! std~={std_val:.10f}")
                    print(f"[DP3 Debug]   [ALERT] All {joint} values: {joint_actions}")
        print(f"[DP3 Debug] Returning to control loop...")
    
    # PURE INFERENCE: Action scaling removed for debugging
    # Need to identify where j5/j6 become fixed without artificial scaling
    # action_chunk = apply_action_scaling(action_chunk, joint_scales=joint_scales)  # DISABLED
    
    return action_for_control, generator, obs_history


def predict_action(observation, policy, device, use_amp, robot, hdf5_path=None, demo_idx=0, current_step=0):
    if hdf5_path is not None:
        try:
            with h5py.File(hdf5_path, 'r') as f:
                demo_key = f'data/demo_{demo_idx+1}'  # HDF5 structure data/demo_1, data/demo_2, ...
                
                if demo_key in f:
                    demo_data = f[demo_key]
                    total_frames = demo_data['obs']['cam_right'].shape[0]
                    frame_idx = current_step % total_frames
                    # Read image data - according to HDF5 data structure
                    cam_right = demo_data['obs']['cam_right'][frame_idx:frame_idx+2]  # shape should be (84, 84, 3)
                    cam_left = demo_data['obs']['cam_left'][frame_idx:frame_idx+2]
                    cam_arm = demo_data['obs']['cam_arm'][frame_idx:frame_idx+2]
                    state = demo_data['obs']['state'][frame_idx:frame_idx+2]
                
                    # Convert to torch tensor and adjust format
                    observation = {
                        'cam_right': torch.from_numpy(cam_right),  # Add batch dimension (1, 84, 84, 3)
                        'cam_left': torch.from_numpy(cam_left),
                        'cam_arm': torch.from_numpy(cam_arm),
                        'state': torch.from_numpy(state)
                    }
                    print("dataset:\n", demo_data['action'][frame_idx:frame_idx+8])
                    
                else:
                    print(f"Warning: {demo_key} not found in HDF5 file")
                    # fallback to raw observation
                    
        except Exception as e:
            print(f"Error reading data from HDF5: {e}")
            import traceback
            traceback.print_exc()
    observation = copy(observation)
    # hardcode
   
    use_pcd = True
    if use_pcd:
        for name in observation:
            if "depths.cam_front" in name:
                cam_name = name.split(".")[-1]
                observation[name.replace("depths","pointclouds")] = torch.Tensor(get_point_cloud(observation[name.replace("depths","images")].numpy(),observation[name].numpy(),robot.cameras[cam_name].intrinsics,robot.inv_extrinsics[cam_name],robot.cameras[cam_name].depth_scale,robot.config.crop_bounds))
                break
    
        observation = {k:observation[k] for k in observation if 'images' not in k and 'depths' not in k}
            
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        # for act_tensor in previous_actions:

        for name in observation:
            # observation[name].squeeze(0)
            if name in ['cam_right', 'cam_left', 'cam_arm']:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(0, 3, 1, 2).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)
        # Compute the next action with the policy
        # based on the current observation
        action = policy.predict_action(observation)

    ### for policy with error detector ###
    # return action

        ### for policy yuan ###
        pred_action=action['action_pred']
        action=action['action']

        # Remove batch dimension
        action = action.squeeze()
        # Move to cpu, if not already the case
        action = action.to("cpu")
        print("policy:\n", action)
    return action, pred_action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_cameras,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_cameras=display_cameras,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
        # eval=True
    )
    # import pdb;pdb.set_trace()
    # print(robot.follower_arms.keys())
    # robot.follower_arms['main'].robot.set_servo_angle(angle=[3.1,-11.1,-48.7,-176.4,-78.7,188.3])
    # time.sleep(1)
def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_cameras,
    policy,
    fps,
    single_task,
    eval,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_cameras=display_cameras,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
        eval=eval
    )
# import matplotlib.pyplot as plt

def filter_obs(observation, data_flag):
    included_keys = []
    if data_flag == "rgb_front":
        included_keys=["observation.images.cam_front"]
    if data_flag == "rgb":
        included_keys=["observation.images.cam_front","observation.images.cam_arm","observation.images.cam_back"]
    if data_flag is None:
        included_keys = [k for k in observation]
    return {k:observation[k] for k in observation if 'cam' not in k or k in included_keys}
    
@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_cameras=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
    eval: bool = False
):
    # hardcode
    data_flag = None
    action_list=[]
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    step_counter = 0
    
    # cv2.namedWindow('Image',cv2.WINDOW_AUTOSIZE)
    # with open('action', 'rb') as file:
    #     previous_actions = pickle.load(file)
    i=0
    while timestamp < control_time_s:
        i=i+1
        start_loop_t = time.perf_counter()
        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
            # print("action from teleop",action)
            # action_tensor = torch.Tensor(
            # [18.017578125,-0.791015625,36.650390625,1.669921875,-39.90234375,201.62109375,102.18978118896484]) #task2
            # action_tensor = torch.Tensor(
            # [2.197265625,-16.259765625,33.92578125,-0.703125,-15.029296875,180.52734375,101.60584259033203]) #task2

            # # send_action expects a tensor sliceable by indices; send the tensor and store the returned actual_action
            # actual_action = robot.send_action(action_tensor)
            # action = {"action": actual_action}
            # observation = robot.capture_observation()
        else:
            observation = robot.capture_observation()

            observation = filter_obs(observation,data_flag)
            expected_keys = ['cam_right', 'cam_left', 'cam_arm']
            new_obs = {}
            for key in expected_keys:
                full_key = f'observation.images.{key}'
                if full_key in observation:
                    img_tensor = observation[full_key]  # (B, H, W, C)
                    resized_imgs = []
                    for i in range(img_tensor.shape[0]):
                        img_np = img_tensor[i].numpy()
                        resized_np = cv2.resize(img_np, (84, 84))
                        resized_tensor = torch.from_numpy(resized_np)
                        resized_imgs.append(resized_tensor)
                    new_obs[key] = torch.stack(resized_imgs)
            new_obs['state']=observation['observation.state']
            if policy is not None:
                ### for policy with error detector ###
                # pred_action = predict_action(
                #     new_obs, policy, get_safe_torch_device("cuda:0"), use_amp=False, robot=robot)
                # time_step = min(i-1, pred_action.shape[1]-1) 
                # current_action = pred_action[0, time_step, :]
                # if hasattr(current_action, 'cpu'):
                #     current_action = current_action.cpu()
                # action_list.append(current_action)
                # actual_action = robot.send_action(current_action)
                # action = {"action": actual_action}

                ### for policy yuan ###
                pred_action, action_pred = predict_action(
                    new_obs, policy, get_safe_torch_device("cuda:0"), use_amp=False, robot=robot, hdf5_path=None, demo_idx=20, current_step=step_counter)
                action_list.append(action_pred)
                for act in pred_action:
                    action = robot.send_action(act)
                    action = {"action": action}
        if dataset is not None and not eval:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        # if display_cameras and not is_headless():
        #     image_keys = [key for key in observation if "image" in key]
        #     for key in image_keys:
        #         # path = f"./test_images/{timestamp}_{key}.jpg"
        #         # import os
        #         # os.makedirs(os.path.dirname(path),exist_yuan/ok=True)
        #         # print(path)
                
        #         # res = cv2.imwrite(path,cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
                
        #         # plt.figure(key)
        #         # plt.imshow(observation[key].numpy())
        #         # plt.show()
        #         # cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
        #         pass
        #     cv2.waitKey(1)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            # print(dt_s)
            busy_wait(1 / fps - dt_s)
        
        dt_s = time.perf_counter() - start_loop_t
        
        log_info = False
        if log_info:
            log_control_info(robot, dt_s, fps=fps)

        step_counter += 1
        timestamp = time.perf_counter() - start_episode_t

        if events["exit_early"]:
            events["exit_early"] = False
            # if dataset is not None and not eval and dataset.episode_buffer and dataset.episode_buffer["size"] > 0:
            #     dataset.save_episode()
            #     dataset.clear_episode_buffer()
            break
    # with open('action','wb') as file:
    #     pickle.dump(action_list,file)


def record_episode_dp3(
    robot,
    dataset,
    events,
    episode_time_s,
    display_cameras,
    policy,
    policy_meta,
    fps,
    single_task,
    eval,
    replay_zarr_path=None,
    replay_episode_idx=0,
):
    control_loop_dp3(
        robot=robot,
        control_time_s=episode_time_s,
        display_cameras=display_cameras,
        dataset=dataset,
        events=events,
        policy=policy,
        policy_meta=policy_meta,
        fps=fps,
        teleoperate=policy is None and replay_zarr_path is None,
        single_task=single_task,
        eval=eval,
        replay_zarr_path=replay_zarr_path,
        replay_episode_idx=replay_episode_idx,
    )
# import matplotlib.pyplot as plt

def filter_obs(observation, data_flag):
    included_keys = []
    if data_flag == "rgb_front":
        included_keys=["observation.images.cam_front"]
    if data_flag == "rgb":
        included_keys=["observation.images.cam_front","observation.images.cam_arm","observation.images.cam_back"]
    if data_flag is None:
        included_keys = [k for k in observation]
    return {k:observation[k] for k in observation if 'cam' not in k or k in included_keys}
    
@safe_stop_image_writer
def control_loop_dp3(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_cameras=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy = None,  # DP3 policy
    policy_meta = None,  # DP3 metadata
    fps: int | None = None,
    single_task: str | None = None,
    eval: bool = False,
    replay_zarr_path: str | None = None,  # Path to zarr file for replaying actions
    replay_episode_idx: int = 0  # Which episode to replay from zarr
):
    """
    Control loop for DP3 policy deployment.
    Generates point clouds from RGB-D and predicts actions with action chunking.
    """
    from collections import deque
    import torch
    import numpy as np
    
    # Default fps if not provided (to prevent None fps issues)
    if fps is None:
        fps = 30  # Default to 30 FPS
        print(f"[DP3] Warning: fps was None, defaulting to {fps} FPS")
    
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    # Initialize for DP3 policy or zarr replay
    generator = None
    obs_history = None
    action_chunk = None
    action_chunk_idx = 0
    
    # Load zarr actions if replay mode
    zarr_actions = None
    import os
    parquet_path = os.getenv("PARQUET_PATH", None)
    # parquet_path = "/media/sealab/data/lerobot_outputs/task1/01/data/chunk-000/episode_000000.parquet"
    replay_zarr_path = None
    parquet_path = None

    if replay_zarr_path is not None:
        import zarr
        print(f"[Zarr Replay] Loading actions from: {replay_zarr_path}")
        print(f"[Zarr Replay] Episode: {replay_episode_idx}")
        
        root = zarr.open(replay_zarr_path, mode='r')
        episode_ends = root['meta/episode_ends'][:]
        
        if replay_episode_idx >= len(episode_ends):
            raise ValueError(f"Episode {replay_episode_idx} not found. Dataset has {len(episode_ends)} episodes.")
        
        # Calculate start and end indices
        start_idx = 0 if replay_episode_idx == 0 else episode_ends[replay_episode_idx - 1]
        end_idx = episode_ends[replay_episode_idx]
        episode_length = end_idx - start_idx
        
        # Load actions for this episode (in training format)
        all_actions = root['data/action'][:]
        zarr_actions_train = all_actions[start_idx:end_idx]
        
        # Convert to control format (training -> control)
        # CRITICAL: Gripper inverse conversion does NOT need rad2deg!
        # Convert script: action_rad = np.deg2rad(action), vector[6] = action_rad[6] / 100.0
        # Here action_rad[6] is already in radians, dividing by 100 is just normalization
        # So inverse: gripper = (vector[6] * 100.0) directly gets radians, then use rad2deg to get "degrees"
        # but gripper itself might not be in degree units! Need to check LeRobot gripper format
        zarr_actions = np.zeros_like(zarr_actions_train)
        zarr_actions[:, :6] = np.rad2deg(zarr_actions_train[:, :6])  # j1-j6: rad -> deg
        zarr_actions[:, 6] = np.rad2deg(zarr_actions_train[:, 6] * 100.0)  # gripper: denormalize only (no rad2deg)
        
        print(f"[Zarr Replay] Loaded {episode_length} actions")
        print(f"[Zarr Replay] Action statistics (degrees):")
        joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'gripper']
        for j in range(7):
            values = zarr_actions[:, j]
            print(f"  {joint_names[j]}: min={values.min():.2f}, max={values.max():.2f}, "
                  f"mean={values.mean():.2f}, std={values.std():.2f}")
        
        # Override control_time_s to match episode length
        if control_time_s is not None and not np.isinf(control_time_s):
            print(f"[Zarr Replay] Warning: control_time_s will be overridden to match episode length")
        control_time_s = episode_length / fps
        print(f"[Zarr Replay] Episode duration: {control_time_s:.2f}s at {fps} FPS")

    elif parquet_path is not None:
        def read_action_data(parquet_path):
            """
            Read action data using same way as second function
            """
            try:
                import pyarrow.parquet as pq
                
                table = pq.ParquetFile(parquet_path)
                action_data = []
                state_data=[]
                # import pdb;pdb.set_trace()
                batches = list(table.iter_batches())
                for batch in batches:
                    batch_df = batch.to_pandas()
                    # Convert each element in action column to list and stack
                    action_batch = np.array([data.tolist() for data in batch_df['action'].values])
                    state_batch=np.array([data.tolist() for data in batch_df['observation.state'].values])
                    action_data.append(action_batch)
                    state_data.append(state_batch)
                if action_data and state_data:
                    return np.vstack(action_data),np.vstack(state_data)
                else:
                    return None
                    
            except Exception as e:
                print(f"Error reading action data {parquet_path}: {str(e)}")
                return None
        action, state = read_action_data(parquet_path=parquet_path)
        zarr_actions = action
        print(f"[Parquet Replay] Loaded {len(zarr_actions)} actions from {parquet_path}")


    if policy is not None:
        if not REALTIME_PC_AVAILABLE:
            raise RuntimeError("realtime_pointcloud module not available for DP3 policy!")
        
        if policy_meta is None:
            raise ValueError("policy_meta is required for DP3 policy!")
        
        n_obs_steps = policy_meta['n_obs_steps']
        n_action_steps = policy_meta['n_action_steps']
        
        # Initialize observation history deque
        obs_history = deque(maxlen=n_obs_steps)
        
        print(f"[DP3] Initializing with n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")
        print("[DP3] Realtime pointcloud generator will be created on first use")
    
    timestamp = 0
    start_episode_t = time.perf_counter()
    step_counter = 0
    if dataset is not None:
        dataset.clear_episode_buffer()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        
        if teleoperate:
            # Teleoperation mode
            observation, action = robot.teleop_step(record_data=True)
        else:
            # Policy control mode or zarr replay mode
            observation = robot.capture_observation()
            
            if zarr_actions is not None:
                # Zarr replay mode - read action from loaded zarr data
                if step_counter >= len(zarr_actions):
                    print(f"[Zarr Replay] Reached end of episode ({len(zarr_actions)} actions)")
                    events["exit_early"] = True
                    break
                
                action_np = zarr_actions[step_counter]
                
                if step_counter % 10 == 0 or step_counter < 5:
                    print(f"[Zarr Replay] Step {step_counter}/{len(zarr_actions)}: action={action_np}")
                
                # Convert to PyTorch tensor
                action_tensor = torch.from_numpy(action_np).float()
                
                # Send action to robot
                actual_action = robot.send_action(action_tensor)
                action = {"action": actual_action}
                
            elif policy is not None:
                # DP3 Policy with action chunking
                try:
                    # Check if we need to predict new action chunk
                    if action_chunk is None or action_chunk_idx >= n_action_steps:
                        # Predict new action chunk
                        action_chunk, generator, obs_history = predict_action_dp3(
                            observation, 
                            policy, 
                            policy_meta,
                            obs_history,
                            generator=generator
                        )
                        action_chunk_idx = 0
                        print(f"[DP3] Step {step_counter}: Predicted new action chunk, shape={action_chunk.shape}")
                    
                    # Use current action from chunk
                    action_np = action_chunk[action_chunk_idx]
                    action_chunk_idx += 1
                    
                    if step_counter % 10 == 0:
                        print(f"[DP3] Step {step_counter}: Using action[{action_chunk_idx-1}/{n_action_steps}]: {action_np}")
                    
                    # Convert numpy array to PyTorch tensor for LeRobot compatibility
                    action_tensor = torch.from_numpy(action_np).float()
                    
                    # Send action to robot (LeRobot expects PyTorch tensor)
                    actual_action = robot.send_action(action_tensor)
                    action = {"action": actual_action}
                    
                except Exception as e:
                    print(f"[DP3] [ERROR] Error in policy prediction: {e}")
                    import traceback
                    traceback.print_exc()
                    # Skip this frame
                    continue
            else:
                raise ValueError("Policy is None in non-teleoperation mode")
                    
        if dataset is not None and not eval:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            print(dt_s)
            busy_wait(1 / fps - dt_s)
        
        dt_s = time.perf_counter() - start_loop_t
        
        log_info = False
        if log_info:
            log_control_info(robot, dt_s, fps=fps)

        step_counter += 1
        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break
    # with open('action','wb') as file:
    #     pickle.dump(action_list,file)


def reset_environment(robot, events, reset_time_s, fps):
    # TODO(rcadene): refactor warmup_record and reset_environment
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    control_loop(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=True,
    )


def stop_recording(robot, listener, display_cameras):
    robot.disconnect()

    if not is_headless():
        if listener is not None:
            listener.stop()

        if display_cameras:
            cv2.destroyAllWindows()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    # Skip this check for DP3 policy (diffusion3D) as it can be used for both training data collection and evaluation
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        if hasattr(policy_cfg, 'type') and policy_cfg.type != 'diffusion3D':
            raise ValueError(
                f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
            )
        else:
            # For DP3/diffusion3D policy, just print a warning
            print(f"⚠️  Warning: Using policy with non-eval dataset name '{dataset_name}'. "
                  f"If this is for evaluation, consider using 'eval_{dataset_name}' instead.")


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
