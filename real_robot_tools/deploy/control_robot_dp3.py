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
"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Recalibrate your robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=calibrate
```

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --robot.cameras='{}' \
    --control.type=teleoperate

# Add the cameras from the robot definition to visualize them:
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/koch_test \
    --control.num_episodes=1 \
    --control.push_to_hub=True
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay \
    --robot.type=so100 \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=$USER/koch_test \
    --control.episode=0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

- For remote controlled robots like LeKiwi, run this script on the robot edge device (e.g. RaspBerryPi):
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=remote_robot
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command and adding `--control.resume=true`.

- Train on this dataset with the ACT policy:
```bash
python lerobot/scrip   log_control_info(robot, dt_s, fps=cfg.fps)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[its/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --device=cuda \
  --wandb.enable=true
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/eval_act_koch_pick_place_lego \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/080000/pretrained_model
```
"""

import logging
import time
from dataclasses import asdict
from pprint import pformat

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from control_utils import (
    control_loop,
    init_keyboard_listener,
    log_control_info,
    record_episode_dp3,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser
import torch
import dill
import hydra
import sys
import os
import pathlib
from omegaconf import OmegaConf
from termcolor import cprint

# Add 3D-Diffusion-Policy path
DP3_DIR = pathlib.Path(__file__).parent / "3D-Diffusion-Policy"
sys.path.insert(0, str(DP3_DIR))

# Import DP3 workspace
try:
    from diffusion_policy_3d.workspace.train_dp3_workspace import TrainDP3Workspace
except ImportError:
    from train import TrainDP3Workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

########################################################################################
# Control modes
########################################################################################
def load_policy(policy_cfg=None):
    """
    Load DP3 Policy for real robot deployment.
    
    Args:
        policy_cfg: Policy configuration from control.policy (OmegaConf dict)
                   Should contain: path, use_ema (optional), device (optional), num_inference_steps (optional)
    
    Returns:
        policy: Loaded DP3 policy ready for inference
        policy_meta: Dict containing shape_meta and other metadata needed for inference
    """
    # Extract configuration
    if policy_cfg is None:
        raise ValueError(
            "❌ Policy configuration is required!\n"
            "Please provide --control.policy.path=/path/to/checkpoint.ckpt"
        )
    
    if not hasattr(policy_cfg, 'path') or policy_cfg.path is None:
        raise ValueError(
            "❌ Policy checkpoint path is missing!\n"
            "Please provide --control.policy.path=/path/to/checkpoint.ckpt"
        )
    
    checkpoint_path = pathlib.Path(policy_cfg.path)
    use_ema = getattr(policy_cfg, 'use_ema', True)  # Default to EMA
    device = getattr(policy_cfg, 'device', 'cuda')
    num_inference_steps = getattr(policy_cfg, 'num_inference_steps', 10)
    
    cprint(f"[Load Policy] Loading DP3 checkpoint: {checkpoint_path}", "cyan")
    
    if not checkpoint_path.exists():
        cprint(f"❌ Checkpoint not found: {checkpoint_path}", "red")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 1. Load checkpoint to get config (matching eval.py workflow)
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    
    cprint(f"[Load Policy] Task: {cfg.task_name}", "green")
    cprint(f"[Load Policy] Config: horizon={cfg.horizon}, n_obs_steps={cfg.n_obs_steps}, n_action_steps={cfg.n_action_steps}", "green")
    
    # 2. Disable DDP for inference (single GPU deployment)
    # Set environment variables to bypass distributed training setup
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    
    # Override config to disable DDP
    cfg.training.use_ddp = False
    
    # 3. Create workspace with checkpoint directory (matching eval.py's DP3_policy.get_policy())
    # The output_dir should be the folder containing checkpoints/ and .hydra/
    output_dir = str(checkpoint_path.parent.parent)
    workspace = TrainDP3Workspace(cfg, output_dir=output_dir)
    
    # 4. Use workspace.get_policy() to properly load model (matching eval.py workflow)
    # This method handles checkpoint loading + normalizer setup
    checkpoint_num = int(checkpoint_path.stem)  # Extract "100" from "100.ckpt"
    cprint(f"[Load Policy] Loading checkpoint #{checkpoint_num}...", "cyan")
    
    policy = workspace.get_policy(cfg, checkpoint_num=checkpoint_num)
    
    # 5. Respect use_ema configuration (get_policy already handles this based on cfg.training.use_ema)
    # But allow override via policy_cfg
    if use_ema and cfg.training.use_ema and workspace.ema_model is not None:
        policy = workspace.ema_model
        cprint(f"[Load Policy] Using EMA model", "green")
    else:
        policy = workspace.model
        cprint(f"[Load Policy] Using standard model", "yellow")
    
    # 6. Prepare for inference
    policy.eval()
    policy.to(device)
    policy.num_inference_steps = num_inference_steps
    
    cprint(f"[Load Policy] ✅ Policy loaded successfully!", "green")
    cprint(f"[Load Policy] Input: point_cloud {cfg.shape_meta.obs.point_cloud.shape}, agent_pos {cfg.shape_meta.obs.agent_pos.shape}", "white")
    cprint(f"[Load Policy] Output: action {cfg.shape_meta.action.shape}", "white")
    
    # 7. Prepare metadata for control loop
    policy_meta = {
        'shape_meta': cfg.shape_meta,
        'n_obs_steps': cfg.n_obs_steps,
        'n_action_steps': cfg.n_action_steps,
        'horizon': cfg.horizon,
        'device': device,
    }
    
    # Attach metadata to policy for easy access
    policy.dp3_meta = policy_meta
    
    return policy, policy_meta

@safe_disconnect
def calibrate(robot: Robot, cfg: CalibrateControlConfig):
    # TODO(aliberts): move this code in robots' classes
    if robot.robot_type.startswith("stretch"):
        if not robot.is_connected:
            robot.connect()
        if not robot.is_homed():
            robot.home()
        return

    arms = robot.available_arms if cfg.arms is None else cfg.arms
    unknown_arms = [arm_id for arm_id in arms if arm_id not in robot.available_arms]
    available_arms_str = " ".join(robot.available_arms)
    unknown_arms_str = " ".join(unknown_arms)

    if arms is None or len(arms) == 0:
        raise ValueError(
            "No arm provided. Use `--arms` as argument with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    if len(unknown_arms) > 0:
        raise ValueError(
            f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`."
        )

    for arm_id in arms:
        arm_calib_path = robot.calibration_dir / f"{arm_id}.json"
        if arm_calib_path.exists():
            print(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()
        else:
            print(f"Calibration file not found '{arm_calib_path}'")

    if robot.is_connected:
        robot.disconnect()

    if robot.robot_type.startswith("lekiwi") and "main_follower" in arms:
        print("Calibrating only the lekiwi follower arm 'main_follower'...")
        robot.calibrate_follower()
        return

    if robot.robot_type.startswith("lekiwi") and "main_leader" in arms:
        print("Calibrating only the lekiwi leader arm 'main_leader'...")
        robot.calibrate_leader()
        return

    # Calling `connect` automatically runs calibration
    # when the calibration file is missing
    robot.connect()
    robot.disconnect()
    print("Calibration is done! You can now teleoperate and record datasets!")


@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,
        display_cameras=cfg.display_cameras,
        eval = eval
    )

@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    # TODO(rcadene): Add option to record logs
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Load pretrained policy
        # For DP3, we need to handle the checkpoint path differently from LeRobot's HuggingFace-based policy system
        # Check if user provided a DP3 checkpoint path via:
        # 1. Custom attribute (set programmatically)
        # 2. Environment variable DP3_CHECKPOINT
        # 3. Hardcoded default path (if uncommented below)
        dp3_checkpoint_path = getattr(cfg, 'dp3_checkpoint', None)
        
        if dp3_checkpoint_path is None:
            # Try environment variable
            dp3_checkpoint_path = os.environ.get('DP3_CHECKPOINT', None)
            if dp3_checkpoint_path:
                cprint(f"[Record] Using DP3 checkpoint from environment: DP3_CHECKPOINT={dp3_checkpoint_path}", "cyan")
        
        # Uncomment and modify the line below to hardcode a default checkpoint path:
        # if dp3_checkpoint_path is None:
        #     dp3_checkpoint_path = "/home/sealab/zhaoby/Multi-task-3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/YOUR_CHECKPOINT_PATH/checkpoints/latest.ckpt"
        
        # Determine if we have a policy (either LeRobot policy or DP3 checkpoint)
        has_policy = cfg.policy is not None or dp3_checkpoint_path is not None
        
        # Create empty dataset or load existing saved episodes
        # Pass cfg.policy for sanity check (will be None if using DP3, which is handled in the function)
        # if has_policy:
        #     # If using DP3, we still pass None to sanity_check but the function won't raise error for eval_ datasets
        #     # because we'll load the DP3 policy right after
        #     pass  # Skip sanity check when DP3 checkpoint is provided via environment variable
        # else:
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            data_flag = cfg.data_flag
        )

    # Check for zarr replay mode
    replay_zarr_path = os.environ.get('REPLAY_ZARR', None)
    replay_zarr_path = None
    replay_episode_idx = int(os.environ.get('REPLAY_EPISODE', '0'))
    
    if replay_zarr_path is not None:
        cprint(f"[Record] 🔄 ZARR REPLAY MODE enabled!", "yellow", attrs=["bold"])
        cprint(f"[Record] Zarr path: {replay_zarr_path}", "cyan")
        cprint(f"[Record] Episode: {replay_episode_idx}", "cyan")
        cprint(f"[Record] Will replay actions from training data instead of policy inference", "yellow")
        
        # In replay mode, we don't need policy but we might still load it for metadata
        policy = None
        policy_meta = None
    # Now load the policy if we have one
    elif cfg.policy is None and dp3_checkpoint_path is None:
        policy = None
        policy_meta = None
        cprint("[Record] No policy provided, teleoperation mode", "yellow")
    elif dp3_checkpoint_path is not None:
        # DP3 policy with local checkpoint
        cprint(f"[Record] Loading DP3 policy from: {dp3_checkpoint_path}", "cyan")
        # Create a minimal policy config for load_policy
        from omegaconf import OmegaConf
        policy_cfg = OmegaConf.create({
            'path': dp3_checkpoint_path,
            'use_ema': True,
            'device': 'cuda',
            'num_inference_steps': 10
        })
        policy, policy_meta = load_policy(policy_cfg)
        cprint(f"[Record] DP3 policy loaded with n_obs_steps={policy_meta['n_obs_steps']}, n_action_steps={policy_meta['n_action_steps']}", "green")
    else:
        # LeRobot policy (HuggingFace-based)
        cprint("[Record] Loading LeRobot policy (not DP3)", "yellow")
        policy, policy_meta = load_policy(cfg.policy)
        cprint(f"[Record] Policy loaded", "green")
    
    if not robot.is_connected:
        robot.connect()
        
    # dataset.get_camera_info(robot.cameras)

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    pcd_only = False
    eval = False
    if not pcd_only:
        log_say("Warmup record", cfg.play_sounds)
        warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_cameras, cfg.fps)

        if has_method(robot, "teleop_safety_stop"):
            robot.teleop_safety_stop()

        recorded_episodes = 0
        episode_indexs = []
        while True:
            if recorded_episodes >= cfg.num_episodes:
                break

            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            record_episode_dp3(
                robot=robot,
                dataset=dataset,
                events=events,
                episode_time_s=cfg.episode_time_s,
                display_cameras=cfg.display_cameras,
                policy=policy,
                policy_meta=policy_meta,
                fps=cfg.fps,
                single_task=cfg.single_task,
                eval=eval,
                replay_zarr_path=replay_zarr_path,
                replay_episode_idx=replay_episode_idx,
            )

            # Execute a few seconds without recording to give time to manually reset the environment
            # Current code logic doesn't allow to teleoperate during this time.
            # TODO(rcadene): add an option to enable teleoperation during reset
            # Skip reset for the last episode to be recorded
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment", cfg.play_sounds)
                reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

            if events["rerecord_episode"]:
                print('clear')
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            if not eval:
                # try:
                    eidx = dataset.save_episode()
                    episode_indexs.append(eidx)
                    # dataset.clear_episode_buffer()
                    print(f"✅ Episode {recorded_episodes} saved successfully")
                # except Exception as e:
                #     print(f"⚠️ Warning: Failed to save episode {recorded_episodes}: {str(e)}")
                #     print("📄 This is likely due to empty camera buffers, but DP3 control worked correctly!")
                #     print("🤖 DP3 policy executed successfully - robot movements were controlled properly")
            recorded_episodes += 1

            if events["stop_recording"]:
                break

        log_say("Stop recording", cfg.play_sounds, blocking=True)
        stop_recording(robot, listener, cfg.display_cameras)
    # if not eval:
        # episode_indexs = range(0,47)
        # dataset.get_pointclouds(episode_indexs,robot,resume=False)

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
):
    # TODO(rcadene, aliberts): refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO(rcadene): Add option to record logs
    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=[cfg.episode])
    actions = dataset.hf_dataset.select_columns("action")

    if not robot.is_connected:
        robot.connect()
    # policy,action = load_policy()
    
    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    # import pdb;pdb.set_trace()
    # with h5py.File('/home/sealab/pick_and_place.hdf5', 'r') as f:
    #     action = f['demo_66']['action'][()]
    #     image=f['demo_66']['obs']['cam_left'][()]

    #     import matplotlib.pyplot as plt
    #     plt.imshow(image[0])
    #     plt.axis('off')
    #     plt.show()
    # policy,action=load_policy()
    
    # for idx in range(action.shape[0]):
    #     start_episode_t = time.perf_counter()
    #     robot.send_action(action[idx])
    #     dt_s = time.perf_counter() - start_episode_t
    #     busy_wait(1 / cfg.fps - dt_s)
    #     dt_s = time.perf_counter() - start_episode_t
    #     log_control_info(robot, dt_s, fps=cfg.fps)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()
        
        action = actions[idx]["action"]
        # print(action)
        robot.send_action(action)
    
        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    """
    Control robot with optional DP3 policy.
    
    To use DP3 policy, set DP3_CHECKPOINT environment variable or uncomment hardcoded path in record() function.
    
    Args:
        cfg: LeRobot control pipeline configuration
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    robot = make_robot_from_config(cfg.robot)
    # print(cfg.robot)
    # import pdb;pdb.set_trace()
    if isinstance(cfg.control, CalibrateControlConfig):
        calibrate(robot, cfg.control)
    elif isinstance(cfg.control, TeleoperateControlConfig):
        teleoperate(robot, cfg.control)
    elif isinstance(cfg.control, RecordControlConfig):
        record(robot, cfg.control)
    elif isinstance(cfg.control, ReplayControlConfig):
        replay(robot, cfg.control)
    elif isinstance(cfg.control, RemoteRobotConfig):
        from lerobot.common.robot_devices.robots.lekiwi_remote import run_lekiwi

        run_lekiwi(cfg.robot)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    # import pdb;pdb.set_trace()
    control_robot()
