"""
RoboTwin 2.0 Environment Manager
Directly adapted from the original eval_3dpolicy.py with path corrections
"""

import sys
import os
import subprocess

# Setup paths for imports
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# Add envs directory to path
envs_dir = os.path.join(current_directory, 'envs')
if envs_dir not in sys.path:
    sys.path.insert(0, envs_dir)

# Add description utils to path
description_utils_dir = os.path.join(current_directory, 'description', 'utils')
if description_utils_dir not in sys.path:
    sys.path.insert(0, description_utils_dir)

from _GLOBAL_CONFIGS import CONFIGS_PATH
from utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import json
import torch


class RoboTwin2EnvManager:
    """
    RoboTwin 2.0 Environment Manager
    
    This is the original Env class from eval_3dpolicy.py with minimal modifications
    to work within the R3D directory structure.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def class_decorator(task_name):
        """Create task environment instance"""
        envs_module = importlib.import_module(f"r3d.env.robotwin2.envs.{task_name}")
        try:
            env_class = getattr(envs_module, task_name)
            env_instance = env_class()
        except:
            raise SystemExit(f"No Task: {task_name}")
        return env_instance
    
    @staticmethod
    def get_camera_config(camera_type):
        """Get camera configuration"""
        camera_config_path = os.path.join(current_directory, "task_config", "_camera_config.yml")
        
        assert os.path.isfile(camera_config_path), "task config file is missing"
        
        with open(camera_config_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        assert camera_type in args, f"camera {camera_type} is not defined"
        return args[camera_type]
    
    @staticmethod
    def get_embodiment_config(robot_file):
        """Get embodiment configuration"""
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        return embodiment_args
    
    @staticmethod
    def get_embodiment_file(embodiment_types, embodiment_type):
        """Get embodiment file path"""
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file
    
    def dual_arm(self):
        """Check if using dual arm"""
        return self.task.get_dual_arm()
    
    def Create_env(self, task_name, head_camera_type, seed, task_num, instruction_type, task_config):
        """
        Create and initialize the environment
        
        Returns:
            tuple: (seed_list, id_list, episode_info_list_total)
        """
        task_config_file_path = os.path.join(current_directory, 'task_config', f'{task_config}.yml')
        self.time_str = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        
        with open(task_config_file_path, 'r', encoding='utf-8') as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        self.args['task_name'] = task_name
        self.args["task_config"] = task_config
        self.args["ckpt_setting"] = None
        
        self.embodiment_type = self.args.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            self._embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
            self._camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        head_camera_type = self.args["camera"]["head_camera_type"]
        self.args["head_camera_h"] = self._camera_config[head_camera_type]["h"]
        self.args["head_camera_w"] = self._camera_config[head_camera_type]["w"]
        
        if len(self.embodiment_type) == 1:
            self.args["left_robot_file"] = self.get_embodiment_file(self._embodiment_types, self.embodiment_type[0])
            self.args["right_robot_file"] = self.get_embodiment_file(self._embodiment_types, self.embodiment_type[0])
            self.args["dual_arm_embodied"] = True
        elif len(self.embodiment_type) == 3:
            self.args["left_robot_file"] = self.get_embodiment_file(self._embodiment_types, self.embodiment_type[0])
            self.args["right_robot_file"] = self.get_embodiment_file(self._embodiment_types, self.embodiment_type[1])
            self.args["embodiment_dis"] = self.embodiment_type[2]
            self.args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3")
        
        if len(self.embodiment_type) == 1:
            self.embodiment_name = str(self.embodiment_type[0])
        else:
            self.embodiment_name = str(self.embodiment_type[0]) + "+" + str(self.embodiment_type[1])
        
        self.args["left_embodiment_config"] = self.get_embodiment_config(self.args["left_robot_file"])
        self.args["right_embodiment_config"] = self.get_embodiment_config(self.args["right_robot_file"])
        
        print("============= Config =============\n")
        print("\033[95mMessy Table:\033[0m " + str(self.args["domain_randomization"]["cluttered_table"]))
        print("\033[95mRandom Background:\033[0m " + str(self.args["domain_randomization"]["random_background"]))
        if self.args["domain_randomization"]["random_background"]:
            print(" - Clean Background Rate: " + str(self.args["domain_randomization"]["clean_background_rate"]))
        print("\033[95mRandom Light:\033[0m " + str(self.args["domain_randomization"]["random_light"]))
        if self.args["domain_randomization"]["random_light"]:
            print(" - Crazy Random Light Rate: " + str(self.args["domain_randomization"]["crazy_random_light_rate"]))
        print("\033[95mRandom Table Height:\033[0m " + str(self.args["domain_randomization"]["random_table_height"]))
        print("\033[95mRandom Head Camera Distance:\033[0m " + str(self.args["domain_randomization"]["random_head_camera_dis"]))
        
        print("\033[94mHead Camera Config:\033[0m " + str(self.args["camera"]["head_camera_type"]) + f", " +
              str(self.args["camera"]["collect_head_camera"]))
        print("\033[94mWrist Camera Config:\033[0m " + str(self.args["camera"]["wrist_camera_type"]) + f", " +
              str(self.args["camera"]["collect_wrist_camera"]))
        print("\033[94mEmbodiment Config:\033[0m " + self.embodiment_name)
        print("\n==================================")
        
        self.task = self.class_decorator(self.args["task_name"])
        self.st_seed = 100000 * (1 + seed)
        self.task_num = task_num
        self.clear_cache_freq = self.args['clear_cache_freq']
        self.args["eval_mode"] = True
        self.instruction_type = instruction_type
        
        return self.find_seed(task_num)
    
    def Init_task_env(self, seed, id, episode_info_list, run_dir, epoch, task_config):
        """Initialize task environment for a specific episode"""
        self.env_state = 0  # 0:running 1:success 2:fail
        self.step = 0
        self.succ_seed = seed
        
        self.task.setup_demo(now_ep_num=id, seed=seed, is_test=True, **self.args)
        
        results = generate_episode_descriptions(self.args["task_name"], episode_info_list, 1)
        instruction = np.random.choice(results[0][self.instruction_type])
        self.task.set_instruction(instruction=instruction)
        
        self.eval_video_log = True
        self.video_size = str(self.args['head_camera_w']) + 'x' + str(self.args['head_camera_h'])
        self.save_dir = str(epoch) + '-' + run_dir + '-' + task_config
        
        if self.eval_video_log:
            # Save to output directory relative to current location
            self.save_dir = Path(current_directory) / 'eval_video' / self.save_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.success_path = os.path.join(self.save_dir, f'success_{seed}.mp4')
            self.fail_path = os.path.join(self.save_dir, f'fail_{seed}.mp4')
            log_file = open(f'{self.save_dir}/{self.time_str}_ffmpeg_log.txt', 'w')
            self.file_path = os.path.join(self.save_dir, f'{seed}.mp4')
            
            self.ffmpeg = subprocess.Popen([
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-pixel_format', 'rgb24',
                '-video_size', self.video_size,
                '-framerate', '4',
                '-i', '-',
                '-pix_fmt', 'yuv420p',
                '-vcodec', 'libx264',
                '-preset', 'veryfast',
                '-tune', 'zerolatency',
                '-g', '15',
                '-threads', '0',
                f'{self.save_dir}/{seed}.mp4'
            ], stdin=subprocess.PIPE, stdout=log_file, stderr=log_file)
        
        return instruction
    
    def save_seed(self, seedlist, episode_info_list=None, st_seed=None):
        """Save valid seeds to file"""
        if st_seed is None:
            st_seed = self.st_seed
        st_seed_key = str(st_seed) + self.args['task_config']
        
        save_path = os.path.join(current_directory, 'seeds_list')
        file_path = os.path.join(save_path, f'{self.args["task_name"]}.json')
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        new_seeds = set(seedlist)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if st_seed_key in data:
                    existing_seeds = set(data[st_seed_key]["seeds"])
                    all_seeds = sorted(list(existing_seeds.union(new_seeds)))
                    data[st_seed_key]["seeds"] = all_seeds
                    
                    if episode_info_list:
                        if "episode_info" not in data[st_seed_key]:
                            data[st_seed_key]["episode_info"] = {}
                        
                        for i, seed in enumerate(seedlist):
                            data[st_seed_key]["episode_info"][str(seed)] = episode_info_list[i]
                else:
                    data[st_seed_key] = {"seeds": sorted(list(new_seeds))}
                    if episode_info_list:
                        data[st_seed_key]["episode_info"] = {}
                        for i, seed in enumerate(seedlist):
                            data[st_seed_key]["episode_info"][str(seed)] = episode_info_list[i]
            except (json.JSONDecodeError, FileNotFoundError):
                data = {
                    st_seed_key: {
                        "seeds": sorted(list(new_seeds)),
                        "episode_info": {}
                    }
                }
                if episode_info_list:
                    for i, seed in enumerate(seedlist):
                        data[st_seed_key]["episode_info"][str(seed)] = episode_info_list[i]
        else:
            data = {
                st_seed_key: {
                    "seeds": sorted(list(new_seeds)),
                    "episode_info": {}
                }
            }
            if episode_info_list:
                for i, seed in enumerate(seedlist):
                    data[st_seed_key]["episode_info"][str(seed)] = episode_info_list[i]
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def find_seed(self, task_num):
        """Find or load valid seeds"""
        save_path = os.path.join(current_directory, 'seeds_list')
        file_path = os.path.join(save_path, f'{self.args["task_name"]}.json')
        st_seed_key = str(self.st_seed) + self.args['task_config']
        existing_seeds = []
        existing_episode_info = {}
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if st_seed_key in data and "seeds" in data[st_seed_key]:
                    existing_seeds = data[st_seed_key]["seeds"]
                    if "episode_info" in data[st_seed_key]:
                        existing_episode_info = data[st_seed_key]["episode_info"]
            except (json.JSONDecodeError, FileNotFoundError):
                existing_seeds = []
        
        valid_seeds = existing_seeds
        
        if len(valid_seeds) >= task_num:
            selected_seeds = valid_seeds[:task_num]
            id_list = list(range(task_num))
            
            episode_info_list = []
            for seed in selected_seeds:
                seed_str = str(seed)
                if seed_str in existing_episode_info:
                    episode_info_list.append(existing_episode_info[seed_str])
                else:
                    episode_info_list.append([])
            
            print(f"Found {len(selected_seeds)} valid seeds in group {st_seed_key}")
            return selected_seeds, id_list, episode_info_list
        
        print(f"Insufficient seeds in group {st_seed_key} ({len(valid_seeds)}/{task_num}), starting to find new seeds...")
        
        needed_seeds = task_num - len(valid_seeds)
        start_seed = self.st_seed
        if valid_seeds:
            start_seed = max(valid_seeds) + 1
        
        new_seeds, new_ids, new_episode_info_list = self.Check_seed(needed_seeds, start_seed)
        
        final_seeds = valid_seeds + new_seeds
        final_seeds = final_seeds[:task_num]
        final_ids = list(range(len(final_seeds)))
        
        existing_episode_info_list = [existing_episode_info[str(s)] for s in valid_seeds if str(s) in existing_episode_info]
        final_episode_info_list = existing_episode_info_list + new_episode_info_list
        final_episode_info_list = final_episode_info_list[:task_num]
        
        self.save_seed(new_seeds, new_episode_info_list)
        
        print(f"Total found {len(final_seeds)} valid seeds in group {st_seed_key}")
        return final_seeds, final_ids, final_episode_info_list
    
    def Check_seed(self, test_num, start_seed):
        """Check and validate seeds"""
        expert_check = True
        print("Task name: ", self.args["task_name"])
        suc_seed_list = []
        now_id_list = []
        succ_tnt = 0
        now_seed = start_seed
        now_id = 0
        self.task.cus = 0
        self.task.test_num = 0
        episode_info_list_total = []
        
        while succ_tnt < test_num:
            render_freq = self.args['render_freq']
            self.args['render_freq'] = 0
            
            if expert_check:
                try:
                    self.task.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **self.args)
                    episode_info = self.task.play_once()
                    self.task.close()
                    
                except UnStableError as e:
                    print(" -------------")
                    print("Error: ", e)
                    print(" -------------")
                    self.task.close_env()
                    now_seed += 1
                    self.args["render_freq"] = render_freq
                    continue
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    print(' -------------')
                    print('Error: ', stack_trace)
                    print(' -------------')
                    self.task.close()
                    now_seed += 1
                    self.args['render_freq'] = render_freq
                    print('error occurs !')
                    continue
            
            if (not expert_check) or (self.task.plan_success and self.task.check_success()):
                suc_seed_list.append(now_seed)
                now_id_list.append(now_id)
                now_id += 1
                succ_tnt += 1
                now_seed += 1
                episode_info_list = [episode_info["info"]]
                episode_info_list_total.append(episode_info_list)
            else:
                now_seed += 1
                self.args["render_freq"] = render_freq
                continue
            
            self.args['render_freq'] = render_freq
        
        return suc_seed_list, now_id_list, episode_info_list_total
    
    def Detect_env_state(self):
        """Detect current environment state"""
        if self.step >= self.task.step_lim:
            self.env_state = 2  # fail
        if self.task.eval_success:
            self.env_state = 1  # success
    
    def Take_action(self, actions, obs_history, n_obs_steps, action_types='qpos', use_ee_space=False):
        """
        Execute action chunk in the environment
        
        Args:
            actions: Action chunk to execute
            obs_history: Observation history deque
            n_obs_steps: Number of observation steps
            action_types: 'qpos' or 'ee'
            use_ee_space: Whether using end effector space
        
        Returns:
            tuple: (status, obs_history)
                status: "success", "fail", or "run"
                obs_history: Updated observation history
        """
        # Split actions into two parts for proper observation history management
        first_stage = len(actions) - (n_obs_steps - 1)
        first_actions = actions[:first_stage]
        second_actions = actions[first_stage:]
        
        # Execute first stage actions
        self.task.take_action(first_actions, action_type=action_types)
        observation = self.get_observation()
        
        # Get agent position based on action space type
        if use_ee_space:
            left_endpose = observation['endpose']['left_endpose']
            right_endpose = observation['endpose']['right_endpose']
            left_gripper = observation['endpose']['left_gripper']
            right_gripper = observation['endpose']['right_gripper']
            agent_pos_vector = np.concatenate([
                left_endpose,
                [left_gripper],
                right_endpose,
                [right_gripper]
            ])
        else:
            agent_pos_vector = observation['joint_action']['vector']
        
        current_obs = {
            'point_cloud': torch.from_numpy(observation['pointcloud']),
            'agent_pos': torch.from_numpy(agent_pos_vector)
        }
        obs_history.append(current_obs)
        
        # Execute second stage actions and update observation history
        for action in second_actions:
            self.task.take_action(np.array([action]), action_type=action_types)
            observation = self.get_observation()
            
            if use_ee_space:
                left_endpose = observation['endpose']['left_endpose']
                right_endpose = observation['endpose']['right_endpose']
                left_gripper = observation['endpose']['left_gripper']
                right_gripper = observation['endpose']['right_gripper']
                agent_pos_vector = np.concatenate([
                    left_endpose,
                    [left_gripper],
                    right_endpose,
                    [right_gripper]
                ])
            else:
                agent_pos_vector = observation['joint_action']['vector']
            
            current_obs = {
                'point_cloud': torch.from_numpy(observation['pointcloud']),
                'agent_pos': torch.from_numpy(agent_pos_vector)
            }
            obs_history.append(current_obs)
        
        # Save video frame
        self.ffmpeg.stdin.write(observation['observation']['head_camera']['rgb'].tobytes())
        self.step += actions.shape[0]
        self.Detect_env_state()
        
        if self.env_state == 1:
            print('Task Success!')
            self.Close_env(success=True)
            return "success", obs_history
        elif self.env_state == 2:
            print('Task Failed!')
            self.Close_env(success=False)
            return "fail", obs_history
        else:
            return "run", obs_history
    
    def Close_env(self, success=False):
        """Close environment and rename video based on success"""
        observation = self.get_observation()
        self.ffmpeg.stdin.write(observation['observation']['head_camera']['rgb'].tobytes())
        self.task.close_env(clear_cache=((self.succ_seed + 1) % self.clear_cache_freq == 0))
        
        if self.eval_video_log:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            del self.ffmpeg
        
        if success:
            os.rename(self.file_path, self.success_path)
        else:
            os.rename(self.file_path, self.fail_path)
        
        if self.task.render_freq:
            self.task.viewer.close()
        
        print('Env Closed!')
        self.task._take_picture()
    
    def get_observation(self):
        """Get current observation from environment"""
        return self.task.get_obs()

