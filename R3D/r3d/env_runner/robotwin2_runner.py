"""
RoboTwin 2.0 Environment Runner for R3D
Directly uses the original RoboTwin 2.0 interface without gym wrapper
"""

import numpy as np
import torch
import tqdm
import os
from pathlib import Path
from datetime import datetime
from termcolor import cprint
from collections import deque

from r3d.policy.base_policy import BasePolicy
from r3d.common.pytorch_util import dict_apply
from r3d.env_runner.base_runner import BaseRunner
import r3d.common.logger_util as logger_util


class RoboTwin2Runner(BaseRunner):
    """
    RoboTwin 2.0 Environment Runner

    This runner directly uses the original RoboTwin 2.0 Env interface,
    supporting action chunk prediction and execution without MultiStepWrapper.
    """

    # robotwin2 package uses many `./assets/...` relative paths, so CWD must
    # be the robotwin2 directory whenever its code runs.
    _ROBOTWIN2_DIR = str(Path(__file__).parent.parent / 'env' / 'robotwin2')


    def __init__(
        self,
        output_dir: str,
        task_name: str = "beat_block_hammer",
        seed: int = 1,
        eval_episodes: int = 20,
        max_steps: int = 1000,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        task_config: str = "demo_clean",
        instruction_type: str = "unseen",
        action_space_type: str = "joint",
        head_camera_type: str = "D435",
        save_video: bool = True,
        tqdm_interval_sec: float = 5.0,
        **kwargs
    ):
        """
        Initialize RoboTwin 2.0 Runner
        
        Args:
            output_dir: Directory to save outputs
            task_name: Name of the task to evaluate
            seed: Random seed
            eval_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            n_obs_steps: Number of observation steps for history
            n_action_steps: Number of action steps in action chunk
            task_config: Task configuration name (demo_clean, demo_randomized, etc.)
            instruction_type: Instruction type ('seen' or 'unseen')
            action_space_type: Action space type ('joint' or 'ee')
            head_camera_type: Head camera type (D435, L515, etc.)
            save_video: Whether to save evaluation videos
            tqdm_interval_sec: Progress bar update interval
        """
        super().__init__(output_dir)
        
        self.task_name = task_name
        self.seed = seed
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.task_config = task_config
        self.instruction_type = instruction_type
        self.action_space_type = action_space_type
        self.head_camera_type = head_camera_type
        self.save_video = save_video
        self.tqdm_interval_sec = tqdm_interval_sec
        
        # Logger utilities
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        
        # Import and create RoboTwin2EnvManager with CWD set to the robotwin2
        # directory, since the package uses many ./assets/... relative paths.
        _orig_cwd = os.getcwd()
        os.chdir(self._ROBOTWIN2_DIR)
        from r3d.env.robotwin2 import RoboTwin2EnvManager
        self.env_manager = RoboTwin2EnvManager()
        os.chdir(_orig_cwd)
        
        cprint(f"[RoboTwin2Runner] Initialized for task: {task_name}", "cyan")
        cprint(f"[RoboTwin2Runner] Action space: {action_space_type}", "cyan")
        cprint(f"[RoboTwin2Runner] Eval episodes: {eval_episodes}", "cyan")
        cprint(f"[RoboTwin2Runner] n_obs_steps: {n_obs_steps}, n_action_steps: {n_action_steps}", "cyan")
    
    def run(self, policy: BasePolicy, epoch, task_config):
        """
        Run evaluation on RoboTwin 2.0 environment
        
        Args:
            policy: Policy to evaluate
        
        Returns:
            log_data: Dictionary containing evaluation metrics
        """
        # robotwin2 uses ./assets/... relative paths throughout, so switch CWD.
        _orig_cwd = os.getcwd()
        os.chdir(self._ROBOTWIN2_DIR)

        device = policy.device

        # Create environment and get valid seeds
        result = self.env_manager.Create_env(
            task_name=self.task_name,
            head_camera_type=self.head_camera_type,
            seed=self.seed,
            task_num=self.eval_episodes,
            instruction_type=self.instruction_type,
            task_config=task_config
        )
        
        if not result:
            cprint("Failed to get valid seeds", "red")
            return {"test_mean_score": 0.0, "success_rate": 0.0}
        
        seed_list, id_list, episode_info_list_total = result
        cprint(f"Found {len(seed_list)} valid task seeds: {seed_list}", "green")
        
        # Evaluation metrics
        all_success = []
        all_episode_rewards = []
        episode_details = []
        
        # Get run_dir identifier from output_dir
        run_dir = os.path.basename(self.output_dir)
        # Determine which checkpoint epoch we're evaluating (default to 0 if not specified)
        # epoch = 0  # This could be passed as a parameter if needed
        
        # Main evaluation loop
        for i, (episode_seed, task_id, episode_info_list) in enumerate(
            tqdm.tqdm(
                zip(seed_list, id_list, episode_info_list_total),
                total=len(seed_list),
                desc=f"Eval RoboTwin2 {self.task_name} ({self.action_space_type})",
                leave=False,
                mininterval=self.tqdm_interval_sec
            )
        ):
            try:
                # Initialize task environment
                inst = self.env_manager.Init_task_env(episode_seed, task_id, episode_info_list, run_dir, epoch, task_config)
                
                # Reset policy
                policy.reset()
                
                # Episode state
                done = False
                episode_reward = 0
                episode_length = 0
                
                # Observation history for multi-step observations
                obs_history = deque(maxlen=self.n_obs_steps)
                
                # Get initial observation
                observation = self.env_manager.get_observation()
                
                # Get agent position based on action space type
                if self.action_space_type == 'ee':
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
                
                # Initialize observation history
                current_obs = {
                    'point_cloud': torch.from_numpy(observation['pointcloud']),
                    'agent_pos': torch.from_numpy(agent_pos_vector)
                }
                
                # Fill observation history
                for _ in range(self.n_obs_steps):
                    obs_history.append(current_obs.copy())
                
                # Episode loop
                while not done and episode_length < self.max_steps:
                    # Build observation dict from history
                    obs_dict = {
                        'point_cloud': torch.stack([o['point_cloud'] for o in obs_history], dim=0),
                        'agent_pos': torch.stack([o['agent_pos'] for o in obs_history], dim=0)
                    }
                    
                    # Add batch dimension and move to device
                    obs_dict_input = dict_apply(
                        obs_dict,
                        lambda x: x.unsqueeze(0).to(device=device)
                    )
                    
                    # Predict action chunk
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict_input)
                    
                    # Get action chunk and convert to numpy
                    action_chunk = action_dict['action'].squeeze(0).detach().cpu().numpy()
                    
                    # Execute action chunk using the original Take_action interface
                    # This handles action chunk execution and observation history update internally
                    action_type = 'ee' if self.action_space_type == 'ee' else 'qpos'
                    use_ee_space = (self.action_space_type == 'ee')
                    
                    status, obs_history = self.env_manager.Take_action(
                        action_chunk,
                        obs_history,
                        self.n_obs_steps,
                        action_types=action_type,
                        use_ee_space=use_ee_space
                    )
                    
                    # Update metrics
                    episode_length += action_chunk.shape[0]
                    
                    # Check status
                    if status == "success":
                        done = True
                        episode_reward = 1.0
                        success = True
                    elif status == "fail":
                        done = True
                        episode_reward = 0.0
                        success = False
                    else:  # status == "run"
                        done = False
                        success = False
                
                # Record episode results
                all_success.append(success)
                all_episode_rewards.append(episode_reward)
                
                episode_details.append({
                    'episode': i,
                    'success': success,
                    'reward': episode_reward,
                    'length': episode_length,
                    'seed': episode_seed
                })
                
                # Print episode result
                status_color = 'green' if success else 'red'
                cprint(
                    f"{task_config}:"
                    f"Episode {i + 1}/{len(seed_list)}: "
                    f"{'SUCCESS' if success else 'FAIL'} "
                    f"(reward: {episode_reward:.2f}, steps: {episode_length})",
                    status_color
                )
                
            except Exception as e:
                cprint(f"Episode {i} (seed {episode_seed}) failed with error: {e}", 'red')
                import traceback
                traceback.print_exc()
                all_success.append(False)
                all_episode_rewards.append(0)
                episode_details.append({
                    'episode': i,
                    'success': False,
                    'reward': 0,
                    'length': 0,
                    'seed': episode_seed,
                    'error': str(e)
                })
        
        # Calculate statistics
        success_rate = np.mean(all_success)
        mean_reward = np.mean(all_episode_rewards)
        
        # Update logger utilities
        self.logger_util_test.record(success_rate)
        self.logger_util_test10.record(success_rate)
        
        # Create log data
        log_data = {
            f'{task_config}: success_rate': success_rate,
        }
        
        # Print summary
        cprint("\n" + "="*60, "cyan")
        cprint(f"RoboTwin 2.0 Evaluation Summary - {self.task_name}", "cyan")
        cprint(f"Task Config - {task_config}", "cyan")
        cprint("="*60, "cyan")
        cprint(f"Success Rate: {success_rate:.2%} ({np.sum(all_success)}/{len(all_success)})", "yellow")
        cprint(f"Mean Reward: {mean_reward:.3f}", "yellow")
        cprint(f"Action Space: {self.action_space_type}", "yellow")
        cprint(f"Instruction Type: {self.instruction_type}", "yellow")
        cprint("="*60 + "\n", "cyan")
        
        # Save detailed results to file
        # self._save_results(episode_details, success_rate, mean_reward)

        os.chdir(_orig_cwd)
        return log_data
    
    def _save_results(self, episode_details, success_rate, mean_reward):
        """Save detailed evaluation results to file"""
        results_dir = os.path.join(self.output_dir, 'evaluation_results')
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(
            results_dir,
            f'{self.task_name}_{self.action_space_type}_{timestamp}.txt'
        )
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"RoboTwin 2.0 Evaluation Results\n")
            f.write(f"="*60 + "\n")
            f.write(f"Task: {self.task_name}\n")
            f.write(f"Action Space: {self.action_space_type}\n")
            f.write(f"Instruction Type: {self.instruction_type}\n")
            f.write(f"Task Config: {self.task_config}\n")
            f.write(f"Eval Episodes: {self.eval_episodes}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"n_obs_steps: {self.n_obs_steps}\n")
            f.write(f"n_action_steps: {self.n_action_steps}\n")
            f.write(f"\n")
            f.write(f"Overall Success Rate: {success_rate:.2%}\n")
            f.write(f"Mean Reward: {mean_reward:.3f}\n")
            f.write(f"\n")
            f.write(f"Episode Details:\n")
            f.write(f"-"*60 + "\n")
            f.write(f"{'Episode':<10} {'Seed':<10} {'Success':<10} {'Reward':<10} {'Length':<10}\n")
            f.write(f"-"*60 + "\n")
            
            for detail in episode_details:
                f.write(
                    f"{detail['episode']:<10} "
                    f"{detail.get('seed', 'N/A'):<10} "
                    f"{'✓' if detail['success'] else '✗':<10} "
                    f"{detail['reward']:<10.2f} "
                    f"{detail['length']:<10}\n"
                )
        
        cprint(f"Detailed results saved to: {results_file}", "green")
