"""
RoboTwin 2.0 Environment Runner for R3D.

Supports both legacy single-task evaluation and multi-task checkpoints that are
conditioned with a task_onehot observation.
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


def _as_plain_dict(item):
    if hasattr(item, "items"):
        return dict(item)
    return item


class RoboTwin2Runner(BaseRunner):
    """
    RoboTwin 2.0 Environment Runner.

    This runner directly uses the original RoboTwin 2.0 Env interface,
    supporting action chunk prediction and execution without MultiStepWrapper.
    """

    _ROBOTWIN2_DIR = str(Path(__file__).parent.parent / 'env' / 'robotwin2')

    def __init__(
        self,
        output_dir: str,
        task_name: str = "beat_block_hammer",
        task_entries=None,
        eval_task_name=None,
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
        super().__init__(output_dir)

        self.task_name = task_name
        self.task_entries = None
        if task_entries is not None:
            self.task_entries = [_as_plain_dict(item) for item in task_entries]
        self.eval_task_name = eval_task_name
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

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        _orig_cwd = os.getcwd()
        os.chdir(self._ROBOTWIN2_DIR)
        from r3d.env.robotwin2 import RoboTwin2EnvManager
        self.env_manager = RoboTwin2EnvManager()
        os.chdir(_orig_cwd)

        if self.task_entries is None:
            cprint(f"[RoboTwin2Runner] Initialized for task: {task_name}", "cyan")
        else:
            cprint(f"[RoboTwin2Runner] Initialized for {len(self.task_entries)} task entries", "cyan")
            if self.eval_task_name is not None:
                cprint(f"[RoboTwin2Runner] Single-task override: {self.eval_task_name}", "cyan")
        cprint(f"[RoboTwin2Runner] Action space: {action_space_type}", "cyan")
        cprint(f"[RoboTwin2Runner] Eval episodes: {eval_episodes}", "cyan")
        cprint(f"[RoboTwin2Runner] n_obs_steps: {n_obs_steps}, n_action_steps: {n_action_steps}", "cyan")

    def _task_onehot(self, task_idx, num_tasks, device=None):
        task_onehot = torch.zeros(num_tasks, dtype=torch.float32, device=device)
        task_onehot[task_idx] = 1.0
        return task_onehot

    def _add_task_onehot(self, obs, task_idx=None, num_tasks=None):
        if task_idx is None or num_tasks is None:
            return obs
        obs = obs.copy()
        obs['task_onehot'] = self._task_onehot(task_idx, num_tasks)
        return obs

    def _ensure_history_task_onehot(self, obs_history, task_idx=None, num_tasks=None):
        if task_idx is None or num_tasks is None:
            return obs_history
        fixed_history = deque(maxlen=self.n_obs_steps)
        for obs in obs_history:
            fixed_history.append(self._add_task_onehot(obs, task_idx, num_tasks))
        return fixed_history

    def _agent_pos_from_observation(self, observation):
        if self.action_space_type == 'ee':
            left_endpose = observation['endpose']['left_endpose']
            right_endpose = observation['endpose']['right_endpose']
            left_gripper = observation['endpose']['left_gripper']
            right_gripper = observation['endpose']['right_gripper']
            return np.concatenate([
                left_endpose,
                [left_gripper],
                right_endpose,
                [right_gripper],
            ])
        return observation['joint_action']['vector']

    def _run_single_task(self, policy: BasePolicy, epoch, task_name, task_config, task_idx=None, num_tasks=None):
        _orig_cwd = os.getcwd()
        os.chdir(self._ROBOTWIN2_DIR)
        try:
            device = policy.device
            result = self.env_manager.Create_env(
                task_name=task_name,
                head_camera_type=self.head_camera_type,
                seed=self.seed,
                task_num=self.eval_episodes,
                instruction_type=self.instruction_type,
                task_config=task_config,
            )

            if not result:
                cprint("Failed to get valid seeds", "red")
                return {f"{task_name}/{task_config}: success_rate": 0.0, "test_mean_score": 0.0}

            seed_list, id_list, episode_info_list_total = result
            cprint(f"Found {len(seed_list)} valid task seeds: {seed_list}", "green")

            all_success = []
            all_episode_rewards = []
            episode_details = []
            run_dir = os.path.basename(self.output_dir)

            for i, (episode_seed, task_id, episode_info_list) in enumerate(
                tqdm.tqdm(
                    zip(seed_list, id_list, episode_info_list_total),
                    total=len(seed_list),
                    desc=f"Eval RoboTwin2 {task_name} ({self.action_space_type})",
                    leave=False,
                    mininterval=self.tqdm_interval_sec,
                )
            ):
                try:
                    self.env_manager.Init_task_env(episode_seed, task_id, episode_info_list, run_dir, epoch, task_config)
                    policy.reset()

                    done = False
                    episode_reward = 0
                    episode_length = 0
                    obs_history = deque(maxlen=self.n_obs_steps)

                    observation = self.env_manager.get_observation()
                    agent_pos_vector = self._agent_pos_from_observation(observation)
                    current_obs = {
                        'point_cloud': torch.from_numpy(observation['pointcloud']),
                        'agent_pos': torch.from_numpy(agent_pos_vector),
                    }
                    current_obs = self._add_task_onehot(current_obs, task_idx, num_tasks)

                    for _ in range(self.n_obs_steps):
                        obs_history.append(current_obs.copy())

                    while not done and episode_length < self.max_steps:
                        obs_dict = {
                            key: torch.stack([o[key] for o in obs_history], dim=0)
                            for key in obs_history[0].keys()
                        }

                        obs_dict_input = dict_apply(
                            obs_dict,
                            lambda x: x.unsqueeze(0).to(device=device),
                        )

                        with torch.no_grad():
                            action_dict = policy.predict_action(obs_dict_input)

                        action_chunk = action_dict['action'].squeeze(0).detach().cpu().numpy()
                        action_type = 'ee' if self.action_space_type == 'ee' else 'qpos'
                        use_ee_space = self.action_space_type == 'ee'

                        status, obs_history = self.env_manager.Take_action(
                            action_chunk,
                            obs_history,
                            self.n_obs_steps,
                            action_types=action_type,
                            use_ee_space=use_ee_space,
                        )
                        obs_history = self._ensure_history_task_onehot(obs_history, task_idx, num_tasks)

                        episode_length += action_chunk.shape[0]

                        if status == "success":
                            done = True
                            episode_reward = 1.0
                            success = True
                        elif status == "fail":
                            done = True
                            episode_reward = 0.0
                            success = False
                        else:
                            done = False
                            success = False

                    all_success.append(success)
                    all_episode_rewards.append(episode_reward)
                    episode_details.append({
                        'episode': i,
                        'success': success,
                        'reward': episode_reward,
                        'length': episode_length,
                        'seed': episode_seed,
                    })

                    status_color = 'green' if success else 'red'
                    cprint(
                        f"{task_name}/{task_config}: "
                        f"Episode {i + 1}/{len(seed_list)}: "
                        f"{'SUCCESS' if success else 'FAIL'} "
                        f"(reward: {episode_reward:.2f}, steps: {episode_length})",
                        status_color,
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
                        'error': str(e),
                    })

            success_rate = float(np.mean(all_success)) if len(all_success) > 0 else 0.0
            mean_reward = float(np.mean(all_episode_rewards)) if len(all_episode_rewards) > 0 else 0.0
            self.logger_util_test.record(success_rate)
            self.logger_util_test10.record(success_rate)

            log_prefix = f"{task_name}/{task_config}"
            log_data = {
                f"{log_prefix}: success_rate": success_rate,
                f"{log_prefix}: mean_reward": mean_reward,
                "test_mean_score": success_rate,
            }

            cprint("\n" + "="*60, "cyan")
            cprint(f"RoboTwin 2.0 Evaluation Summary - {task_name}", "cyan")
            cprint(f"Task Config - {task_config}", "cyan")
            cprint("="*60, "cyan")
            cprint(f"Success Rate: {success_rate:.2%} ({np.sum(all_success)}/{len(all_success)})", "yellow")
            cprint(f"Mean Reward: {mean_reward:.3f}", "yellow")
            cprint(f"Action Space: {self.action_space_type}", "yellow")
            cprint(f"Instruction Type: {self.instruction_type}", "yellow")
            cprint("="*60 + "\n", "cyan")

            return log_data
        finally:
            os.chdir(_orig_cwd)

    def run(self, policy: BasePolicy, epoch, task_config):
        if self.task_entries is None:
            resolved_task_config = task_config or self.task_config
            return self._run_single_task(
                policy=policy,
                epoch=epoch,
                task_name=self.task_name,
                task_config=resolved_task_config,
            )

        selected_entries = []
        for idx, entry in enumerate(self.task_entries):
            entry = _as_plain_dict(entry)
            entry_task_name = entry['task_name']
            if self.eval_task_name is not None and entry_task_name != self.eval_task_name:
                continue
            selected_entries.append((idx, entry))

        if len(selected_entries) == 0:
            raise ValueError(f"No RoboTwin eval tasks match eval_task_name={self.eval_task_name}")

        log_data = {}
        success_rates = []
        num_tasks = len(self.task_entries)
        for task_idx, entry in selected_entries:
            entry_task_name = entry['task_name']
            entry_task_config = entry.get('setting') or entry.get('task_config') or task_config or self.task_config
            task_log = self._run_single_task(
                policy=policy,
                epoch=epoch,
                task_name=entry_task_name,
                task_config=entry_task_config,
                task_idx=task_idx,
                num_tasks=num_tasks,
            )
            log_data.update(task_log)
            success_rates.append(task_log.get('test_mean_score', 0.0))

        mean_success = float(np.mean(success_rates)) if len(success_rates) > 0 else 0.0
        log_data['multi_task/mean_success_rate'] = mean_success
        log_data['test_mean_score'] = mean_success
        return log_data

    def _save_results(self, episode_details, success_rate, mean_reward):
        results_dir = os.path.join(self.output_dir, 'evaluation_results')
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(
            results_dir,
            f'{self.task_name}_{self.action_space_type}_{timestamp}.txt',
        )

        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("RoboTwin 2.0 Evaluation Results\n")
            f.write("="*60 + "\n")
            f.write(f"Task: {self.task_name}\n")
            f.write(f"Action Space: {self.action_space_type}\n")
            f.write(f"Instruction Type: {self.instruction_type}\n")
            f.write(f"Task Config: {self.task_config}\n")
            f.write(f"Eval Episodes: {self.eval_episodes}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"n_obs_steps: {self.n_obs_steps}\n")
            f.write(f"n_action_steps: {self.n_action_steps}\n")
            f.write("\n")
            f.write(f"Overall Success Rate: {success_rate:.2%}\n")
            f.write(f"Mean Reward: {mean_reward:.3f}\n")
            f.write("\n")
            f.write("Episode Details:\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Episode':<10} {'Seed':<10} {'Success':<10} {'Reward':<10} {'Length':<10}\n")
            f.write("-"*60 + "\n")

            for detail in episode_details:
                f.write(
                    f"{detail['episode']:<10} "
                    f"{detail.get('seed', 'N/A'):<10} "
                    f"{detail['success']!s:<10} "
                    f"{detail['reward']:<10.2f} "
                    f"{detail['length']:<10}\n"
                )

        cprint(f"Detailed results saved to: {results_file}", "green")
