import numpy as np
import torch
import os

from r3d.policy.base_policy import BasePolicy
from r3d.common.pytorch_util import dict_apply
from r3d.env_runner.base_runner import BaseRunner


class ManiskillRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=2,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512
                 ):
        self.task_name = task_name

        from mani_skill2.utils.eval_3dpolicy import Maniskill2Env
        from r3d.gym_util.maniskill_multistep_wrapper import MultiStepWrapper

        def env_fn(task_name):
            return MultiStepWrapper(
                env_name=task_name,
                down_sample_point=1024,
                save_dir=os.path.join(output_dir, 'maniskill_eval'),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec


    def run(self, policy: BasePolicy, epoch=None, task_config=None):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        # Reset per-run stats on the persistent env so counts don't carry over
        # between eval rounds. The underlying SAPIEN scene is kept alive — we
        # must not call close() between rounds, otherwise the next reset() will
        # hit a NoneType scene in _clear_sim_state.
        env.reset_stats()

        for episode_idx in range(self.eval_episodes):
            # start rollout
            obs = env.reset(seed=episode_idx + 100000)
            policy.reset()

            done = False
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['pointcloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['state'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, done, success = env.step(action)

                done = np.all(done)
                is_success = success or is_success
                if done:
                    break

            all_success_rates.append(is_success)


        # IMPORTANT: do NOT call env.close_env() here. The runner is created
        # once and reused for every eval round; closing would destroy the
        # SAPIEN scene and the next reset() would crash in _clear_sim_state.
        final_success_num, final_eval_num = env.get_stats()

        # Create log data
        log_data = {
            'success_rate': final_success_num / final_eval_num,
        }

        return log_data