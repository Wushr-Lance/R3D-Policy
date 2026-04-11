import numpy as np
import torch
from collections import defaultdict, deque
from mani_skill2.utils.eval_3dpolicy import Maniskill2Env
import dill


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x[-n:])
    else:
        return np.array(x[-n:])



def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method='max'):
    if isinstance(data[0], torch.Tensor):
        if method == 'max':
            # equivalent to any
            return torch.max(torch.stack(data))
        elif method == 'min':
            # equivalent to all
            return torch.min(torch.stack(data))
        elif method == 'mean':
            return torch.mean(torch.stack(data))
        elif method == 'sum':
            return torch.sum(torch.stack(data))
        else:
            raise NotImplementedError()
    else:
        if method == 'max':
            # equivalent to any
            return np.max(data)
        elif method == 'min':
            # equivalent to all
            return np.min(data)
        elif method == 'mean':
            return np.mean(data)
        elif method == 'sum':
            return np.sum(data)
        else:
            raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    if isinstance(all_obs[0], np.ndarray):
        result = np.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], torch.Tensor):
        result = torch.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    else:
        raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
    return result


class MultiStepWrapper():
    def __init__(self, 
            env_name,
            down_sample_point,
            save_dir,
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
        ):
        self.env = Maniskill2Env(env_name, down_sample_point, save_dir)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.done = list()
    
    def reset(self,seed):
        """Resets the environment using kwargs."""
        obs = self.env.reset(seed = seed)

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.done = list()

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, state, success = self.env.step(act)

            self.obs.append(observation)
            # self.reward.append(reward)
            done = not state
            self.done.append(done)
            if state == False:
                done = True
                observation = self._get_obs(self.n_obs_steps)
                return observation, done, success

        observation = self._get_obs(self.n_obs_steps)
        done = aggregate(self.done, 'max')
        return observation, done, False

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        result = dict()
        for key in ['pointcloud','state']:
            result[key] = stack_last_n_obs(
                [obs[key] for obs in self.obs],
                n_steps
            )
        return result

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
    def close_env(self):
        final_success_num, final_eval_num =self.env.close()
        return final_success_num, final_eval_num