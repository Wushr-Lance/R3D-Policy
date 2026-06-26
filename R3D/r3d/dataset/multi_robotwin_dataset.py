from typing import Dict, List
import copy
import os

import numpy as np
import torch
import zarr
from termcolor import cprint

from r3d.common.pytorch_util import dict_apply
from r3d.common.replay_buffer import ReplayBuffer
from r3d.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from r3d.dataset.base_dataset import BaseDataset
from r3d.dataset.robotwin_dataset import add_noise, apply_color_jitter
from r3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def _as_plain_dict(item):
    if hasattr(item, "items"):
        return dict(item)
    return item


def _identity_normalizer(dim, dtype=torch.float32):
    scale = torch.ones(dim, dtype=dtype)
    offset = torch.zeros(dim, dtype=dtype)
    input_stats_dict = {
        "min": torch.zeros(dim, dtype=dtype),
        "max": torch.ones(dim, dtype=dtype),
        "mean": torch.zeros(dim, dtype=dtype),
        "std": torch.ones(dim, dtype=dtype),
    }
    return SingleFieldLinearNormalizer.create_manual(scale, offset, input_stats_dict)


def _fit_arrays_normalizer(
        arrays,
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
    if mode not in ["limits", "gaussian"]:
        raise ValueError(f"Unsupported normalizer mode: {mode}")
    if last_n_dims <= 0:
        raise ValueError("MultiRobotwinDataset expects last_n_dims > 0")

    input_min = None
    input_max = None
    total_sum = None
    total_sumsq = None
    total_count = 0

    for array in arrays:
        arr = np.asarray(array)
        dim = int(np.prod(arr.shape[-last_n_dims:]))
        arr = arr.reshape(-1, dim).astype(np.float64, copy=False)
        arr_min = arr.min(axis=0)
        arr_max = arr.max(axis=0)
        arr_sum = arr.sum(axis=0)
        arr_sumsq = np.square(arr).sum(axis=0)
        if input_min is None:
            input_min = arr_min
            input_max = arr_max
            total_sum = arr_sum
            total_sumsq = arr_sumsq
        else:
            input_min = np.minimum(input_min, arr_min)
            input_max = np.maximum(input_max, arr_max)
            total_sum += arr_sum
            total_sumsq += arr_sumsq
        total_count += arr.shape[0]

    input_mean = total_sum / total_count
    variance = np.maximum(total_sumsq / total_count - np.square(input_mean), 0.0)
    input_std = np.sqrt(variance)

    input_min = torch.as_tensor(input_min, dtype=dtype)
    input_max = torch.as_tensor(input_max, dtype=dtype)
    input_mean = torch.as_tensor(input_mean, dtype=dtype)
    input_std = torch.as_tensor(input_std, dtype=dtype)

    if mode == "limits":
        if fit_offset:
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        else:
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    else:
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale
        if fit_offset:
            offset = -input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    input_stats_dict = {
        "min": input_min,
        "max": input_max,
        "mean": input_mean,
        "std": input_std,
    }
    return SingleFieldLinearNormalizer.create_manual(scale, offset, input_stats_dict)


class MultiRobotwinDataset(BaseDataset):
    def __init__(self,
            tasks: List[Dict],
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            balanced_sampling=True,
            use_data_augmentation=False,
            pc_xyz_noise_std=0.002,
            pc_rgb_noise_std=0.01,
            agent_pos_noise_std=0.0002,
            use_color_jitter=False,
            brightness_range=(-0.125, 0.125),
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            use_target_ee=False):
        super().__init__()

        if tasks is None or len(tasks) == 0:
            raise ValueError("MultiRobotwinDataset requires a non-empty tasks list")

        self.tasks = [_as_plain_dict(task) for task in tasks]
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.balanced_sampling = balanced_sampling
        self.use_target_ee = use_target_ee
        self.use_data_augmentation = use_data_augmentation
        self.pc_xyz_noise_std = pc_xyz_noise_std
        self.pc_rgb_noise_std = pc_rgb_noise_std
        self.agent_pos_noise_std = agent_pos_noise_std
        self.use_color_jitter = use_color_jitter
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.num_tasks = len(self.tasks)
        self.task_to_idx = {
            self._task_key(task, idx): idx
            for idx, task in enumerate(self.tasks)
        }

        cprint("--------------------------", "cyan")
        cprint(f"Multi-task RoboTwin dataset: {self.num_tasks} task entries", "cyan")
        cprint(f"Task one-hot mapping: {self.task_to_idx}", "cyan")
        cprint(f"Balanced sampling: {self.balanced_sampling}", "cyan")
        cprint("--------------------------", "cyan")

        self.task_datasets = []
        self._load_task_datasets()
        self._build_index()

    @staticmethod
    def _task_key(task, idx):
        task_name = task.get("task_name", str(idx))
        setting = task.get("setting", "")
        return f"{task_name}:{setting}"

    def _load_task_datasets(self):
        reference_shapes = None
        for task_idx, task in enumerate(self.tasks):
            zarr_path = task.get("zarr_path")
            if zarr_path is None:
                raise ValueError(f"Task entry {task_idx} is missing zarr_path")
            keys = ["state", "action", "point_cloud"]
            if self.use_target_ee:
                keys.append("target_ee")

            group = zarr.open(os.path.expanduser(zarr_path), "r")
            for key in keys:
                if key not in group["data"]:
                    raise KeyError(f"{zarr_path} is missing data/{key}")
            shapes = {key: tuple(group["data"][key].shape[1:]) for key in keys}
            if reference_shapes is None:
                reference_shapes = shapes
            elif shapes != reference_shapes:
                raise ValueError(
                    f"Incompatible zarr shapes for {zarr_path}: {shapes}, "
                    f"expected {reference_shapes}"
                )

            replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,
                val_ratio=self.val_ratio,
                seed=self.seed + task_idx)
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=self.max_train_episodes,
                seed=self.seed + task_idx)

            train_sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=train_mask)
            val_sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=~train_mask)

            self.task_datasets.append({
                "task_idx": task_idx,
                "task": task,
                "replay_buffer": replay_buffer,
                "train_mask": train_mask,
                "train_sampler": train_sampler,
                "val_sampler": val_sampler,
                "sampler": train_sampler,
            })

        self.data_shapes = reference_shapes

    def _build_index(self):
        self.indices = []
        sampler_lengths = [len(item["sampler"]) for item in self.task_datasets]
        if self.balanced_sampling:
            max_len = max(sampler_lengths)
            for task_idx, length in enumerate(sampler_lengths):
                if length == 0:
                    continue
                for local_idx in range(max_len):
                    self.indices.append((task_idx, local_idx % length))
        else:
            for task_idx, length in enumerate(sampler_lengths):
                for local_idx in range(length):
                    self.indices.append((task_idx, local_idx))

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.task_datasets = []
        for item in self.task_datasets:
            copied = item.copy()
            copied["sampler"] = copied["val_sampler"]
            val_set.task_datasets.append(copied)
        val_set._build_index()
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        actions = []
        agent_pos = []
        point_cloud = []
        for item in self.task_datasets:
            replay_buffer = item["replay_buffer"]
            if self.use_target_ee:
                actions.append(np.concatenate([
                    replay_buffer["action"],
                    replay_buffer["target_ee"],
                ], axis=-1))
            else:
                actions.append(replay_buffer["action"])
            agent_pos.append(replay_buffer["state"])
            point_cloud.append(replay_buffer["point_cloud"])

        normalizer = LinearNormalizer()
        normalizer["action"] = _fit_arrays_normalizer(actions, mode=mode, **kwargs)
        normalizer["agent_pos"] = _fit_arrays_normalizer(agent_pos, mode=mode, **kwargs)
        normalizer["point_cloud"] = _fit_arrays_normalizer(point_cloud, mode=mode, **kwargs)
        normalizer["task_onehot"] = _identity_normalizer(self.num_tasks)
        return normalizer

    def __len__(self) -> int:
        return len(self.indices)

    def _sample_to_data(self, sample, task_idx):
        agent_pos = sample["state"][:,].astype(np.float32)
        point_cloud = sample["point_cloud"][:,].astype(np.float32)
        joint_action = sample["action"].astype(np.float32)

        if self.use_target_ee:
            target_ee = sample["target_ee"][:,].astype(np.float32)
            action = np.concatenate([joint_action, target_ee], axis=-1)
        else:
            action = joint_action

        task_onehot = np.zeros((self.horizon, self.num_tasks), dtype=np.float32)
        task_onehot[:, task_idx] = 1.0

        return {
            "obs": {
                "point_cloud": point_cloud,
                "agent_pos": agent_pos,
                "task_onehot": task_onehot,
            },
            "action": action,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_idx, local_idx = self.indices[idx]
        item = self.task_datasets[task_idx]
        sample = item["sampler"].sample_sequence(local_idx)
        data = self._sample_to_data(sample, task_idx)

        if self.use_data_augmentation:
            if "point_cloud" in data["obs"]:
                point_cloud = data["obs"]["point_cloud"]
                xyz = point_cloud[..., :3]
                rgb = point_cloud[..., 3:]
                xyz_noisy = add_noise(
                    xyz,
                    noise_std=self.pc_xyz_noise_std,
                    clip_range=2*self.pc_xyz_noise_std,
                )
                rgb_noisy = add_noise(
                    rgb,
                    noise_std=self.pc_rgb_noise_std,
                    clip_range=2*self.pc_rgb_noise_std,
                )
                data["obs"]["point_cloud"] = np.concatenate([xyz_noisy, rgb_noisy], axis=-1)

            if "agent_pos" in data["obs"]:
                data["obs"]["agent_pos"] = add_noise(
                    data["obs"]["agent_pos"],
                    noise_std=self.agent_pos_noise_std,
                    clip_range=2*self.agent_pos_noise_std,
                )

        if self.use_color_jitter:
            if "point_cloud" in data["obs"]:
                point_cloud = data["obs"]["point_cloud"]
                xyz = point_cloud[..., :3]
                rgb = point_cloud[..., 3:]
                rgb_jittered = apply_color_jitter(
                    rgb,
                    brightness_range=self.brightness_range,
                    contrast_range=self.contrast_range,
                    saturation_range=self.saturation_range,
                )
                data["obs"]["point_cloud"] = np.concatenate([xyz, rgb_jittered], axis=-1)

        return dict_apply(data, torch.from_numpy)
