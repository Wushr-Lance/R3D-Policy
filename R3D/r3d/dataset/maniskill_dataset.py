from typing import Dict
import torch
import numpy as np
import os
import copy
from r3d.common.pytorch_util import dict_apply
from r3d.common.replay_buffer import ReplayBuffer
from r3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from r3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from r3d.dataset.base_dataset import BaseDataset
from termcolor import cprint


def add_noise(data, noise_std=0.01, clip_range=0.02):
    """
    Add clipped Gaussian noise to the input.
    
    Args:
        data: Input data (numpy array)
        noise_std: Standard deviation of Gaussian noise, default 0.01
        clip_range: Noise clipping range, default 0.02
    
    Returns:
        Data with added noise
    """
    if data is None:
        return None
    
    # Generate Gaussian noise matching input shape
    noise = np.random.normal(0, noise_std, data.shape)
    
    # Clip noise to prevent outliers
    noise = np.clip(noise, -clip_range, clip_range)
    
    # Add noise to data
    noisy_data = data + noise
    
    return noisy_data


def apply_color_jitter(rgb, brightness_range=(-0.125, 0.125), 
                       contrast_range=(0.5, 1.5), 
                       saturation_range=(0.5, 1.5)):
    """
    Apply color jitter augmentation to point cloud RGB colors.
    
    Args:
        rgb: RGB color data (numpy array), shape (..., 3), typically in [0, 1]
        brightness_range: Brightness adjustment range, default (-0.125, 0.125)
        contrast_range: Contrast adjustment range, default (0.5, 1.5)
        saturation_range: Saturation adjustment range, default (0.5, 1.5)
    
    Returns:
        RGB data after color jitter augmentation
    """
    if rgb is None:
        return None
    
    # Save original shape
    original_shape = rgb.shape
    # Flatten to (N, 3) for easier processing
    rgb_flat = rgb.reshape(-1, 3)
    
    # 1. Brightness adjustment
    brightness_delta = np.random.uniform(brightness_range[0], brightness_range[1])
    rgb_flat = rgb_flat + brightness_delta
    
    # 2. Contrast adjustment
    contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
    rgb_mean = rgb_flat.mean()
    rgb_flat = (rgb_flat - rgb_mean) * contrast_factor + rgb_mean
    
    # 3. Saturation adjustment
    saturation_factor = np.random.uniform(saturation_range[0], saturation_range[1])
    # Convert to grayscale
    gray = 0.299 * rgb_flat[:, 0:1] + 0.587 * rgb_flat[:, 1:2] + 0.114 * rgb_flat[:, 2:3]
    # Interpolate between grayscale and original color
    rgb_flat = gray + (rgb_flat - gray) * saturation_factor
    
    # Clip to valid range [0, 1]
    rgb_flat = np.clip(rgb_flat, 0.0, 1.0)
    
    # Reshape back to original shape
    rgb_jittered = rgb_flat.reshape(original_shape)
    
    return rgb_jittered


class ManiskillDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            use_data_augmentation=False,
            pc_xyz_noise_std=0.002,
            pc_rgb_noise_std=0.01,
            agent_pos_noise_std=0.0002,
            task_name=None,
            use_endpose=False,
            # Color jitter parameters
            use_color_jitter=False,
            brightness_range=(-0.125, 0.125),
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            use_target_ee=False
            ):
        super().__init__()

        # auxiliary task
        self.use_target_ee = use_target_ee

        if self.use_target_ee:
            cprint("--------------------------", "cyan")
            cprint("EE Auxiliary Task enabled: action dim = joint + ee", "cyan")
            cprint("--------------------------", "cyan")
        else:
            cprint("--------------------------", "yellow")
            cprint("EE Auxiliary Task disabled: action dim = joint only", "yellow")
            cprint("--------------------------", "yellow")
        
        self.use_data_augmentation = use_data_augmentation
        self.pc_xyz_noise_std = pc_xyz_noise_std
        self.pc_rgb_noise_std = pc_rgb_noise_std
        self.agent_pos_noise_std = agent_pos_noise_std

        # Color jitter parameters
        self.use_color_jitter = use_color_jitter
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        
        if use_data_augmentation:
            cprint("--------------------------", "green")
            cprint(f"Using noise data augmentation: pc_xyz_noise_std={pc_xyz_noise_std}, pc_rgb_noise_std={pc_rgb_noise_std}, agent_pos_noise_std={agent_pos_noise_std}", "green")
            cprint("--------------------------", "green")
        else:
            cprint("--------------------------", "red")
            cprint("Noise data augmentation disabled", "red")
            cprint("--------------------------", "red")
        
        if use_color_jitter:
            cprint("--------------------------", "magenta")
            cprint(f"Using color jitter augmentation: brightness={brightness_range}, contrast={contrast_range}, saturation={saturation_range}", "magenta")
            cprint("--------------------------", "magenta")
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'target_ee']) # 'img'
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        if self.use_target_ee:
            # Concatenate joint action and target_ee
            joint_action = self.replay_buffer['action']  # (N, T, 14)
            target_ee = self.replay_buffer['target_ee']  # (N, T, 14)
            combined_action = np.concatenate([joint_action, target_ee], axis=-1)  # (N, T, 28)

            data = {
                'action': combined_action,  # 28-dim
                'agent_pos': self.replay_buffer['state'][...,:],
                'point_cloud': self.replay_buffer['point_cloud'],
            }
        
        else:
            data = {
                'action': self.replay_buffer['action'],
                'agent_pos': self.replay_buffer['state'][...,:],
                'point_cloud': self.replay_buffer['point_cloud'],
            }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:].astype(np.float32)  # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:].astype(np.float32)  # (T, 1024, 6)

        # Keep only the first 1024 points
        if point_cloud.shape[1] > 1024:
            point_cloud = point_cloud[:, :1024, :]  # (T, 1024, 6)

        # Determine action dimension based on use_target_ee
        joint_action = sample['action'].astype(np.float32)  # (T, 14)
        
        if self.use_target_ee:
            target_ee = sample['target_ee'][:,].astype(np.float32)  # (T, 14)
            action = np.concatenate([joint_action, target_ee], axis=-1)  # (T, 28)
        else:
            action = joint_action  # (T, 14)

        data = {
            'obs': {
                'point_cloud': point_cloud,  # T, 1024, 6
                'agent_pos': agent_pos,      # T, 14
            },
            'action': action  # T, 14 or 28
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        if self.use_data_augmentation:
            if 'point_cloud' in data['obs']:
                # Point cloud format: (T, N, 6) where first 3 cols = xyz, last 3 = rgb
                point_cloud = data['obs']['point_cloud']
                
                # Separate xyz and rgb
                xyz = point_cloud[..., :3]  # (T, N, 3)
                rgb = point_cloud[..., 3:]  # (T, N, 3)
                
                # Add noise to xyz coordinates
                xyz_noisy = add_noise(
                    xyz, 
                    noise_std=self.pc_xyz_noise_std, 
                    clip_range=2*self.pc_xyz_noise_std
                )
                
                # Add noise to rgb colors
                rgb_noisy = add_noise(
                    rgb, 
                    noise_std=self.pc_rgb_noise_std, 
                    clip_range=2*self.pc_rgb_noise_std
                )
                
                # Recombine
                data['obs']['point_cloud'] = np.concatenate([xyz_noisy, rgb_noisy], axis=-1)
                
            if 'agent_pos' in data['obs']:
                # Add small Gaussian noise
                data['obs']['agent_pos'] = add_noise(
                    data['obs']['agent_pos'], 
                    noise_std=self.agent_pos_noise_std, 
                    clip_range=2*self.agent_pos_noise_std
                )
        
        # Apply color jitter if enabled
        if self.use_color_jitter:
            if 'point_cloud' in data['obs']:
                # Point cloud format: (T, N, 6) where first 3 cols = xyz, last 3 = rgb
                point_cloud = data['obs']['point_cloud']
                
                # Separate xyz and rgb
                xyz = point_cloud[..., :3]  # (T, N, 3)
                rgb = point_cloud[..., 3:]  # (T, N, 3)
                
                # Apply color jitter to rgb colors
                rgb_jittered = apply_color_jitter(
                    rgb,
                    brightness_range=self.brightness_range,
                    contrast_range=self.contrast_range,
                    saturation_range=self.saturation_range
                )
                
                # Recombine
                data['obs']['point_cloud'] = np.concatenate([xyz, rgb_jittered], axis=-1)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

