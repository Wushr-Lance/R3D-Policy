import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import numpy as np

from pytorch3d.ops import sample_farthest_points
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs
                 ):
        """Initialize PointNet encoder for XYZ+RGB point clouds.

        Args:
            in_channels (int): Feature size of input (3 or 6).
            out_channels (int): Output feature dimension.
            use_layernorm (bool): Whether to use LayerNorm after each MLP layer.
            final_norm (str): Normalization after final projection ('layernorm' or 'none').
            use_projection (bool): Whether to apply the final projection layer.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x, eval):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs
                 ):
        """Initialize PointNet encoder for XYZ-only point clouds.

        Args:
            in_channels (int): Feature size of input (must be 3).
            out_channels (int): Output feature dimension.
            use_layernorm (bool): Whether to use LayerNorm after each MLP layer.
            final_norm (str): Normalization after final projection ('layernorm' or 'none').
            use_projection (bool): Whether to apply the final projection layer.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

    def forward(self, x, eval):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class DP3Encoder(nn.Module):
    def __init__(self,
                 observation_space: Dict,
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 fps_random_config=None,
                 cat_on_token=False
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        state_mlp_size = (64, pointcloud_encoder_cfg['embed_dim'])

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        feature_mode = pointcloud_encoder_cfg.get('feature_mode', None)
        self.pc_encoder_extract_global_feature = feature_mode != 'pointsam'
        cprint(f"[DP3Encoder] extract_global_feature: {self.pc_encoder_extract_global_feature}", "yellow")
        
        # FPS randomness config - set defaults
        self.fps_random_config = fps_random_config or {
            'use_random': True,
            'random_start': True,
            'random_noise_scale': 0,
            'shuffle_output': True
        }

        self.cat_on_token = cat_on_token
        
        # Support for Uni3D encoder
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        elif pointnet_type == "uni3d":
            cprint(f"[DP3Encoder] Using Uni3D encoder", "yellow")
            # Default config for Uni3D encoder
            uni3d_config = {
                'pc_model': 'eva02_large_patch14_448',
                'pc_feat_dim': 1024,
                'embed_dim': out_channel,
                'group_size': 32,
                'num_group': 512,
                'patch_dropout': 0.5,
                'drop_path_rate': 0.2,
                'pretrained_pc': None,
                'pc_encoder_dim': 512,
                'use_pretrained_weights': False,
                'pretrained_weights_path': None,
            }
            
            # Override defaults with user-provided config
            if pointcloud_encoder_cfg:
                uni3d_config.update(pointcloud_encoder_cfg)
            
            # Add FPS randomness config
            uni3d_config['fps_random_config'] = self.fps_random_config
            
            self.extractor = Uni3DPointcloudEncoder(**uni3d_config)
            
            # Adjust output channels to match Uni3D output
            self.n_output_channels = uni3d_config['embed_dim']
            
        elif pointnet_type == "uni3d_pretrained":
            cprint(f"[DP3Encoder] Using pretrained Uni3D encoder", "yellow")
            # Config for pretrained Uni3D encoder
            uni3d_config = {
                'pc_model': 'eva02_large_patch14_448',
                'pc_feat_dim': 1024,
                'embed_dim': out_channel,
                'group_size': 32,
                'num_group': 512,
                'patch_dropout': 0.5,
                'drop_path_rate': 0.2,
                'pretrained_pc': None,
                'pc_encoder_dim': 512,
                'use_pretrained_weights': True,
                'pretrained_weights_path': 'Uni3D_large/model.pt',
            }
            
            # Override defaults with user-provided config
            if pointcloud_encoder_cfg:
                uni3d_config.update(pointcloud_encoder_cfg)
            
            # Add FPS randomness config
            uni3d_config['fps_random_config'] = self.fps_random_config
            
            self.extractor = Uni3DPointcloudEncoder(**uni3d_config)
            
            # Adjust output channels to match Uni3D output
            self.n_output_channels = uni3d_config['embed_dim']

        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]
        
        if self.cat_on_token:
            self.n_output_channels = uni3d_config['embed_dim']
        else:
            self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] Final output dim: {self.n_output_channels}", "yellow")


    def forward(self, observations: Dict, eval=False) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        # Handle different encoder types
        if self.pointnet_type in ["uni3d", "uni3d_pretrained"]:
            # Uni3D encoder requires 6-channel input (xyz + rgb)
            if points.shape[-1] == 3:
                # If only xyz, pad with zero colors
                colors = torch.zeros_like(points)
                points = torch.cat([points, colors], dim=-1)
            elif points.shape[-1] > 6:
                # If more than 6 channels, keep only the first 6
                points = points[..., :6]

        # points: B * N * (3 or 6)
        # PointSAM: pc_embedding: [B*n_obs_steps, num_patches, embed_dim], pc_pe: [B*n_obs_steps, num_patches, embed_dim]
        if not self.pc_encoder_extract_global_feature:
            pn_feat, pc_pe = self.extractor(points, eval)
        else:
            pn_feat = self.extractor(points, eval)

        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # [B*n_obs_steps, embed_dim]

        if not self.pc_encoder_extract_global_feature:
            if self.cat_on_token:
                state_feat = state_feat.unsqueeze(1)
            else:
                state_feat = state_feat.unsqueeze(1).expand(-1, pn_feat.shape[1], -1)

        # Prepare feature list for concatenation
        features = [pn_feat, state_feat]
        if self.cat_on_token:
            final_feat = torch.cat(features, dim=-2)
        else:
            final_feat = torch.cat(features, dim=-1)
        if not self.pc_encoder_extract_global_feature:
            return final_feat, pc_pe
        return final_feat


    def output_shape(self):
        return self.n_output_channels


# =============================================================================
# Uni3D encoder components
# =============================================================================

def fps(data, number, use_random=True, random_start=True, random_noise_scale=0, shuffle_output=True):
    '''
    Enhanced FPS with randomness options
    Args:
        data: B N 3 (or more channels)
        number: int, number of points to sample
        use_random: bool, whether to enable randomness
        random_start: bool, whether to use random starting point
        random_noise_scale: float, scale of random noise added to distances
        shuffle_output: bool, whether to randomly shuffle the output order
    '''
    xyz_coordinates = data[:, :, :3]
    B, N, _ = xyz_coordinates.shape
    
    if not use_random:
        # Original deterministic FPS
        _, fps_idx = sample_farthest_points(xyz_coordinates, K=number)
    else:
        # Enhanced FPS with randomness
        if random_start:
            # Randomly select starting points for each batch
            start_indices = torch.randint(0, N, (B,), device=data.device)
            
            # Create modified coordinates with random starting points moved to front
            modified_xyz = xyz_coordinates.clone()
            for b in range(B):
                start_idx = start_indices[b]
                # Swap the randomly selected point to the first position
                modified_xyz[b, [0, start_idx]] = modified_xyz[b, [start_idx, 0]]
        else:
            modified_xyz = xyz_coordinates
        
        if random_noise_scale > 0:
            # Add small random noise to coordinates for FPS computation
            noise = torch.randn_like(modified_xyz) * random_noise_scale
            noisy_xyz = modified_xyz + noise
        else:
            noisy_xyz = modified_xyz
        
        # Perform FPS on modified/noisy coordinates
        _, fps_idx = sample_farthest_points(noisy_xyz, K=number)
        
        # If we used random start, we need to map back the indices
        if random_start:
            for b in range(B):
                start_idx = start_indices[b]
                # Map indices back to original positions
                mask_0 = fps_idx[b] == 0
                mask_start = fps_idx[b] == start_idx
                fps_idx[b][mask_0] = start_idx
                fps_idx[b][mask_start] = 0
        
        if shuffle_output:
            # Randomly shuffle the order of selected indices
            for b in range(B):
                perm = torch.randperm(number, device=data.device)
                fps_idx[b] = fps_idx[b][perm]
    
    # Gather the selected points using the (possibly randomized) indices
    fps_data = torch.gather(
        data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    
    return fps_data


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    B, N, _ = batch_pc.shape
    result = torch.clone(batch_pc)
    for b in range(B):
        dropout_ratio = torch.rand(1).item() * max_dropout_ratio  # 0 ~ 0.875
        drop_idx = torch.where(torch.rand(N) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            result[b, drop_idx, :] = batch_pc[b, 0, :].unsqueeze(0)  # set to the first point
    return result


class PatchDropout(nn.Module):
    """
    Patch dropout for Uni3D
    https://arxiv.org/abs/2212.00794
    """
    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Group(nn.Module):
    """Point cloud grouping module with configurable FPS randomness."""
    def __init__(self, num_group, group_size, fps_random_config=None):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.fps_random_config = fps_random_config or {}
        cprint(f"[Group] FPS randomness config: {fps_random_config}", "cyan")

    def forward(self, xyz, color):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        
        # Enhanced FPS with randomness for centers
        center = fps(xyz, self.num_group, **self.fps_random_config) # B G 3
        
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features


class KNNGrouper(nn.Module):
    """Group points based on K nearest neighbors.

    A number of points are sampled as centers by farthest point sampling (FPS).
    Each group is formed by the center and its k nearest neighbors.
    """

    def __init__(self, num_groups, group_size, radius=None, centralize_features=False, fps_random_config=None):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.radius = radius
        self.centralize_features = centralize_features
        self.fps_random_config = fps_random_config or {}
        cprint(f"[Group] FPS randomness config: {fps_random_config}", "cyan")

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, use_fps=True):
        """
        Args:
            xyz: [B, N, 3]. Input point clouds.
            features: [B, N, C]. Point features.
            use_fps: bool. Whether to use farthest point sampling.
                If not, `xyz` should already be sampled by FPS.

        Returns:
            dict: {
                features: [B, G, K, 3 + C]. Group features.
                centers: [B, G, 3]. Group centers.
                knn_idx: [B, G, K]. The indices of k nearest neighbors.
            }
        """
        batch_size, num_points, _ = xyz.shape
        with torch.no_grad():
            centers = fps(xyz, self.num_groups, **self.fps_random_config) # B G 3
            _, knn_idx = knn_points(centers, xyz, self.group_size)  # [B, G, K]

        batch_offset = torch.arange(batch_size, device=xyz.device) * num_points
        batch_offset = batch_offset.reshape(-1, 1, 1)
        knn_idx_flat = (knn_idx + batch_offset).reshape(-1)  # [B * G * K]

        nbr_xyz = xyz.reshape(-1, 3)[knn_idx_flat]
        nbr_xyz = nbr_xyz.reshape(batch_size, self.num_groups, self.group_size, 3)
        nbr_xyz = nbr_xyz - centers.unsqueeze(2)  # [B, G, K, 3]
        # NOTE: Follow PointNext to normalize the relative position
        if self.radius is not None:
            nbr_xyz = nbr_xyz / self.radius

        nbr_feats = features.reshape(-1, features.shape[-1])[knn_idx_flat]
        nbr_feats = nbr_feats.reshape(
            batch_size, self.num_groups, self.group_size, features.shape[-1]
        )

        group_feats = torch.cat([nbr_xyz, nbr_feats], dim=-1)
        return dict(
            features=group_feats, centers=centers, knn_idx=knn_idx
        )


class NNGrouper(nn.Module):
    """Group points based on the nearest neighbors."""

    def __init__(self, num_groups, fps_random_config=None):
        super().__init__()
        self.num_groups = num_groups
        self.fps_random_config = fps_random_config or {}
        cprint(f"[Group] FPS randomness config: {fps_random_config}", "cyan")

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        with torch.no_grad():
            centers = fps(xyz, self.num_groups, **self.fps_random_config) # B G 3
            _, nn_idx = knn_points(xyz, centers, 1)  # [B, N, 1]

        # Compute the relative position of each point to its nearest center
        nn_idx = nn_idx.squeeze(-1)
        nbr_xyz = xyz - batch_index_select(centers, nn_idx, dim=1)  # [B, N, 3]

        # Normalize the relative position
        dist = torch.linalg.norm(nbr_xyz, dim=-1, keepdim=True, ord=2)
        nbr_xyz = nbr_xyz / torch.clamp(dist, min=1e-8)

        group_feats = torch.cat([nbr_xyz, dist, features], dim=-1)
        return dict(features=group_feats, centers=centers, nn_idx=nn_idx)


class PatchEncoder(nn.Module):
    """Encode point patches following the PointNet structure for segmentation."""

    def __init__(self, in_channels, out_channels, hidden_dims: list[int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # NOTE: The original Uni3D implementation uses BatchNorm1d, while we use LayerNorm.
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[0] * 2, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], out_channels),
        )

    def forward(self, point_patches: torch.Tensor):
        # point_patches: [B, L, K, C_in]
        x = self.conv1(point_patches)
        y = torch.max(x, dim=-2, keepdim=True).values
        x = torch.cat([y.expand_as(x), x], dim=-1)
        x = self.conv2(x)  # [B, L, K, C_out]
        y = torch.max(x, dim=-2).values  # [B, L, C_out]
        return y



class Block(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        # Follow timm.layers.mlp
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_channels),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # PreLN. Follow timm.models.vision_transformer
        return x + self.mlp(self.norm(x))


class PatchEmbedNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_patches, fps_random_config=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_dim = hidden_dim or out_channels
        self.num_patches = num_patches
        self.grouper = NNGrouper(num_patches, fps_random_config)
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.blocks1 = nn.Sequential(
            *[Block(hidden_dim, hidden_dim, hidden_dim) for _ in range(3)]
        )
        self.blocks2 = nn.Sequential(
            *[Block(hidden_dim, hidden_dim, hidden_dim) for _ in range(3)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, coords: torch.tensor, features: torch.tensor):
        patches = self.grouper(coords, features)
        patch_features = patches["features"]  # [B, N, D]
        nn_idx = patches["nn_idx"]  # [B, N]
        x = self.in_proj(patch_features)
        x = self.blocks1(x)  # [B, N, D]
        y = x.new_zeros(x.shape[0], self.grouper.num_groups, x.shape[-1])
        y.scatter_reduce_(
            1, nn_idx.unsqueeze(-1).expand_as(x), x, "amax", include_self=False
        )
        x = self.blocks2(y)
        x = self.norm(x)
        x = self.out_proj(x)
        patches["embeddings"] = x
        return patches


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_patches,
        patch_size,
        radius: float = None,
        centralize_features=False,
        fps_random_config=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.grouper = KNNGrouper(
            num_patches,
            patch_size,
            radius=radius,
            centralize_features=centralize_features,
            fps_random_config=fps_random_config
        )

        self.patch_encoder = PatchEncoder(in_channels, out_channels, [128, 512])
        self.fps_random_config = fps_random_config or {}

    def forward(self, coords: torch.Tensor, features: torch.Tensor):
        patches = self.grouper(coords, features)
        patch_features = patches["features"]  # [B, L, K, C_in]
        x = self.patch_encoder(patch_features)
        patches["embeddings"] = x
        return patches


def knn_points(
    query: torch.Tensor,
    key: torch.Tensor,
    k: int,
    sorted: bool = False,
    transpose: bool = False,
):
    """Compute k nearest neighbors.

    Args:
        query: [B, N1, D], query points. [B, D, N1] if @transpose is True.
        key:  [B, N2, D], key points. [B, D, N2] if @transpose is True.
        k: the number of nearest neighbors.
        sorted: whether to sort the results
        transpose: whether to transpose the last two dimensions.

    Returns:
        torch.Tensor: [B, N1, K], distances to the k nearest neighbors in the key.
        torch.Tensor: [B, N1, K], indices of the k nearest neighbors in the key.
    """
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    # Compute pairwise distances, [B, N1, N2]
    distance = torch.cdist(query, key)
    if k == 1:
        knn_dist, knn_ind = torch.min(distance, dim=2, keepdim=True)
    else:
        knn_dist, knn_ind = torch.topk(distance, k, dim=2, largest=False, sorted=sorted)
    return knn_dist, knn_ind


def batch_index_select(input, index, dim):
    """The batched version of `torch.index_select`.

    Args:
        input (torch.Tensor): [B, ...]
        index (torch.Tensor): [B, N] or [B]
        dim (int): the dimension to index

    """

    if index.dim() == 1:
        index = index.unsqueeze(1)
        squeeze_dim = True
    else:
        assert (
            index.dim() == 2
        ), "index is expected to be 2-dim (or 1-dim), but {} received.".format(
            index.dim()
        )
        squeeze_dim = False
    assert input.size(0) == index.size(0), "Mismatched batch size: {} vs {}".format(
        input.size(0), index.size(0)
    )
    views = [1 for _ in range(input.dim())]
    views[0] = index.size(0)
    views[dim] = index.size(1)
    expand_shape = list(input.shape)
    expand_shape[dim] = -1
    index = index.view(views).expand(expand_shape)
    out = torch.gather(input, dim, index)
    if squeeze_dim:
        out = out.squeeze(1)
    return out


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [-1,1]."""
        # assuming coords are in [-1, 1] and have d_1 x ... x d_n x D shape
        coords = coords @ self.positional_encoding_gaussian_matrix
        # TODO: Why using 2 * np.pi?
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: shape (..., coord_dim), normalized coordinates in [-1, 1].

        Returns:
            torch.Tensor: shape (..., num_pos_feats), positional encoding.
        """
        if (coords < -1 - 1e-6).any() or (coords > 1 + 1e-6).any():
            print("Bounds: ", (coords.min(), coords.max()))
            raise ValueError(f"Input coordinates must be normalized to [-1, 1].")
        # TODO: whether to convert to float?
        return self._pe_encoding(coords)


class Uni3DPointcloudEncoder(nn.Module):
    """
    Uni3D point cloud encoder.
    Supports both pretrained weight loading and training from scratch.
    """
    def __init__(self,
                 pc_model='eva02_large_patch14_448',
                 pc_feat_dim=1024,
                 embed_dim=1024,
                 group_size=32,
                 num_group=512,
                 patch_dropout=0.5,
                 drop_path_rate=0.2,
                 pretrained_pc=None,
                 pc_encoder_dim=512,
                 use_pretrained_weights=False,
                 pretrained_weights_path=None,
                 normalization_type="batch_norm",
                 feature_mode="pointsam",
                 extract_global_feature=True,
                 fps_random_config=None,
                 **kwargs):
        super().__init__()

        # vit backbone
        self.transformer = timm.create_model(pc_model, checkpoint_path=pretrained_pc, drop_path_rate=drop_path_rate)
        self.transformer_dim = self.transformer.embed_dim
        self.embed_dim = embed_dim
        self.num_group = num_group
        self.use_pretrained_weights = use_pretrained_weights

        self.patch_embed = PatchEmbed(in_channels=6, out_channels=512, num_patches=num_group, patch_size=group_size, fps_random_config=fps_random_config)

        # 7 = xyz + rgb + dist
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.transformer_dim)
        )

        self.extract_global_feature = feature_mode != 'pointsam'

        # for pointsam output pc_pe
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Patch dropout
        self.patch_dropout = PatchDropout(patch_dropout, exclude_first_token=(feature_mode=="cls")) if patch_dropout > 0. else nn.Identity()
        # Project transformer output to embedding dim
        self.out_proj = nn.Linear(self.transformer_dim, self.embed_dim)
        self.patch_proj = nn.Linear(self.patch_embed.out_channels, self.transformer_dim)
        self.feature_mode = feature_mode

        if self.feature_mode == "cls":
            cprint(f"[Uni3DPointcloudEncoder] use cls token", "red")

            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.transformer_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.transformer_dim))
        elif self.feature_mode == "max_pooling":
            cprint(f"[Uni3DPointcloudEncoder] use max pooling", "red")
        else:  # pointsam
            cprint(f"[Uni3DPointcloudEncoder] use pointsam, do not extract global feature", "red")

        # Load pretrained weights if specified
        if use_pretrained_weights:
            self._load_pretrained_weights_selective(pretrained_weights_path, normalization_type)
        else:
            cprint(f"[Uni3DPointcloudEncoder] Using random initialization (training from scratch)", "red")

    def _load_pretrained_weights_selective(self, pretrained_weights_path, normalization_type):
        """
        Selectively load pretrained weights based on normalization_type.

        Args:
            pretrained_weights_path: Path to pretrained weights
            normalization_type: Normalization type ("batch_norm", "layer_norm", "none")
        """
        load_weight_path = pretrained_weights_path
        if not os.path.exists(load_weight_path):
            cprint(f"[Uni3DPointcloudEncoder] Pretrained weights file not found: {load_weight_path}", "red")
            return

        # Load pretrained weights
        from safetensors.torch import load_file
        checkpoint = load_file(os.path.join(load_weight_path, "model.safetensors"))
        # Remap key names
        processed_state_dict = {}
        for key in list(checkpoint.keys()):
            if key.startswith('pc_encoder.'):
                new_key = key.replace('pc_encoder.', '')
                processed_state_dict[new_key] = checkpoint[key]
        missing_keys, unexpected_keys = self.load_state_dict(processed_state_dict, strict=False)
        cprint(f"  Missing keys: {missing_keys}", "yellow")
        cprint(f"  Unexpected keys: {unexpected_keys}", "yellow")

        cprint(f"[Uni3DPointcloudEncoder] Pretrained weights loaded: {load_weight_path}", "red")

    def forward(self, pcd, eval):
        # Apply point cloud dropout (data augmentation)
        if not eval:
            pcd = random_point_dropout(pcd, max_dropout_ratio=0.8)

        pts = pcd[..., :3].contiguous()
        colors = pcd[..., 3:].contiguous()
        # Group points into patches and get embeddings
        patches = self.patch_embed(pts, colors)
        if isinstance(patches, list):
            patch_embed = patches[-1]["embeddings"]
            centers = patches[-1]["centers"]
        else:
            patch_embed = patches["embeddings"]  # [B, L, D]
            centers = patches["centers"]  # [B, L, 3]
        patch_embed = self.patch_proj(patch_embed)

        # Add positional embedding
        pos_embed = self.pos_embed(centers)

        if self.feature_mode == "cls":

            # prepare cls
            cls_tokens = self.cls_token.expand(patch_embed.size(0), -1, -1)  
            cls_pos = self.cls_pos.expand(pos_embed.size(0), -1, -1) 

            # final input
            patch_embed = torch.cat((cls_tokens, patch_embed), dim=1)
            pos_embed = torch.cat((cls_pos, pos_embed), dim=1)
        
        x = patch_embed + pos_embed
        # patch dropout
        if not eval:
            x = self.patch_dropout(x)
            x = self.transformer.pos_drop(x)

        for block in self.transformer.blocks:
            x = block(x)

        if self.extract_global_feature:

            # Extract features based on whether CLS token is used
            if self.feature_mode == "cls":
                # Use CLS token (first token) for classification
                x = self.transformer.norm(x[:, 0, :])
            elif self.feature_mode == "max_pooling":
                # Use global max pooling over all patch tokens
                x = self.transformer.norm(torch.max(x, dim=1)[0])
        else: 
            # pointsam, do not extract global feature
            x = self.transformer.norm(x)
        
        x = self.transformer.fc_norm(x)
        x = self.out_proj(x)

        if not self.extract_global_feature:
            pc_pe = self.pe_layer(centers)
            return x, pc_pe
        else:
            return x