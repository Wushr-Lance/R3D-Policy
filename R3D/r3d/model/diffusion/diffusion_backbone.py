from typing import Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from termcolor import cprint
from r3d.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from r3d.model.diffusion.positional_embedding import SinusoidalPosEmb



logger = logging.getLogger(__name__)


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type


class OneWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                OneWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        global_feature_embeded: Tensor,
        global_pe: Tensor,
        sample_embedded: Tensor,
        sample_pe: Tensor,
        attn_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          pc_embedding (torch.Tensor): point cloud to attend to. Should be shape
            B x N_pc_tokens x embedding_dim.
          pc_pe (torch.Tensor): the positional encoding to add to the point cloud. 
            Must have the same shape as pc_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed pc_embedding
        """
        # Prepare queries
        queries = sample_embedded
        keys = global_feature_embeded

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=sample_pe,
                key_pe=global_pe,
                attn_mask=attn_mask
            )

        return queries


class OneWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )


    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, attn_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        q = queries + query_pe
        attn_out = self.self_attn(q=q, k=q, v=queries, attn_mask=attn_mask)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        if keys.shape[1] != key_pe.shape[1]: # cat on token
            k = keys.clone()
            k[:, :key_pe.shape[1], :] += key_pe
        else:
            k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # we dont need back transformer
        # # Cross attention block, image embedding attending to tokens
        # q = queries + query_pe
        # k = keys + key_pe
        # attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        # keys = keys + attn_out
        # keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor = None) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_patches
        attn = attn / math.sqrt(c_per_head)

        # Apply mask if provided
        if attn_mask is not None:
            # Handle different mask dimensions
            if attn_mask.dim() == 2:
                # [N_queries, N_keys] -> [1, 1, N_queries, N_keys]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [B, N_queries, N_keys] -> [B, 1, N_queries, N_keys]
                attn_mask = attn_mask.unsqueeze(1)
            
            # Mask out positions: True = masked = -inf
            attn = attn.masked_fill(attn_mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


# https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/common.py#L13
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))



class CrossAttention(nn.Module):
    def __init__(self, in_dim, cond_dim, out_dim):
        super().__init__()
        self.query_proj = nn.Linear(in_dim, out_dim)
        self.key_proj = nn.Linear(cond_dim, out_dim)
        self.value_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, x, cond):
        # x: [batch_size, t_act, in_dim]
        # cond: [batch_size, t_obs, cond_dim]

        # Project x and cond to query, key, and value
        query = self.query_proj(x)  # [batch_size, horizon, out_dim]
        key = self.key_proj(cond)   # [batch_size, horizon, out_dim]
        value = self.value_proj(cond)  # [batch_size, horizon, out_dim]


        # Compute attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, horizon, n_obs_steps]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention
        attn_output = torch.matmul(attn_weights, value)  # [batch_size, horizon, out_dim]
        
        return attn_output
    

class ConditionalResidualBlock1D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8,
                 condition_type='film'):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
            Conv1dBlock(out_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
        ])

        
        self.condition_type = condition_type

        cond_channels = out_channels
        if condition_type == 'film': # FiLM modulation https://arxiv.org/abs/1709.07871
            # predicts per-channel scale and bias
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'add':
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'cross_attention_add':
            self.cond_encoder = CrossAttention(in_channels, cond_dim, out_channels)
        elif condition_type == 'cross_attention_film':
            cond_channels = out_channels * 2
            self.cond_encoder = CrossAttention(in_channels, cond_dim, cond_channels)
        elif condition_type == 'mlp_film':
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_dim),
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'one_way_transformer':
            pass
        else:
            raise NotImplementedError(f"condition_type {condition_type} not implemented")
        
        self.out_channels = out_channels
        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)  
        if cond is not None:      
            if self.condition_type == 'film':
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == 'add':
                embed = self.cond_encoder(cond)
                out = out + embed
            elif self.condition_type == 'cross_attention_add':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)
                embed = embed.permute(0, 2, 1) # [batch_size, out_channels, horizon]
                out = out + embed
            elif self.condition_type == 'cross_attention_film':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)
                embed = embed.permute(0, 2, 1)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == 'mlp_film':
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == 'one_way_transformer':
                pass
            else:
                raise NotImplementedError(f"condition_type {self.condition_type} not implemented")
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        condition_type='film',
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        transformer_config=None,
        pe_type='learnable',  # 'learnable' or 'sinusoidal'
        use_target_ee = False,
        cat_on_token=False
        ):
        super().__init__()
        self.cat_on_token = cat_on_token
        self.use_target_ee = use_target_ee
        self.condition_type = condition_type
        self.input_dim = input_dim
        self.pe_type = pe_type
        
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition
        
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            if self.cat_on_token:
                cond_dim = global_cond_dim
            else:
                cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        # Initialize OneWayTransformer if condition_type is one_way_transformer
        self.one_way_transformer = None
        self.sample_proj = None
        self.global_cond_proj = None
        self.output_proj = None
        self.temporal_pe_obs = None
        self.temporal_pe_horizon = None
        
        if condition_type == 'one_way_transformer':
            if transformer_config is None:
                raise ValueError("transformer_config must be provided when condition_type is 'one_way_transformer'")
            
            embedding_dim = transformer_config['embedding_dim']
            depth = transformer_config['depth']
            num_heads = transformer_config['num_heads']
            mlp_dim = transformer_config['mlp_dim']
            
            # Initialize OneWayTransformer
            self.one_way_transformer = OneWayTransformer(
                depth=depth,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
            )
            
            # Projection layers
            # cond_dim includes diffusion_step_embed_dim + global_cond_dim
            
            if self.use_target_ee:
                self.global_cond_proj = nn.Linear(cond_dim, embedding_dim)
                ee_dim = input_dim // 2
                joint_dim = input_dim - ee_dim
                
                # Joint and EE share the same embedding dimension
                self.joint_proj = nn.Linear(joint_dim, embedding_dim)
                self.ee_proj = nn.Linear(ee_dim, embedding_dim)

                # Separate output projections
                self.joint_output_proj = nn.Linear(embedding_dim, joint_dim)
                self.ee_output_proj = nn.Linear(embedding_dim, ee_dim)
            else:
                self.sample_proj = nn.Linear(input_dim, embedding_dim)
                self.global_cond_proj = nn.Linear(cond_dim, embedding_dim)
                self.output_proj = nn.Linear(embedding_dim, input_dim)
            
            # Positional Encoding modules
            # We'll initialize these with a max length, and use slicing at forward time
            max_n_obs_steps = transformer_config.get('max_n_obs_steps', 2)
            max_horizon = transformer_config.get('max_horizon', 16)

            if pe_type == 'learnable':
                # Learnable positional encoding
                self.temporal_pe_obs = nn.Parameter(torch.zeros(1, max_n_obs_steps, embedding_dim))
                self.temporal_pe_horizon = nn.Parameter(torch.zeros(1, max_horizon, embedding_dim))
                nn.init.normal_(self.temporal_pe_obs, std=0.02)
                nn.init.normal_(self.temporal_pe_horizon, std=0.02)
            elif pe_type == 'sinusoidal':
                # Fixed sinusoidal positional encoding
                self.temporal_pe_obs = self._create_sinusoidal_pe(max_n_obs_steps, embedding_dim)
                self.temporal_pe_horizon = self._create_sinusoidal_pe(max_horizon, embedding_dim)
            else:
                raise ValueError(f"pe_type must be 'learnable' or 'sinusoidal', got {pe_type}")

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _create_sinusoidal_pe(self, max_len, embedding_dim):
        """
        Create fixed sinusoidal positional encoding.
        Args:
            max_len: Maximum sequence length
            embedding_dim: Encoding dimension
        Returns:
            pe: [1, max_len, embedding_dim] positional encoding
        """
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, embedding_dim]
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, pc_pe=None, n_obs_steps=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        timestep_embed = self.diffusion_step_encoder(timesteps) # cat_on_token: 256
        if global_cond is not None:
            if 'cross_attention' in self.condition_type or self.condition_type == 'one_way_transformer':
                if self.cat_on_token:
                    timestep_embed = timestep_embed.unsqueeze(1)
                else:
                    timestep_embed = timestep_embed.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
            if self.cat_on_token:
                global_feature = torch.cat([timestep_embed, global_cond], axis=-2)
            else:
                global_feature = torch.cat([timestep_embed, global_cond], axis=-1)
        else:
            global_feature = timestep_embed

        # Handle one_way_transformer branch
        if self.condition_type == 'one_way_transformer':
            # sample shape: [B, action_dim, horizon]
            # Need to rearrange to [B, horizon, action_dim]
            sample_rearranged = sample.permute(0, 2, 1)  # [B, horizon, action_dim]
            
            global_feature_embedded = self.global_cond_proj(global_feature)  # [B, n_obs_steps*num_tokens + 1, embedding_dim]
            
            # Build positional encodings
            batch_size = sample.shape[0]
            horizon = sample_rearranged.shape[1]
            if self.cat_on_token:
                num_tokens = (global_feature_embedded.shape[1] - 1) // n_obs_steps
                num_patches = num_tokens - 1
            else:
                num_patches = global_feature_embedded.shape[1] // n_obs_steps
                num_tokens = num_patches

            if self.use_target_ee:
                ee_dim = self.input_dim // 2
                joint_dim =self.input_dim - ee_dim
                
                # Separate joint and ee
                joint_sample = sample_rearranged[:, :, :joint_dim]  # [B, horizon, 14]
                ee_sample = sample_rearranged[:, :, joint_dim:]     # [B, horizon, 14]
                
                # Embed joint and ee separately
                joint_embedded = self.joint_proj(joint_sample)  # [B, horizon, embedding_dim]
                ee_embedded = self.ee_proj(ee_sample)           # [B, horizon, embedding_dim]
                
                # Concatenate along sequence dim so joint and ee become separate tokens
                sample_embedded = torch.cat([joint_embedded, ee_embedded], dim=1)

                # Both joint and ee share the same temporal positional encoding
                if self.pe_type == 'learnable':
                    sample_pe_joint = self.temporal_pe_horizon[:, :horizon, :].expand(batch_size, -1, -1)
                    sample_pe_ee = self.temporal_pe_horizon[:, :horizon, :].expand(batch_size, -1, -1)
                else:
                    sample_pe_joint = self.temporal_pe_horizon[:, :horizon, :].expand(batch_size, -1, -1).to(sample.device)
                    sample_pe_ee = self.temporal_pe_horizon[:, :horizon, :].expand(batch_size, -1, -1).to(sample.device)
                
                sample_pe = torch.cat([sample_pe_joint, sample_pe_ee], dim=1)  # [B, 2*horizon, embedding_dim]

                # Create attention mask
                # mask shape: [2*horizon, 2*horizon]
                # First horizon tokens = joint, last horizon tokens = ee
                # Joint can only attend to joint; ee can attend to both joint + ee
                seq_len = 2 * horizon
                attn_mask = torch.zeros(seq_len, seq_len, device=sample.device, dtype=torch.bool)
                
                # Block joint from attending to ee
                attn_mask[:horizon, horizon:] = True
                
                # ee can see everything (default False)
            else:

                # Project to embedding_dim
                sample_embedded = self.sample_proj(sample_rearranged)  # [B, horizon, embedding_dim]
            
            
            
            
                # Build sample_pe: [B, horizon, embedding_dim]
                if self.pe_type == 'learnable':
                    sample_pe = self.temporal_pe_horizon[:, :horizon, :].expand(batch_size, -1, -1)
                else:  # sinusoidal
                    sample_pe = self.temporal_pe_horizon[:, :horizon, :].expand(batch_size, -1, -1).to(sample.device)

                attn_mask = None

            # Build global_pe: [B, n_obs_steps*num_patches, embedding_dim]
            # Combines temporal (n_obs_steps) and spatial (pc_pe) information
            if self.pe_type == 'learnable':
                temporal_pe_obs_slice = self.temporal_pe_obs[:, :n_obs_steps, :].expand(batch_size, -1, -1)
            else:  # sinusoidal
                temporal_pe_obs_slice = self.temporal_pe_obs[:, :n_obs_steps, :].expand(batch_size, -1, -1).to(sample.device)
            
            # Repeat temporal encoding num_patches times
            temporal_pe_expanded = temporal_pe_obs_slice.repeat_interleave(num_tokens, dim=1)
            
            # Add temporal and spatial (pc_pe) encodings
            global_pe = temporal_pe_expanded.clone()
            global_pe[:, :(n_obs_steps*num_patches), :] = temporal_pe_expanded[:, :(n_obs_steps*num_patches), :] +  pc_pe # if cat on token [B, n_obs_steps*num_tokens + 1, embedding_dim]
            # global_pe = temporal_pe_expanded + pc_pe  # [B, n_obs_steps*num_patches, embedding_dim]
            
            # Apply OneWayTransformer
            # point_embedding (queries) = sample_embedded
            # pc_embedding (keys) = global_feature_embedded
            output = self.one_way_transformer(
                global_feature_embeded=global_feature_embedded,
                global_pe=global_pe,
                sample_embedded=sample_embedded,
                sample_pe=sample_pe,
                attn_mask=attn_mask
            )  # [B, horizon, embedding_dim]

            # Separate joint and ee outputs
            if self.use_target_ee:
                joint_output = output[:, :horizon, :]  # [B, horizon, embedding_dim]
                ee_output = output[:, horizon:, :]     # [B, horizon, embedding_dim]
                
                # Project back
                joint_pred = self.joint_output_proj(joint_output)  # [B, horizon, 14]
                ee_pred = self.ee_output_proj(ee_output)            # [B, horizon, 14]
                
                output = torch.cat([joint_pred, ee_pred], dim=-1)  # [B, horizon, 28]
            else:
                output = self.output_proj(output)
            
            # # Project back to action_dim
            # output = self.output_proj(output)  # [B, horizon, action_dim]
            
            # Rearrange back to [B, action_dim, horizon]
            output = output.permute(0, 2, 1)  # [B, action_dim, horizon]
            
            # Rearrange to [B, horizon, action_dim] for final output
            x = einops.rearrange(output, 'b t h -> b h t')
            
            return x

        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            if self.use_down_condition:
                x = resnet(x, global_feature)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x)
            h.append(x)
            x = downsample(x)


        for mid_module in self.mid_modules:
            if self.use_mid_condition:
                x = mid_module(x, global_feature)
            else:
                x = mid_module(x)


        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            if self.use_up_condition:
                x = resnet(x, global_feature)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x)
            x = upsample(x)


        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x

