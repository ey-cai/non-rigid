# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import omegaconf
import torch
import torch.nn as nn
import torch_geometric.data as tgd
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from typing import Optional

from non_rigid.nets.dgcnn import DGCNN
from non_rigid.nets.pn2 import PN2Dense, PN2DenseParams
from non_rigid.models.dit.relative_encoding import RotaryPositionEncoding3D, MultiheadRelativeAttentionWrapper

from functools import partial

torch.set_printoptions(precision=8, sci_mode=True)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                         Reference Frame Predictor                             #
#################################################################################

class ReferenceFramePredictor(nn.Module):
    """
    Reference frame predictor for TAX3D.
    """
    def __init__(self, hidden_size, num_heads, **block_kwargs):
        super().__init__()
        self.input_mlp = nn.Conv1d(
            3,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = CrossAttention(
            dim_x=hidden_size,
            dim_y=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )
        self.output_mlp = nn.Conv1d(
            hidden_size,
            4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        """
        x: (B, C, N)
        """
        B, C, N = x.shape
        # TODO: this module should be able to visualize attention weights and residuals easily

        # input point cloud attends to itself, with final output (logit, residual)
        x_embed = self.input_mlp(x)
        x_embed = x_embed.permute(0, 2, 1)
        x_embed = self.norm1(x_embed)
        x_embed = self.attn(x_embed, x_embed)
        x_embed = self.output_mlp(x_embed.permute(0, 2, 1))
        x_embed = x_embed.permute(0, 2, 1)

        # gumbel softmax, and sample residual
        logits = x_embed[:, :, 0]
        indices = torch.nn.functional.gumbel_softmax(logits, tau=1, hard=True)
        indices = indices.argmax(dim=-1)

        residuals = x_embed[torch.arange(B), indices, 1:]
        ref_points = x[torch.arange(B), :, indices]

        # trying just points, instead of points + residuals
        # return ref_points, x_embed
        return ref_points + residuals, x_embed


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class RelativePoseEmbedder(nn.Module):
    """
    Embeds relative poses into vector representations.
    """
    def __init__(self, hidden_size, rel_pose_type):
        super().__init__()
        if rel_pose_type == "quaternion":
            input_size = 7
        elif rel_pose_type == "rotation_6d":
            input_size = 9
        elif rel_pose_type == "logmap":
            input_size = 6
        elif rel_pose_type == "translation":
            input_size = 3
        else:
            raise ValueError(f"Unknown relative pose rotation type: {rel_pose_type}")
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    
    def forward(self, poses):
        return self.mlp(poses)

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                                 Custom Attention Layers                       #
#################################################################################


class CrossAttention(nn.Module):
    """
    Cross attention layer adapted from
    https://github.com/pprp/timm/blob/e9aac412de82310e6905992e802b1ee4dc52b5d1/timm/models/crossvit.py#L132
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim_x % num_heads == 0, "dim x must be divisible by num_heads"
        head_dim = dim_x // num_heads
        self.scale = head_dim**-0.5

        self.wq = nn.Linear(dim_x, dim_x, bias=qkv_bias)
        self.wk = nn.Linear(dim_y, dim_x, bias=qkv_bias)
        self.wv = nn.Linear(dim_y, dim_x, bias=qkv_bias)
        self.proj = nn.Linear(dim_x, dim_x)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, Cx = x.shape
        # _, _, Cy = y.shape
        _, Ny, Cy = y.shape
        q = (
            self.wq(x)
            .reshape(B, N, self.num_heads, Cx // self.num_heads)
            .transpose(1, 2)
        )
        k = (
            self.wk(y)
            .reshape(B, Ny, self.num_heads, Cx // self.num_heads)
            .transpose(1, 2)
        )
        v = (
            self.wv(y)
            .reshape(B, Ny, self.num_heads, Cx // self.num_heads)
            .transpose(1, 2)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, Cx)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#                                 Core DiT Layers                               #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiTRelativeBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning 
    and 3D relative self attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadRelativeAttentionWrapper(
            embed_dim=hidden_size, 
            num_heads=num_heads,
            dropout=0.0,
            bias=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, x_pos=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            query=x, key=x, value=x, rotary_pe=(x_pos, x_pos)
        )[0] # [0] is the attention output

        x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x)
        return x

class DiTCrossBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning and scene cross attention.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            dim_x=hidden_size,
            dim_y=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, y, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_x,
            scale_x,
            gate_x,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            modulate(self.norm2(x), shift_mca, scale_mca), y
        )
        x = x + gate_x.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_x, scale_x)
        )
        return x


class DiTRelativeCrossBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning 
    and 3D relative self attention + 3D relative scene cross attention.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiheadRelativeAttentionWrapper(
            embed_dim=hidden_size, 
            num_heads=num_heads,
            dropout=0.0,
            bias=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiheadRelativeAttentionWrapper(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            bias=True
        )
        
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, y, c, x_pos=None, y_pos=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_x,
            scale_x,
            gate_x,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)
        
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            query=x, key=x, value=x, rotary_pe=(x_pos, x_pos)
        )[0] # [0] is the attention output
        
        x = modulate(self.norm2(x), shift_mca, scale_mca)
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            query=x, key=y, value=y, rotary_pe=(x_pos, y_pos)
        )[0] # [0] is the attention output
        
        x = x + gate_x.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_x, scale_x)
        )
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

#################################################################################
#                                 Core DiT Models                               #
#################################################################################

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class LinearRegressionModel(nn.Module):
    """
    Linear regression baseline, with attention - no diffusion.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            model_cfg=None,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.out_channels = 3

        # initializing embedder for action point cloud
        self.x_embedder = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # initializing embedder for anchor point cloud
        self.y_embedder = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        class LinearCrossBlock(nn.Module):
            """
            Cross attention block for linear regression model.
            """
            def __init__(self, hidden_size, num_heads, mlp_ratio):
                super().__init__()
                self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
                self.cross_attn = CrossAttention(
                    dim_x=hidden_size,
                    dim_y=hidden_size,
                    num_heads=num_heads,
                    qkv_bias=True,
                )
                mlp_hidden_dim = int(hidden_size * mlp_ratio)
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = Mlp(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=approx_gelu,
                    drop=0,
                )

            def forward(self, x, y):
                x = self.cross_attn(self.norm(x), y)
                x = self.mlp(x)
                return x
            
        class LinearFinalLayer(nn.Module):
            """
            Final layer of the linear regression model.
            """
            def __init__(self, hidden_size, out_channels):
                super().__init__()
                self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
                self.linear = nn.Linear(hidden_size, out_channels, bias=True)

            def forward(self, x):
                x = self.norm_final(x)
                x = self.linear(x)
                return x


        # TODO: DUPLICATE THE CROSS ATTENTION LAYER BASED ON DEPTH
        self.blocks = nn.ModuleList(
            [
                LinearCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.final_layer = LinearFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of linear regression model.
        """

        if self.model_cfg.center_noise:
            raise NotImplementedError("Center noise not implemented for linear regression model.")
        if self.model_cfg.rotary:
            raise NotImplementedError("Rotary not implemented for linear regression model.")
        
        # encode x and y
        x = torch.transpose(self.x_embedder(x), -1, -2)
        y = torch.transpose(self.y_embedder(y), -1, -2)
        # forward pass through cross attention blocks
        for block in self.blocks:
            x = block(x, y)
        
        # final layer
        x = self.final_layer(x)
        x = torch.transpose(x, -1, -2)
        return x


# Custom point cloud DiT for TAX3D
class DiT_PointCloud(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses scene-level self-attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        # Encoder for current timestep x features
        self.x_embedder = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeBlock if self.model_cfg.rotary else DiTBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, L, 3) tensor of spatial inputs (point clouds)
        t: (N,) tensor of diffusion timesteps
        x0: (N, L, 3) tensor of un-noised x (e.g. scene) features
        """
        # noise-centering, if enabled
        if self.model_cfg.center_noise:
            relative_center = torch.mean(x, dim=2, keepdim=True)
            x = x - relative_center
            x0 = x0 - relative_center

        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))

        # encode x, x0 features
        x = torch.cat((x, x0), dim=1)
        x = torch.transpose(self.x_embedder(x), -1, -2)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            if self.model_cfg.rotary:
                x = block(x, t_emb, x_pos)
            else:
                x = block(x, t_emb)

        # final layer
        x = self.final_layer(x, t_emb)
        x = torch.transpose(x, -1, -2)
        return x



def pointwise_mlp(in_channels, out_channels):
    """
    Helper function to create pointwise MLP.
    """
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
    )

def pointnet_encoder(model_cfg, out_channels, in_channels=0):
    pn_params = PN2DenseParams()
    if model_cfg.scale_inputs != "none":
        pn_params.sa1.r, pn_params.sa2.r = 0.2, 0.4
    else:
        pn_params.sa1.r, pn_params.sa2.r = 2.4, 4.8
    
    class PN2DenseWrapper(nn.Module):
        def __init__(self, in_channels, out_channels, p):
            super().__init__()
            self.pn2dense = PN2Dense(in_channels=in_channels, out_channels=out_channels, p=p)
        
        def forward(self, x):
            batch_size, num_channels = x.shape[0], x.shape[1]
            batch_indices = torch.arange(
                batch_size, device=x.device
            ).repeat_interleave(x.shape[2])

            if num_channels == 3:
                input_batch = tgd.Batch(
                    pos=x.permute(0, 2, 1).reshape(-1, 3), batch=batch_indices
                )
            elif num_channels == 6:
                input_batch = tgd.Batch(
                    pos=x[:, :3, :].permute(0, 2, 1).reshape(-1, 3),
                    x=x[:, 3:, :].permute(0, 2, 1).reshape(-1, 3),
                    batch=batch_indices,
                )
            else:
                raise ValueError(f"Invalid number of input channels: {num_channels}")
            
            output = self.pn2dense(input_batch)
            output = output.reshape(batch_size, -1, output.shape[-1]).permute(0, 2, 1)
            return output
    
    return PN2DenseWrapper(in_channels=in_channels, out_channels=out_channels, p=pn_params)



# Custom poitn cloud DiT with cross attention for TAX3D
class DiT_PointCloud_Cross(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2
        # if using extra features, halve the hidden size again
        if self.model_cfg.extra_features:
            x_encoder_hidden_dims = x_encoder_hidden_dims // 2

        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            self.x_embedder = pointwise_mlp(in_channels, x_encoder_hidden_dims)
        elif self.model_cfg.x_encoder == "pn":
            self.x_embedder = pointnet_encoder(self.model_cfg, x_encoder_hidden_dims)
        elif self.model_cfg.x_encoder == "dgcnn":
            self.x_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = pointwise_mlp(in_channels, hidden_size)
        elif self.model_cfg.y_encoder == "pn":
            self.y_embedder = pointnet_encoder(self.model_cfg, hidden_size)
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = pointwise_mlp(in_channels, x_encoder_hidden_dims)
        elif self.model_cfg.x0_encoder == "pn":
            self.x0_embedder = pointnet_encoder(self.model_cfg, x_encoder_hidden_dims)
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
        
        # Creating extra feature encoders based on model type
        if self.model_cfg.extra_features:
            # creating encoder function
            if self.model_cfg.x_encoder == "mlp":
                encoder_fn = partial(pointwise_mlp, in_channels=in_channels)
            elif self.model_cfg.x_encoder == "pn":
                encoder_fn = partial(pointnet_encoder, model_cfg=self.model_cfg)
            else:
                raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")

            # creating extra feature encoders
            if self.model_cfg.type == "flow":
                self.recon_encoder = encoder_fn(out_channels=x_encoder_hidden_dims)
                self.zeromean_encoder = encoder_fn(out_channels=x_encoder_hidden_dims)
            elif self.model_cfg.type == "point":
                self.flow_encoder = encoder_fn(out_channels=x_encoder_hidden_dims)
                self.zeromean_encoder = encoder_fn(out_channels=x_encoder_hidden_dims)

        # if self.model_cfg.extra_features and self.model_cfg.type == "flow":
        #     # need a recon encoder and a zeromean encoder
        #     self.recon_encoder = pointwise_mlp(in_channels, x_encoder_hidden_dims)
        #     self.zeromean_encoder = pointwise_mlp(in_channels, x_encoder_hidden_dims)
        # elif self.model_cfg.extra_features and self.model_cfg.type == "point":
        #     # need a zero mean encoder and a flow encoder
        #     self.flow_encoder = pointwise_mlp(in_channels, x_encoder_hidden_dims)
        #     self.zeromean_encoder = pointwise_mlp(in_channels, x_encoder_hidden_dims)

        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Relative action-anchor pose embedding, if enabled
        if self.model_cfg.rel_pose:
            self.pose_embedder = RelativePoseEmbedder(hidden_size, self.model_cfg.rel_pose_type)
        else:
            self.pose_embedder = None

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        #w = self.x_embedder.weight.data
        #nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        #nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize rel pose embedding MLP:
        if self.pose_embedder is not None:
            nn.init.normal_(self.pose_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.pose_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
            rel_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
            rel_pose (Optional[torch.Tensor]): (B, Dp, N) tensor of relative poses between x0 and y
        """
        # noise-centering, if enabled
        if self.model_cfg.center_noise:
            relative_center = torch.mean(x, dim=2, keepdim=True)
            x = x - relative_center
            y = y - relative_center
        
        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))
            y_pos = self.rotary_pos_enc(y.permute(0, 2, 1))

        # encode x, y, x0 features
        x_emb = self.x_embedder(x)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            x_emb = torch.cat([x_emb, x0_emb], dim=1)

        # If necessary, encode extra features
        if self.model_cfg.extra_features and self.model_cfg.type == "flow":
            # encode extra flow features
            x_recon = self.recon_encoder(x + x0)
            x_zeromean = self.zeromean_encoder(x - torch.mean(x, dim=2, keepdim=True))
            x_emb = torch.cat([x_emb, x_recon, x_zeromean], dim=1)
        elif self.model_cfg.extra_features and self.model_cfg.type == "point":
            # encode extra point features
            flow = x - x0
            x_flow = self.flow_encoder(flow)
            x_zeromean = self.zeromean_encoder(flow - torch.mean(flow, dim=2, keepdim=True))
            x_emb = torch.cat([x_emb, x_flow, x_zeromean], dim=1)



        if self.model_cfg.y_encoder is not None:
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        x = x_emb.permute(0, 2, 1)

        # timestep embedding
        c = self.t_embedder(t)

        # relative pose embedding
        if self.model_cfg.rel_pose:
            assert rel_pose is not None, "relative poses must be provided if rel_pose is enabled"
            rel_pose_emb = self.pose_embedder(rel_pose)
            c = c + rel_pose_emb

        # forward pass through DiT blocks
        for block in self.blocks:
            if self.model_cfg.rotary:
                x = block(x, y_emb, c, x_pos, y_pos)
            else:
                x = block(x, y_emb, c)

        # final layer
        x = self.final_layer(x, c)
        x = x.permute(0, 2, 1)
        return x


class DiT_PointCloud_Cross_Joint(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # get encoder fn
        if self.model_cfg.x_encoder != self.model_cfg.x0_encoder:
            raise ValueError("Joint encoder not supported for different x and x0 encoders.")
        if self.model_cfg.x_encoder != self.model_cfg.y_encoder:
            raise ValueError("Joint encoder not supported for different x and y encoders.")
        
        if self.model_cfg.x_encoder == "mlp":
            encoder_fn = partial(pointwise_mlp, in_channels=in_channels)
        elif self.model_cfg.x_encoder == "pn":
            encoder_fn = partial(pointnet_encoder, model_cfg=self.model_cfg)
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")

        self.query_encoder = encoder_fn(out_channels=hidden_size)
        self.context_encoder = encoder_fn(out_channels=hidden_size)
        # handle extra features, and query mixer
        if self.model_cfg.extra_features:
            self.shape_encoder = encoder_fn(in_channels=3, out_channels=hidden_size)
            self.query_mixer = pointwise_mlp(3 * hidden_size, hidden_size)
        else:
            self.query_mixer = pointwise_mlp(2 * hidden_size, hidden_size)

        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Relative action-anchor pose embedding, if enabled
        if self.model_cfg.rel_pose:
            self.pose_embedder = RelativePoseEmbedder(hidden_size, self.model_cfg.rel_pose_type)
        else:
            self.pose_embedder = None
        
        # DiT blocks
        self.blocks = nn.ModuleList(
            [
                DiTCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize rel pose embedding MLP:
        if self.pose_embedder is not None:
            nn.init.normal_(self.pose_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.pose_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
            rel_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if self.model_cfg.type == "flow":
            x_flow = x
            x_recon = x + x0
        elif self.model_cfg.type == "point":
            x_recon = x
            x_flow = x - x0

        query_input = x0
        context_input = torch.cat([x_recon, y], dim=-1)
        q = query_input.shape[-1]

        query_feature = self.query_encoder(query_input)
        context_feature = self.context_encoder(context_input)
        query_context_feature, context_context_feature = context_feature[:, :, :q], context_feature[:, :, q:]
        
        if self.model_cfg.extra_features:
            x_recon_mean = x_recon - torch.mean(x_recon, dim=2, keepdim=True)
            shape_input = torch.cat([x_recon_mean, x_flow], dim=-2)
            shape_feature = self.shape_encoder(shape_input)

            query_emb = torch.cat([query_feature, query_context_feature, shape_feature], dim=-2)
        else:
            query_emb = torch.cat([query_feature, query_context_feature], dim=-2)
        context_emb = context_context_feature.permute(0, 2, 1)

        # pass through mixers
        query_emb = self.query_mixer(query_emb).permute(0, 2, 1)

        # timestep embedding
        c = self.t_embedder(t)

        # relative pose embedding
        if self.model_cfg.rel_pose:
            assert rel_pose is not None, "relative poses must be provided if rel_pose is enabled"
            rel_pose_emb = self.pose_embedder(rel_pose)
            c = c + rel_pose_emb
        
        # forward pass through DiT blocks
        for block in self.blocks:
            emb = block(query_emb, context_emb, c)
        
        # final layer
        out = self.final_layer(emb, c).permute(0, 2, 1)
        return out


class PointCloudDiT2(nn.Module):
    """
    New DiT architecture to handle spatial queries.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg
        self.diffuse_ref_frame = model_cfg.diffuse_ref_frame

        if not self.diffuse_ref_frame:
            raise ValueError("Currently only diffusing reference frame for TAX3Dv2")

        # TODO: this is hacked in for now because of config overload, reorganize later
        if self.model_cfg.center_query and self.model_cfg.extra_features:
            raise ValueError("Center query and extra features cannot be used together.")
        
        # Query and context encoder. (Output is hidden_size - 1 to allow for query mask.)
        if self.model_cfg.center_query:
            self.num_query_inputs = 4
            self.num_context_inputs = 2
        elif self.model_cfg.extra_features:
            self.num_query_inputs = 5
            self.num_context_inputs = 2
        else:
            self.num_query_inputs = 2
            self.num_context_inputs = 2

        if self.model_cfg.tax3dv2_encoder == "mlp":
            encoder_fn = partial(pointwise_mlp, in_channels=3)
        elif self.model_cfg.tax3dv2_encoder == "pn":
            encoder_fn = partial(pointnet_encoder, model_cfg=self.model_cfg)
        else:
            raise ValueError(f"Invalid tax3dv2_encoder: {self.model_cfg.tax3dv2_encoder}")

        query_hidden_size_base = (hidden_size - 1) // self.num_query_inputs
        context_hidden_size_base = (hidden_size - 1) // self.num_context_inputs
        self.query_encoders = nn.ModuleList(
            [
                encoder_fn(out_channels=query_hidden_size_base + 1)
                if qi < (hidden_size - 1) - self.num_query_inputs * query_hidden_size_base
                else encoder_fn(out_channels=query_hidden_size_base)
                for qi in range(self.num_query_inputs)
            ]
        )

        self.context_encoders = nn.ModuleList(
            [
                encoder_fn(out_channels=context_hidden_size_base + 1)
                if ci < (hidden_size - 1) - self.num_context_inputs * context_hidden_size_base
                else encoder_fn(out_channels=context_hidden_size_base)
                for ci in range(self.num_context_inputs)
            ]
        )

        # Timestamp embedding.
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks.
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # Final layers for query and context.
        self.query_final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        if self.model_cfg.context_token:
            # Learnable context token.
            self.ref_frame_token = nn.Parameter(torch.randn(1, hidden_size - 1, 1))
            self.context_final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        else:
            # TODO: eventually remove this line, very hacky bugfix to make previous checkpoints work
            self.ref_frame_token = nn.Parameter(torch.randn(1, 3, 1))
            self.context_final_layer = FinalLayer(hidden_size, 1, self.out_channels + 1)

        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # # Initialize query encoders
        # for query_encoder in self.query_encoders:
        #     w = query_encoder.weight.data
        #     nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        #     nn.init.constant_(query_encoder.bias, 0)
        # # w = self.query_encoder.weight.data
        # # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # # nn.init.constant_(self.query_encoder.bias, 0)

        # # Initialize context encoder
        # for context_encoder in self.context_encoders:
        #     w = context_encoder.weight.data
        #     nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        #     nn.init.constant_(context_encoder.bias, 0)
        # # w = self.context_encoder.weight.data
        # # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # # nn.init.constant_(self.context_encoder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers.
        nn.init.constant_(self.query_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.query_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.query_final_layer.linear.weight, 0)
        nn.init.constant_(self.query_final_layer.linear.bias, 0)
        nn.init.constant_(self.context_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.context_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.context_final_layer.linear.weight, 0)
        nn.init.constant_(self.context_final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            q: int,
            ref_frame: Optional[torch.Tensor] = None,
            query_center: Optional[torch.Tensor] = None,
            query_scale: Optional[torch.Tensor] = None,
            scene_center: Optional[torch.Tensor] = None,
            scene_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, 3, Nq) tensor of batch current timestep x features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, 3, Nq + Nc) tensor of batch scene point clouds
            q (int): Size of spatial query point cloud (not necessarily Nq)
            ref_frame (Optional[torch.Tensor]): (B, 3, 1) tensor of batch reference frame point clouds to inpaint
        """
        # ensure that all query size aligns with input size
        input_size = q + self.diffuse_ref_frame 
        assert input_size == x.shape[2]

        ##################################################################
        # Extract and encode spatial query.
        ##################################################################
        query_points = y[:, :, :q]

        # Extracting query point cloud - in normalized query frame and normalized scene frame.
        if self.model_cfg.scale_inputs != "none":
            query_points_q = (query_points - query_center) / query_scale
            query_points_s = (query_points - scene_center) / scene_scale
        else:
            query_points_q, query_points_s = query_points, query_points
            query_points_q = query_points_q - query_points_q.mean(dim=-1, keepdim=True)

        # Extracting noisy query prediction - in normalized query frame (shape) and normalized scene frame (query).
        noisy_query_shape_q = x[:, :, :q]
        noisy_ref_frame_s = ref_frame if ref_frame is not None else x[:, :, -1:]
        if self.model_cfg.scale_inputs != "none":
            noisy_query_shape_s = (noisy_query_shape_q * query_scale) / scene_scale
            noisy_query_s = noisy_query_shape_s + noisy_ref_frame_s
        else:
            noisy_query_s = noisy_query_shape_q + noisy_ref_frame_s

        # not diffusing reference frame not implemented for now
        if not self.diffuse_ref_frame:
            raise NotImplementedError("Not diffusing reference frame not implemented for now.")
            # noisy_query_s = x[:, :, :q]
            # noisy_query_shape_q = noisy_query_s - noisy_query_s.mean(dim=-1, keepdim=True)

        # handling additional query inputs 
        if self.model_cfg.center_query:
            feature_list = [
                query_points_s, # initial query in scene frame
                noisy_query_s, # query prediction in scene frame
                query_points_q, # initial query in query frame
                noisy_query_shape_q, # query prediction in query-centric frame
            ]
        elif self.model_cfg.extra_features:
            feature_list = [
                query_points_s, # initial query in scene frame
                noisy_query_s, # query prediction in scene frame
                query_points_q, # initial query in query frame
                noisy_query_shape_q, # query prediction in query-centric frame
                # noisy_query_s - query_points_s, # flow prediction in query-centric frame # TODO: BUGGED, RETRAIN THIS
                noisy_query_shape_q - query_points_q, # flow prediction in query-centric frame
            ]
        else:
            feature_list = [
                query_points_s, # initial query in scene frame
                noisy_query_s, # query prediction in scene frame
            ]
        
        # query = self.query_encoder(torch.cat(feature_list, dim=1))
        assert len(feature_list) == self.num_query_inputs, "Number of given query inputs does not match expected input."
        query = torch.cat(
            [self.query_encoders[i](feature) for i, feature in enumerate(feature_list)],
            dim=1,
        )

        ##################################################################
        # Extract and encode spatial context.
        ##################################################################
        context_points = y[:, :, q:]

        # Extracting context point cloud - in normalized query frame and normalized scene frame.
        if self.model_cfg.scale_inputs != "none":
            context_points_s = (context_points - scene_center) / scene_scale
            context_points_q = (context_points_s - noisy_ref_frame_s) * scene_scale / query_scale
        else:
            context_points_s = context_points
            context_points_q = context_points_s - noisy_ref_frame_s
        
        # context_list = [context_points_s, context_points_q]
        context_list = [context_points_q, context_points_s]
        assert len(context_list) == self.num_context_inputs, "Number of given context inputs does not match expected input."
        # context = self.context_encoder(context_points_s)
        context = torch.cat(
            [self.context_encoders[i](context_feature) for i, context_feature in enumerate(context_list)],
            dim=1,
        )

        # import rpad.visualize_3d.plots as vpl
        # query_points_q_viz = query_points_q[0].permute(1, 0).detach().cpu().numpy()
        # query_points_s_viz = query_points_s[0].permute(1, 0).detach().cpu().numpy()
        # noisy_query_shape_q_viz = noisy_query_shape_q[0].permute(1, 0).detach().cpu().numpy()
        # noisy_query_s_viz = noisy_query_s[0].permute(1, 0).detach().cpu().numpy()
        # context_points_s_viz = context_points_s[0].permute(1, 0).detach().cpu().numpy()
        # context_points_q_viz = context_points_q[0].permute(1, 0).detach().cpu().numpy()

        # query_points_q_seg = np.ones(query_points_q_viz.shape[0]) * 0
        # noisy_query_shape_q_seg = np.ones(noisy_query_shape_q_viz.shape[0]) * 1        
        # context_points_q_seg = np.ones(context_points_q_viz.shape[0]) * 2

        # query_points_s_seg = np.ones(query_points_s_viz.shape[0]) * 3
        # noisy_query_s_seg = np.ones(noisy_query_s_viz.shape[0]) * 4
        # context_points_s_seg = np.ones(context_points_s_viz.shape[0]) * 5

        # fig = vpl.segmentation_fig(
        #     np.concatenate([
        #         query_points_q_viz,
        #         query_points_s_viz,
        #         noisy_query_shape_q_viz,
        #         noisy_query_s_viz,
        #         context_points_s_viz,
        #         context_points_q_viz,
        #     ]),
        #     np.concatenate([
        #         query_points_q_seg,
        #         query_points_s_seg,
        #         noisy_query_shape_q_seg,
        #         noisy_query_s_seg,
        #         context_points_s_seg,
        #         context_points_q_seg,
        #     ]).astype(int),
        # )
        # fig.show()
        # breakpoint()

        # Add ref frame token to context, if necessary.
        if self.model_cfg.context_token:
            context = torch.cat([context, self.ref_frame_token.expand(context.shape[0], -1, -1)], dim=2)

        # Creating and appending query mask.
        emb = torch.cat([query, context], dim=2).permute(0, 2, 1)
        query_mask = torch.zeros(emb.shape[0], emb.shape[1], 1, device=emb.device)
        query_mask[:, :q, :] = 1
        emb = torch.cat([emb, query_mask], dim=2)

        # Timestep embedding.
        c = self.t_embedder(t)

        # Forward pass through DiT blocks.
        for block in self.blocks:
            emb = block(emb, c)
        query_emb, context_emb = emb[:, :q, :], emb[:, q:, :]
        
        # Query prediction.
        out = self.query_final_layer(query_emb, c).permute(0, 2, 1)
        # Context prediction, if necessary.
        if self.model_cfg.context_token:
            context_token_emb = context_emb[:, -1:, :]
            context_out = self.context_final_layer(context_token_emb, c).permute(0, 2, 1)
            out = torch.cat([out, context_out], dim=-1)
        elif self.diffuse_ref_frame and context_emb.shape[2] > 0:
            context_dense_out = self.context_final_layer(context_emb, c).permute(0, 2, 1)
            # Split context output into logits and residuals.
            weights = torch.softmax(context_dense_out[:, [0], :], dim=-1)
            ref_frame_means = context_dense_out[:, 1:4, :] + context_points

            # Aggregate context predictions, and include sigma prediction if necessary.
            context_out = torch.sum(weights * ref_frame_means, dim=-1, keepdim=True)

            if self.learn_sigma:
                sigmas = torch.sum(weights * context_dense_out[:, 4:, :], dim=-1, keepdim=True)
                context_out = torch.cat([context_out, sigmas], dim=1)

            # Concatenate query and context predictions. Also return weights and residuals.
            out = (torch.cat([out, context_out], dim=-1), torch.cat([weights, ref_frame_means], dim=1))
        return out



class PointCloudDit2_2(nn.Module):
    """
    New DiT architecture to handle spatial queries.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg
        self.diffuse_ref_frame = model_cfg.diffuse_ref_frame

        if not self.diffuse_ref_frame:
            raise ValueError("Currently only diffusing reference frame for TAX3Dv2")
        
        if not self.model_cfg.center_query:
            raise ValueError("Center query must be enabled for TAX3Dv2")
        
        if self.model_cfg.context_token and self.model_cfg.scene_residual:
            raise ValueError("Context token and scene residual cannot be enabled together.")
        
        if self.model_cfg.tax3dv2_encoder == "mlp":
            encoder_fn = partial(pointwise_mlp, in_channels=3)
        elif self.model_cfg.tax3dv2_encoder == "pn":
            encoder_fn = partial(pointnet_encoder, model_cfg=self.model_cfg)
        else:
            raise ValueError(f"Invalid tax3dv2_encoder: {self.model_cfg.tax3dv2_encoder}")
        
        # query, flow, and scene cnoders
        self.query_encoder = encoder_fn(out_channels=hidden_size)
        self.flow_encoder = encoder_fn(out_channels=hidden_size)
        self.scene_encoder = encoder_fn(out_channels=hidden_size)
        
        # query and context mixers
        self.query_mixer = pointwise_mlp(4 * hidden_size, hidden_size - 1)
        self.context_mixer = pointwise_mlp(2 * hidden_size, hidden_size - 1)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # query decoder
        self.query_final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        # self.context_final_layer = FinalLayer(hidden_size, 1, self.out_channels + 1)

        if self.model_cfg.context_token:
            # Learnable context token.
            self.ref_frame_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.context_final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        else:
            self.context_final_layer = FinalLayer(hidden_size, 1, self.out_channels + 1)

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers.
        nn.init.constant_(self.query_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.query_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.query_final_layer.linear.weight, 0)
        nn.init.constant_(self.query_final_layer.linear.bias, 0)
        nn.init.constant_(self.context_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.context_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.context_final_layer.linear.weight, 0)
        nn.init.constant_(self.context_final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            q: int,
            ref_frame: Optional[torch.Tensor] = None,
            query_center: Optional[torch.Tensor] = None,
            query_scale: Optional[torch.Tensor] = None,
            scene_center: Optional[torch.Tensor] = None,
            scene_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, 3, Nq) tensor of batch current timestep x features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, 3, Nq + Nc) tensor of batch scene point clouds
            q (int): Size of spatial query point cloud (not necessarily Nq)
            ref_frame (Optional[torch.Tensor]): (B, 3, 1) tensor of batch reference frame point clouds to inpaint
        """
        # ensure that all query size aligns with input size
        input_size = q + self.diffuse_ref_frame
        assert input_size == x.shape[2]

        ##################################################################
        # Extract scene and query-scale input.s
        ##################################################################
        query_points = y[:, :, :q]
        # Extracting query point cloud - in normalized query frame and normalized scene frame.
        if self.model_cfg.scale_inputs != "none":
            query_points_q = (query_points - query_center) / query_scale
            query_points_s = (query_points - scene_center) / scene_scale
        else:
            query_points_q, query_points_s = query_points, query_points
            query_points_q = query_points_q - query_points_q.mean(dim=-1, keepdim=True)

        # Extracting noisy query prediction - in normalized query frame (shape) and normalized scene frame (query).
        noisy_query_shape_q = x[:, :, :q]
        noisy_ref_frame_s = ref_frame if ref_frame is not None else x[:, :, -1:]
        if self.model_cfg.scale_inputs != "none":
            noisy_query_shape_s = (noisy_query_shape_q * query_scale) / scene_scale
            noisy_query_s = noisy_query_shape_s + noisy_ref_frame_s
        else:
            noisy_query_s = noisy_query_shape_q + noisy_ref_frame_s
        
        context_points = y[:, :, q:]
        # Extracting context point cloud - in normalized query frame and normalized scene frame.
        if self.model_cfg.scale_inputs != "none":
            context_points_s = (context_points - scene_center) / scene_scale
            context_points_q = (context_points_s - noisy_ref_frame_s) * scene_scale / query_scale
        else:
            context_points_s = context_points
            context_points_q = context_points_s - noisy_ref_frame_s

        ##################################################################
        # Encode query, flow, and scene.
        ##################################################################
        # query input: query_points_q
        # flow input: noisy_query_shape_q, context_points_q
        # scene input: query_points_s, noisy_query_s, context_points_s

        query_encoder_input = query_points_q
        flow_encoder_input = torch.cat([noisy_query_shape_q, context_points_q], dim=2)
        scene_encoder_input = torch.cat([query_points_s, noisy_query_s, context_points_s], dim=2)

        query_features = self.query_encoder(query_encoder_input)
        flow_features = self.flow_encoder(flow_encoder_input)
        scene_features = self.scene_encoder(scene_encoder_input)

        ##################################################################
        # Mix query and context features.
        ##################################################################
        # query mixing
        noisy_query_flow_features, context_flow_features = flow_features[:, :, :q], flow_features[:, :, q:]
        query_scene_features, noisy_query_scene_features = scene_features[:, :, :q], scene_features[:, :, q:2*q]
        context_scene_features = scene_features[:, :, 2*q:]

        query_emb = self.query_mixer(
            torch.cat([query_features, noisy_query_flow_features, query_scene_features, noisy_query_scene_features], dim=1)
        )
        context_emb = self.context_mixer(
            torch.cat([context_flow_features, context_scene_features], dim=1)
        )

        emb = torch.cat([query_emb, context_emb], dim=2).permute(0, 2, 1)
        query_mask = torch.zeros(emb.shape[0], emb.shape[1], 1, device=emb.device)
        query_mask[:, :q, :] = 1
        emb = torch.cat([emb, query_mask], dim=2)

        # Add ref frame token, if necessary.
        if self.model_cfg.context_token:
            emb = torch.cat([emb, self.ref_frame_token.expand(emb.shape[0], -1, -1)], dim=-2)

        # Timestep embedding.
        c = self.t_embedder(t)

        # Forward pass through DiT blocks.
        for block in self.blocks:
            emb = block(emb, c)
        query_emb, context_emb = emb[:, :q, :], emb[:, q:, :]

        # Query prediction.
        out = self.query_final_layer(query_emb, c).permute(0, 2, 1)

        # Context prediction, if necessary.
        if self.model_cfg.context_token:
            context_token_emb = context_emb[:, -1:, :]
            context_out = self.context_final_layer(context_token_emb, c).permute(0, 2, 1)
            out = torch.cat([out, context_out], dim=-1)
        elif self.diffuse_ref_frame and context_emb.shape[2] > 0:

            # either context residuals, or scene residuals
            if self.model_cfg.scene_residual:
                context_dense_out = self.context_final_layer(emb, c).permute(0, 2, 1)
                # Split context output into logits and residuals.
                weights = torch.softmax(context_dense_out[:, [0], :], dim=-1)
                scene_points_s = torch.cat([query_points_s, context_points_s], dim=2)
                ref_frame_means = context_dense_out[:, 1:4, :] + scene_points_s
            else:
                context_dense_out = self.context_final_layer(context_emb, c).permute(0, 2, 1)
                # Split context output into logits and residuals.
                weights = torch.softmax(context_dense_out[:, [0], :], dim=-1)
                ref_frame_means = context_dense_out[:, 1:4, :] + context_points_s

            # # context_dense_out = self.context_final_layer(emb, c).permute(0, 2, 1)
            # context_dense_out = self.context_final_layer(context_emb, c).permute(0, 2, 1)
            # # Split context output into logits and residuals.
            # weights = torch.softmax(context_dense_out[:, [0], :], dim=-1)
            # # scene_points_s = torch.cat([query_points_s, context_points_s], dim=2)
            # # ref_frame_means = context_dense_out[:, 1:4, :] + scene_points_s
            # ref_frame_means = context_dense_out[:, 1:4, :] + context_points_s

            # Aggregate context predictions, and include sigma prediction if necessary.
            context_out = torch.sum(weights * ref_frame_means, dim=-1, keepdim=True)

            if self.learn_sigma:
                sigmas = torch.sum(weights * context_dense_out[:, 4:, :], dim=-1, keepdim=True)
                context_out = torch.cat([context_out, sigmas], dim=1)

            # Concatenate query and context predictions. Also return weights and residuals.
            out = (torch.cat([out, context_out], dim=-1), torch.cat([weights, ref_frame_means], dim=1))
        return out



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
