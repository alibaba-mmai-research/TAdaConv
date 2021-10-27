#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Transformers. """

import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from models.base.backbone import BACKBONE_REGISTRY
from models.base.base_blocks import (
    STEM_REGISTRY, BRANCH_REGISTRY, HEAD_REGISTRY, DropPath, BaseHead
)

from models.utils.init_helper import lecun_normal_, trunc_normal_, _init_transformer_weights

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, ff_dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    Self-attention module. 
    Currently supports both full self-attention on all the input tokens,
    or only-spatial/only-temporal self-attention. 

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer.
    and 
    Gedas Bertasius, Heng Wang, Lorenzo Torresani.
    Is Space-Time Attention All You Need for Video Understanding?

    Modified from 
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(
        self,
        dim,
        num_heads=12,
        attn_dropout=0.,
        ff_dropout=0.,
        einops_from=None,
        einops_to=None,
        **einops_dims,
    ):
        super().__init__()
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5

        self.to_qkv         = nn.Linear(dim, dim * 3)
        self.attn_dropout   = nn.Dropout(attn_dropout)
        self.proj           = nn.Linear(dim, dim)
        self.ff_dropout     = nn.Dropout(ff_dropout)

        if einops_from is not None and einops_to is not None:
            self.partial = True
            self.einops_from = einops_from
            self.einops_to = einops_to
            self.einops_dims = einops_dims
        else:
            self.partial = False

    def forward(self, x):
        if self.partial:
            return self.forward_partial(
                x,
                self.einops_from,
                self.einops_to,
                **self.einops_dims,
            )
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

    def forward_partial(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q *= self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_attn = (cls_q @ k.transpose(1,2)).softmax(-1)
        cls_attn = self.attn_dropout(cls_attn)
        cls_out = cls_attn @ v

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r = r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        attn = (q_ @ k_.transpose(1, 2)).softmax(-1)
        attn = self.attn_dropout(attn)
        x = attn @ v_

        # merge back time or space
        x = rearrange(x, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        x = torch.cat((cls_out, x), dim = 1)

        # merge back the heads
        x = rearrange(x, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

@BRANCH_REGISTRY.register()
class BaseTransformerLayer(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        mlp_mult        = cfg.VIDEO.BACKBONE.MLP_MULT       if cfg is not None else 4
        drop_path       = drop_path_rate

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

@BRANCH_REGISTRY.register()
class TimesformerLayer(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        patch_size      = cfg.VIDEO.BACKBONE.PATCH_SIZE     if cfg is not None else 16  # default 16
        drop_path       = drop_path_rate
        
        num_patches = (image_size // patch_size) ** 2

        self.norm_temporal = nn.LayerNorm(dim, eps=1e-6)
        self.attn_temporal = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout,
            einops_from='b (f n) d', einops_to='(b n) f d', n = num_patches
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout,
            einops_from='b (f n) d', einops_to='(b f) n d', f = num_frames
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn_temporal(self.norm_temporal(x)))
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

@BACKBONE_REGISTRY.register()
class Transformer(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        num_features    = cfg.VIDEO.BACKBONE.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = cfg.VIDEO.BACKBONE.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = cfg.VIDEO.BACKBONE.DEPTH                if cfg is not None else 12  # default 12
        drop_path       = cfg.VIDEO.BACKBONE.DROP_PATH            if cfg is not None else 16  # default 16
        if hasattr(cfg.VIDEO.BACKBONE, "TUBELET_SIZE"):
            tubelet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE         if cfg is not None else 2
        else:
            tubelet_size = 1

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches_per_frame = (image_size // patch_size) ** 2
        num_patches = num_frames * num_patches_per_frame // tubelet_size

        # constructs the tokenization module.
        self.stem = STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        self.pos_embd = nn.Parameter(torch.zeros(1, num_patches + 1, num_features))
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_features))

        # construct transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]

        x = self.stem(x)

        cls_token = self.cls_token.repeat((x.shape[0],1,1))
        x =  torch.cat((cls_token, x), dim = 1)

        x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)

        return x[:, 0]

@BACKBONE_REGISTRY.register()
class FactorizedTransformer(nn.Module):
    """
    The factorized transformer. 

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer. 
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()


        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        num_features    = cfg.VIDEO.BACKBONE.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = cfg.VIDEO.BACKBONE.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = cfg.VIDEO.BACKBONE.DEPTH                if cfg is not None else 12  # default 12
        depth_temp      = cfg.VIDEO.BACKBONE.DEPTH_TEMP           if cfg is not None else 4   # default 4
        drop_path       = cfg.VIDEO.BACKBONE.DROP_PATH            if cfg is not None else 16  # default 16
        if hasattr(cfg.VIDEO.BACKBONE, "TUBELET_SIZE"):
            tubelet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE         if cfg is not None else 2
        else:
            tubelet_size = 1

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        # constructs the tokenization module.
        self.stem = STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        # both spatial and temporal embeddings/cls_token needs to be constructed
        # for the factorized transformer video model 
        self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 1, num_features))
        self.temp_embd          = nn.Parameter(torch.zeros(1, num_frames // tubelet_size + 1, num_features))
        self.cls_token          = nn.Parameter(torch.randn(1, 1, num_features))
        self.cls_token_out = nn.Parameter(torch.randn(1, 1, num_features))

        # construct spatial transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth+depth_temp)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # construct temporal transformer layers
        self.layers_temporal = nn.Sequential(*[
            BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, drop_path_rate=dpr[i+depth])
            for i in range(depth_temp)])

        self.norm_out = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.temp_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_token_out, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]

        h, w = x.shape[-2:]

        actual_num_patches_per_frame = (h // self.patch_size) * (w // self.patch_size)
        x = self.stem(x)
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            assert not self.training 
            x = rearrange(x, "b (t n) c -> (b t) n c", n=actual_num_patches_per_frame)
        else:
            x = rearrange(x, "b (t n) c -> (b t) n c", n=self.num_patches_per_frame)

        cls_token = self.cls_token.repeat((x.shape[0],1,1))
        x = torch.cat((cls_token, x), dim = 1)

        # to make the input video size changable
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            actual_num_pathces_per_side = int(math.sqrt(actual_num_patches_per_frame))
            if not hasattr(self, "new_pos_embd") or self.new_pos_embd.shape[1] != (actual_num_pathces_per_side**2+1):
                cls_pos_embd = self.pos_embd[:,0,:].unsqueeze(1)
                pos_embd = self.pos_embd[:, 1:, :]
                num_patches_per_side = int(math.sqrt(self.num_patches_per_frame))
                pos_embd = pos_embd.reshape(1, num_patches_per_side, num_patches_per_side, -1).permute(0,3,1,2)
                pos_embd = torch.nn.functional.interpolate(
                    pos_embd, size=(actual_num_pathces_per_side,actual_num_pathces_per_side), mode="bilinear"
                ).permute(0,2,3,1).reshape(1, actual_num_pathces_per_side**2, -1)
                self.new_pos_embd = torch.cat((cls_pos_embd, pos_embd), dim=1)
            x += self.new_pos_embd
        else:
            x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)[:, 0]

        x = rearrange(x, "(b t) c -> b t c", t=self.num_patches//self.num_patches_per_frame)

        cls_token_out = self.cls_token_out.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token_out, x), dim=1)

        x += self.temp_embd

        x = self.layers_temporal(x)
        x = self.norm_out(x)

        return x[:, 0]