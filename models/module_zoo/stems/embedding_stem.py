#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Embedding stems. """

import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from models.base.backbone import BACKBONE_REGISTRY
from models.base.base_blocks import (
    STEM_REGISTRY, BRANCH_REGISTRY, HEAD_REGISTRY, DropPath, BaseHead
)

@STEM_REGISTRY.register()
class PatchEmbedStem(nn.Module):
    """ 
    Video to Patch Embedding.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        channels        = cfg.DATA.NUM_INPUT_CHANNELS       if cfg is not None else 3   # default 3
        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 16
        patch_size      = cfg.VIDEO.BACKBONE.PATCH_SIZE     if cfg is not None else 16  # default 16
        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768

        num_patches_per_image = (image_size // patch_size) ** 2
        num_patches = num_patches_per_image * num_frames

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches

        self.conv1 = nn.Conv3d(
            in_channels     =channels, 
            out_channels    =dim, 
            kernel_size     =[1, patch_size, patch_size], 
            stride          =[1, patch_size, patch_size], 
        )

    def forward(self, x):
        b, c, t, h, w, p = *x.shape, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        x = self.conv1(x)
        # b, c, t, h, w -> b, c, p (p: num patches)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # b, c, p -> b, p, c
        x = x.permute(0, 2, 1)
        return x

@STEM_REGISTRY.register()
class TubeletEmbeddingStem(nn.Module):
    """ 
    Video to Tubelet Embedding.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        channels        = cfg.DATA.NUM_INPUT_CHANNELS       if cfg is not None else 3   # default 3
        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 16
        patch_size      = cfg.VIDEO.BACKBONE.PATCH_SIZE     if cfg is not None else 16  # default 16
        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        tubelet_size    = cfg.VIDEO.BACKBONE.TUBELET_SIZE   if cfg is not None else 2

        num_patches_per_image = (image_size // patch_size) ** 2
        num_patches = num_patches_per_image * num_frames

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches

        self.conv1 = nn.Conv3d(
            in_channels     =channels, 
            out_channels    =dim, 
            kernel_size     =[tubelet_size, patch_size, patch_size], 
            stride          =[tubelet_size, patch_size, patch_size], 
        )

    def forward(self, x):
        b, c, t, h, w, p = *x.shape, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        x = self.conv1(x)
        # b, c, t, h, w -> b, c, p (p: num patches)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # b, c, p -> b, p, c
        x = x.permute(0, 2, 1)
        return x