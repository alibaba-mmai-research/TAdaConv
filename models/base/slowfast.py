#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

import torch
import torch.nn as nn
from utils.registry import Registry
from models.base.base_blocks import (
    Base3DResStage, STEM_REGISTRY, BRANCH_REGISTRY
)
from models.base.backbone import BACKBONE_REGISTRY, _n_conv_resnet
from models.utils.init_helper import _init_convnet_weights

@BACKBONE_REGISTRY.register()
class Slowfast(nn.Module):
    """
    Constructs SlowFast model.
    
    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."

    Modified from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py.
    """
    def __init__(self, cfg):
        super(Slowfast, self).__init__()
        self.mode = cfg.VIDEO.BACKBONE.SLOWFAST.MODE
        if self.mode == "slowfast":
            self.slow_enable = True
            self.fast_enable = True
        elif self.mode == "slowonly":
            self.slow_enable = True 
            self.fast_enable = False
        elif self.mode == "fastonly":
            self.slow_enable = False
            self.fast_enable = True
        self._construct_backbone(cfg)
    
    def _construct_slowfast_cfg(self, cfg):
        cfgs = []
        for i in range(2):
            pseudo_cfg = cfg.deep_copy()
            pseudo_cfg.VIDEO.BACKBONE.KERNEL_SIZE = pseudo_cfg.VIDEO.BACKBONE.KERNEL_SIZE[i]
            pseudo_cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK = pseudo_cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[i]
            if i == 1:
                pseudo_cfg.VIDEO.BACKBONE.ADD_FUSION_CHANNEL = False
                for idx, k in enumerate(pseudo_cfg.VIDEO.BACKBONE.NUM_FILTERS):
                    pseudo_cfg.VIDEO.BACKBONE.NUM_FILTERS[idx] = k//cfg.VIDEO.BACKBONE.SLOWFAST.BETA
            else:
                pseudo_cfg.VIDEO.BACKBONE.ADD_FUSION_CHANNEL = self.fast_enable
            cfgs.append(pseudo_cfg)
        return cfgs

    def _construct_slowfast_module(self, cfgs, module, **kwargs):
        modules = []
        for idx, cfg in enumerate(cfgs):
            if (idx == 0 and self.slow_enable == True) or (idx == 1 and self.fast_enable == True):
                modules.append(
                    module(cfg, **kwargs)
                )
            else:
                modules.append(
                    nn.Identity
                )
        return modules


    def _construct_backbone(self, cfg):
        # ------------------- Stem -------------------
        cfgs = self._construct_slowfast_cfg(cfg)
        self.slow_conv1, self.fast_conv1 = self._construct_slowfast_module(
            cfgs, STEM_REGISTRY.get(cfg.VIDEO.BACKBONE.STEM.NAME)
        )

        self.slowfast_fusion1 = FuseFastToSlow(cfgs, stage_idx=0, mode=self.mode)

        (n1, n2, n3, n4) = _n_conv_resnet[cfg.VIDEO.BACKBONE.DEPTH]

        # ------------------- Main arch -------------------
        self.slow_conv2, self.fast_conv2 = self._construct_slowfast_module(
            cfgs, Base3DResStage, num_blocks = n1, stage_idx = 1,
        )
        self.slowfast_fusion2 = FuseFastToSlow(cfgs, stage_idx=1, mode=self.mode)

        self.slow_conv3, self.fast_conv3 = self._construct_slowfast_module(
            cfgs, Base3DResStage, num_blocks = n2, stage_idx = 2,
        )
        self.slowfast_fusion3 = FuseFastToSlow(cfgs, stage_idx=2, mode=self.mode)


        self.slow_conv4, self.fast_conv4 = self._construct_slowfast_module(
            cfgs, Base3DResStage, num_blocks = n3, stage_idx = 3,
        )
        self.slowfast_fusion4 = FuseFastToSlow(cfgs, stage_idx=3, mode=self.mode)


        self.slow_conv5, self.fast_conv5 = self._construct_slowfast_module(
            cfgs, Base3DResStage, num_blocks = n4, stage_idx = 4,
        )

        _init_convnet_weights(self)
    
    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]
        assert isinstance(x, list), "Input to SlowFast should be lists"
        x_slow = x[0]
        x_fast = x[1]

        x_slow, x_fast = self.slow_conv1(x_slow), self.fast_conv1(x_fast)
        x_slow, x_fast = self.slowfast_fusion1(x_slow, x_fast)
        x_slow, x_fast = self.slow_conv2(x_slow), self.fast_conv2(x_fast)
        x_slow, x_fast = self.slowfast_fusion2(x_slow, x_fast)
        x_slow, x_fast = self.slow_conv3(x_slow), self.fast_conv3(x_fast)
        x_slow, x_fast = self.slowfast_fusion3(x_slow, x_fast)
        x_slow, x_fast = self.slow_conv4(x_slow), self.fast_conv4(x_fast)
        x_slow, x_fast = self.slowfast_fusion4(x_slow, x_fast)
        x_slow, x_fast = self.slow_conv5(x_slow), self.fast_conv5(x_fast)
        return x_slow, x_fast

class FuseFastToSlow(nn.Module):
    def __init__(self, cfg, stage_idx, mode):
        super(FuseFastToSlow, self).__init__()
        self.mode = mode
        if mode == "slowfast":
            slow_cfg, fast_cfg = cfg
            dim_in      = fast_cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_idx]
            dim_out     = fast_cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_idx] * fast_cfg.VIDEO.BACKBONE.SLOWFAST.CONV_CHANNEL_RATIO
            kernel_size = [fast_cfg.VIDEO.BACKBONE.SLOWFAST.KERNEL_SIZE, 1, 1]
            stride      = [fast_cfg.VIDEO.BACKBONE.SLOWFAST.ALPHA, 1, 1]
            padding     = [fast_cfg.VIDEO.BACKBONE.SLOWFAST.KERNEL_SIZE//2, 0, 0]
            bias        = fast_cfg.VIDEO.BACKBONE.SLOWFAST.FUSION_CONV_BIAS
            self.conv_fast_to_slow = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size,
                stride,
                padding,
                bias=bias
            )
            if fast_cfg.VIDEO.BACKBONE.SLOWFAST.FUSION_BN:
                self.bn = nn.BatchNorm3d(
                    dim_out, eps=fast_cfg.BN.EPS, momentum=fast_cfg.BN.MOMENTUM
                )
            if fast_cfg.VIDEO.BACKBONE.SLOWFAST.FUSION_RELU:
                self.relu = nn.ReLU(inplace=True)

    def forward(self, x_slow, x_fast):
        if self.mode == "slowfast":
            fuse = self.conv_fast_to_slow(x_fast)
            if hasattr(self, "bn"):
                fuse = self.bn(fuse)
            if hasattr(self, "relu"):
                fuse = self.relu(fuse)
            return torch.cat((x_slow, fuse), 1), x_fast
        else:
            return x_slow, x_fast
