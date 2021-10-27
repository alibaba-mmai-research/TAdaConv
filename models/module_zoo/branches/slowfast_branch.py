#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" SlowFast architectures. """

import torch
import torch.nn as nn

from models.base.base_blocks import BaseBranch
from models.base.base_blocks import BRANCH_REGISTRY
from models.utils.init_helper import _init_convnet_weights

@BRANCH_REGISTRY.register()
class SlowfastBranch(BaseBranch):
    """
    Constructs SlowFast conv branch.

    See Christoph Feichtenhofer et al.
    SlowFast Networks for Video Recognition.
    """
    def __init__(self, cfg, block_idx):
        super(SlowfastBranch, self).__init__(cfg, block_idx)

    def _construct_simple_block(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters,
            kernel_size     = self.kernel_size,
            stride          = self.stride,
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = nn.Conv3d(
            in_channels     = self.num_filters,
            out_channels    = self.num_filters,
            kernel_size     = self.kernel_size,
            stride          = 1,
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_bn.transform_final_bn = True
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [3, 1, 1] if self.cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[self.stage_id] else 1,
            stride          = 1,
            padding         = [1, 0, 0] if self.cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[self.stage_id] else 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = self.kernel_size,
            stride          = self.stride,
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.c_bn.transform_final_bn = True
    
    def forward(self, x):
        if self.transformation == 'simple_block':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x)
            x = self.b_bn(x)
            return x
        elif self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x