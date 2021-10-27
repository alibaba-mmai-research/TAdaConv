#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" R2Plus1D stem. """ 

import math
import torch
import torch.nn as nn

from models.base.base_blocks import Base3DStem
from models.base.base_blocks import STEM_REGISTRY

@STEM_REGISTRY.register()
class R2Plus1DStem(Base3DStem):
    """
    R(2+1)D Stem.
    """
    def __init__(
        self, 
        cfg
    ):
        super(R2Plus1DStem, self).__init__(cfg)
        
    def _construct_block(
        self, 
        cfg,
        dim_in, 
        num_filters, 
        kernel_sz,
        stride,
        bn_eps=1e-5,
        bn_mmt=0.1
    ):
        
        mid_dim = int(
            math.floor((kernel_sz[0] * kernel_sz[1] * kernel_sz[2] * dim_in * num_filters) / \
                       (kernel_sz[1] * kernel_sz[2] * dim_in + kernel_sz[0] * num_filters)))

        self.a1 = nn.Conv3d(
            in_channels     = dim_in,
            out_channels    = mid_dim,
            kernel_size     = [1, kernel_sz[1], kernel_sz[2]],
            stride          = [1, stride[1], stride[2]],
            padding         = [0, kernel_sz[1]//2, kernel_sz[2]//2],
            bias            = False
        )
        self.a1_bn = nn.BatchNorm3d(mid_dim, eps=bn_eps, momentum=bn_mmt)
        self.a1_relu = nn.ReLU(inplace=True)

        self.a2 = nn.Conv3d(
            in_channels     = mid_dim,
            out_channels    = num_filters,
            kernel_size     = [kernel_sz[0], 1, 1],
            stride          = [stride[0], 1, 1],
            padding         = [kernel_sz[0]//2, 0, 0],
            bias            = False
        )
        self.a2_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a2_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.a1(x)
        x = self.a1_bn(x)
        x = self.a1_relu(x)

        x = self.a2(x)
        x = self.a2_bn(x)
        x = self.a2_relu(x)
        return x