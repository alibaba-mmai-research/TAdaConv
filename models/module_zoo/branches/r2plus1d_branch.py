#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" R2Plus1D branch. """ 

import math
import torch
import torch.nn as nn

from models.base.base_blocks import BaseBranch
from models.base.base_blocks import BRANCH_REGISTRY

@BRANCH_REGISTRY.register()
class R2Plus1DBranch(BaseBranch):
    """
    The R(2+1)D Branch. 

    See Du Tran et al.
    A Closer Look at Spatiotemporal Convoluitions for Action Recognition.
    """
    def __init__(self, cfg, block_idx):
        """
        Args: 
            cfg              (Config): global config object. 
            block_idx        (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(R2Plus1DBranch, self).__init__(cfg, block_idx)

    def _construct_simple_block(self):
        mid_dim = int(
            math.floor((self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.dim_in * self.num_filters) / \
                       (self.kernel_size[1] * self.kernel_size[2] * self.dim_in + self.kernel_size[0] * self.num_filters)))

        self.a1 = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = mid_dim,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.a1_bn = nn.BatchNorm3d(mid_dim, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a1_relu = nn.ReLU(inplace=True)

        self.a2 = nn.Conv3d(
            in_channels     = mid_dim,
            out_channels    = self.num_filters,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.a2_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a2_relu = nn.ReLU(inplace=True)

        mid_dim = int(
            math.floor((self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.num_filters * self.num_filters) / \
                       (self.kernel_size[1] * self.kernel_size[2] * self.num_filters + self.kernel_size[0] * self.num_filters)))

        self.b1 = nn.Conv3d(
            in_channels     = self.num_filters,
            out_channels    = mid_dim,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = 1,
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b1_bn = nn.BatchNorm3d(mid_dim, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b1_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = mid_dim,
            out_channels    = self.num_filters,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = 1,
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b1 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b1_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b1_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if self.transformation == 'simple_block':
            x = self.a1(x)
            x = self.a1_bn(x)
            x = self.a1_relu(x)
            
            x = self.a2(x)
            x = self.a2_bn(x)
            x = self.a2_relu(x)

            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)

            x = self.b2(x)
            x = self.b2_bn(x)
            return x
        elif self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)

            x = self.b2(x)
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

