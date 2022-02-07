#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" TAda Branch. """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

from models.base.base_blocks import BaseBranch, Base3DStem, BaseHead
from models.base.base_blocks import BRANCH_REGISTRY
from models.module_zoo.ops.tadaconv import RouteFuncMLP, TAdaConv2d

@BRANCH_REGISTRY.register()
class TAdaConvBlockAvgPool(BaseBranch):
    """
    The TAdaConv branch with average pooling as the feature aggregation scheme.

    For details, see
    Ziyuan Huang, Shiwei Zhang, Liang Pan, Zhiwu Qing, Mingqian Tang, Ziwei Liu, and Marcelo H. Ang Jr.
    "TAda! Temporally-Adaptive Convolutions for Video Understanding."
    
    """
    def __init__(self, cfg, block_idx):
        super(TAdaConvBlockAvgPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TAdaConv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = RouteFuncMLP(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_R,
            kernels=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_K,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)

        self.b_avgpool = nn.AvgPool3d(
            kernel_size=[
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_K[0],
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_K[1],
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_K[2]
            ],
            stride=1,
            padding=[
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_K[0]//2,
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_K[1]//2,
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_K[2]//2
            ],
        )
        self.b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_avgpool_bn.skip_init=True
        self.b_avgpool_bn.weight.data.zero_()
        self.b_avgpool_bn.bias.data.zero_()
        
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
    
    def forward(self, x):
        if self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x